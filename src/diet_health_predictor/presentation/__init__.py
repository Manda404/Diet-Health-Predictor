"""
Presentation Layer - API and User Interface
=============================================

The Presentation layer contains:
- Interfaces with the outside world
- Controllers/endpoints
- User-facing functions
- Formatting of responses

This layer depends on Application and Infrastructure but not vice versa.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from diet_health_predictor.application import (
    AnalyzeHealthStatsUseCase,
    CompareModelsUseCase,
    CrossValidateModelUseCase,
    CrossValidationResult,
    LoadHealthDietDataUseCase,
    ModelTrainingResult,
    ModelType,
    PreprocessDataUseCase,
    PreprocessingResult,
    TrainModelUseCase,
    save_best_model,
)
from diet_health_predictor.config import get_settings
from diet_health_predictor.infrastructure import HealthDietDataLoader

logger = logging.getLogger(__name__)


class HealthDietAPI:
    """
    Main API for interacting with the health diet prediction system.

    This is the entry point for external consumers of the package.
    """

    def __init__(self):
        """Initialize the API with configuration"""
        self.settings = get_settings()
        self.data_path = Path(self.settings.data.raw_data_path)
        logger.info(f"Initialized HealthDietAPI with environment: {self.settings.environment}")

    def load_data(self):
        """
        Load health diet data from configured path.

        Returns:
            List of DietAssessment objects
        """
        logger.info("Loading data via API...")
        loader = HealthDietDataLoader(str(self.data_path))
        use_case = LoadHealthDietDataUseCase(loader)

        return use_case.execute(sample_size=self.settings.data.sample_size)

    def get_health_statistics(self, assessments=None):
        """
        Analyze health statistics.

        Args:
            assessments: List of DietAssessment objects. If None, loads from data.

        Returns:
            Dictionary with statistics
        """
        if assessments is None:
            assessments = self.load_data()

        use_case = AnalyzeHealthStatsUseCase()
        return use_case.execute(assessments)

    def preprocess_data(self) -> PreprocessingResult:
        """
        Run the Phase 2 pipeline: clean, engineer features, split, encode/scale,
        and persist train/test data + fitted transformer to the configured
        processed data path.

        Returns:
            PreprocessingResult with X_train, X_test, y_train, y_test, feature
            names, and the path to the saved transformer.
        """
        logger.info("Preprocessing data via API...")
        loader = HealthDietDataLoader(str(self.data_path))
        use_case = PreprocessDataUseCase(
            data_loader=loader,
            output_dir=self.settings.data.processed_data_path,
            target_column=self.settings.data.target_column,
            test_size=self.settings.model.test_size,
            random_state=self.settings.model.random_state,
        )
        return use_case.execute(sample_size=self.settings.data.sample_size)

    def train_model(
        self,
        model_type: ModelType,
        preprocessing_result: PreprocessingResult,
        hyperparameters: Optional[dict] = None,
    ) -> ModelTrainingResult:
        """
        Run the Phase 3 pipeline: train one boosting model on the output of
        `preprocess_data()`, evaluate it, and persist the fitted model +
        label encoder to the configured models directory.

        Two ways to control hyperparameters:
        - Omit `hyperparameters` (or pass None): they come from
          `settings.model.{model_type}_params` (YAML); an empty/missing
          section falls back to the wrapper's built-in defaults.
        - Pass `hyperparameters` explicitly: it *replaces* the YAML section
          entirely for this call -- useful for quick experiments (e.g. in a
          notebook) without editing the config file.

        Args:
            model_type: Which model to train (ModelType.XGBOOST or ModelType.CATBOOST).
            preprocessing_result: Output of `preprocess_data()`.
            hyperparameters: Optional explicit hyperparameters overriding the
                YAML-configured ones for this call only.

        Returns:
            ModelTrainingResult with the fitted model, evaluation metrics,
            train/eval curves, and artifact paths.
        """
        logger.info(f"Training {model_type.value} via API...")
        resolved_hyperparameters = (
            hyperparameters
            if hyperparameters is not None
            else self._hyperparameters_for(model_type)
        )
        use_case = TrainModelUseCase(
            model_type=model_type,
            output_dir=self.settings.model.models_output_dir,
            random_state=self.settings.model.random_state,
            **resolved_hyperparameters,
        )
        return use_case.execute(
            preprocessing_result.X_train,
            preprocessing_result.y_train,
            preprocessing_result.X_test,
            preprocessing_result.y_test,
        )

    def cross_validate_model(
        self,
        model_type: ModelType,
        preprocessing_result: PreprocessingResult,
        n_splits: int = 5,
    ) -> CrossValidationResult:
        """
        Stratified k-fold cross-validation for one model type, using all rows
        from `preprocessing_result` (train + test recombined) -- a sanity
        check on how stable a model's metrics are before committing to the
        single train/test split `train_model()` persists.

        Args:
            model_type: Which model to validate.
            preprocessing_result: Output of `preprocess_data()`.
            n_splits: Number of stratified folds.

        Returns:
            CrossValidationResult with per-fold metrics and their mean/std.
        """
        logger.info(f"Cross-validating {model_type.value} via API...")
        X = pd.concat(
            [preprocessing_result.X_train, preprocessing_result.X_test], ignore_index=True
        )
        y = pd.concat(
            [preprocessing_result.y_train, preprocessing_result.y_test], ignore_index=True
        )

        use_case = CrossValidateModelUseCase(
            model_type=model_type,
            n_splits=n_splits,
            random_state=self.settings.model.random_state,
            **self._hyperparameters_for(model_type),
        )
        return use_case.execute(X, y)

    def compare_models(
        self, preprocessing_result: PreprocessingResult
    ) -> dict[ModelType, ModelTrainingResult]:
        """
        Train every registered `ModelType` on the same train/test split, each
        with its own hyperparameters from settings, so their metrics can be
        compared directly. See `diet_health_predictor.application.best_model()`
        to pick a winner by a chosen metric.
        """
        logger.info("Comparing models via API...")
        use_case = CompareModelsUseCase(
            output_dir=self.settings.model.models_output_dir,
            random_state=self.settings.model.random_state,
            hyperparameters_by_model={
                model_type: self._hyperparameters_for(model_type) for model_type in ModelType
            },
        )
        return use_case.execute(
            preprocessing_result.X_train,
            preprocessing_result.y_train,
            preprocessing_result.X_test,
            preprocessing_result.y_test,
        )

    def select_best_model(
        self,
        comparison: dict[ModelType, ModelTrainingResult],
        metric: Optional[str] = None,
    ) -> ModelType:
        """
        Pick the best model from a `compare_models()` result and persist its
        artifacts to `settings.model.models_output_dir/best/` -- a canonical
        location a downstream consumer can load "the" model from without
        knowing which `ModelType` won.

        Args:
            comparison: Output of `compare_models()`.
            metric: Metric to compare by (one of `ModelTrainingResult.metrics`'
                keys). Defaults to `settings.model.selection_metric` ("mcc").

        Returns:
            The winning `ModelType`.
        """
        resolved_metric = metric or self.settings.model.selection_metric
        logger.info(f"Selecting best model via API (metric={resolved_metric})...")
        return save_best_model(
            comparison, self.settings.model.models_output_dir, metric=resolved_metric
        )

    def _hyperparameters_for(self, model_type: ModelType) -> dict:
        """settings.model.{xgboost,catboost}_params for the given model type."""
        params_by_model: dict[ModelType, dict] = {
            ModelType.XGBOOST: self.settings.model.xgboost_params,
            ModelType.CATBOOST: self.settings.model.catboost_params,
        }
        return params_by_model[model_type]

    def print_summary(self):
        """Print a summary of the dataset to console"""
        logger.info("Generating summary...")

        try:
            loader = HealthDietDataLoader(str(self.data_path))
            summary = loader.get_summary()

            print("\n" + "=" * 60)
            print("DIET-HEALTH-PREDICTOR - DATA SUMMARY")
            print("=" * 60)
            print(f"Environment: {self.settings.environment.upper()}")
            print(f"Total Records: {summary['total_records']}")
            print(f"Total Columns: {summary['total_columns']}")
            print(f"Age Range: {summary['age_range']} years")
            print(f"BMI Range: {summary['bmi_range']}")
            print("\nHealth Status Distribution:")
            for status, count in summary["health_status_distribution"].items():
                print(f"  - {status}: {count}")
            print("\nDiet Type Distribution:")
            for diet, count in summary["diet_type_distribution"].items():
                print(f"  - {diet}: {count}")
            print("=" * 60 + "\n")

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            print(f"Error: {e}")


def main():
    """Entry point for the package"""
    api = HealthDietAPI()
    api.print_summary()


if __name__ == "__main__":
    main()

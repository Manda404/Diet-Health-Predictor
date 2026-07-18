"""
Integration test: Phase 2 (PreprocessDataUseCase) feeding into Phase 3's
cross-validation and model comparison use cases, against the shared mock
dataset.

The mock dataset's smallest class has only 2 members after cleaning (see
tests/fixtures/README.md), so cross-validation here uses n_splits=2 --
StratifiedKFold requires every class to have at least n_splits members.
"""

import pandas as pd
import pytest

from diet_health_predictor.application.feature_engineering import PreprocessDataUseCase
from diet_health_predictor.application.model_evaluation import (
    CompareModelsUseCase,
    CrossValidateModelUseCase,
    best_model,
)
from diet_health_predictor.application.model_training import ModelType
from diet_health_predictor.infrastructure import HealthDietDataLoader

pytestmark = pytest.mark.integration

_QUIET_KWARGS = {ModelType.CATBOOST: {"verbose": False}}


@pytest.fixture
def preprocessing_result(mock_csv_path, tmp_path):
    loader = HealthDietDataLoader(str(mock_csv_path))
    use_case = PreprocessDataUseCase(
        data_loader=loader,
        output_dir=str(tmp_path / "processed"),
        target_column="Health_Status",
        test_size=0.25,
        random_state=42,
    )
    return use_case.execute()


@pytest.mark.parametrize("model_type", list(ModelType))
class TestCrossValidateModelUseCasePipeline:
    def test_cross_validates_on_real_preprocessed_output(self, model_type, preprocessing_result):
        X = pd.concat(
            [preprocessing_result.X_train, preprocessing_result.X_test], ignore_index=True
        )
        y = pd.concat(
            [preprocessing_result.y_train, preprocessing_result.y_test], ignore_index=True
        )

        use_case = CrossValidateModelUseCase(
            model_type=model_type,
            n_splits=2,
            random_state=42,
            **_QUIET_KWARGS.get(model_type, {}),
        )
        result = use_case.execute(X, y)

        assert len(result.fold_metrics) == 2
        for metric_name in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            assert 0.0 <= result.mean_metrics[metric_name] <= 1.0
        assert -1.0 <= result.mean_metrics["mcc"] <= 1.0


class TestCompareModelsUseCasePipeline:
    def test_compares_both_models_on_real_preprocessed_output(self, preprocessing_result, tmp_path):
        use_case = CompareModelsUseCase(
            output_dir=str(tmp_path / "models"),
            random_state=42,
            hyperparameters_by_model={ModelType.CATBOOST: {"verbose": False}},
        )

        results = use_case.execute(
            preprocessing_result.X_train,
            preprocessing_result.y_train,
            preprocessing_result.X_test,
            preprocessing_result.y_test,
        )

        assert set(results.keys()) == set(ModelType)
        winner = best_model(results, metric="f1_macro")
        assert winner in ModelType

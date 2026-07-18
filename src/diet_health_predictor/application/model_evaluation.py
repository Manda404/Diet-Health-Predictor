"""
Application Layer - Cross-Validation & Model Comparison Use Cases
====================================================================

Two use cases that build on `TrainModelUseCase` / `MODEL_WRAPPER_REGISTRY`:

- `CrossValidateModelUseCase` -- stratified k-fold cross-validation for a
  single model type, reporting mean/std metrics across folds (a single
  train/test split, as `TrainModelUseCase` does, can be noisy on a dataset
  this small).
- `CompareModelsUseCase` -- trains every registered `ModelType` on the same
  train/test split and returns their results side by side.

Our wrappers aren't scikit-learn estimators (no `get_params`/`set_params`),
so `sklearn.model_selection.cross_val_score` can't drive them directly --
cross-validation is implemented here as an explicit fold loop instead, with a
freshly-constructed wrapper per fold (never re-fitting one over the previous
fold's state).
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from diet_health_predictor.application.model_training import (
    MODEL_WRAPPER_REGISTRY,
    ModelTrainingResult,
    ModelType,
    TrainModelUseCase,
    compute_classification_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class CrossValidationResult:
    """Per-fold metrics plus their mean/std across all folds."""

    model_type: ModelType
    n_splits: int
    fold_metrics: list[dict] = field(default_factory=list)
    mean_metrics: dict = field(default_factory=dict)
    std_metrics: dict = field(default_factory=dict)


class CrossValidateModelUseCase:
    """
    Use Case: stratified k-fold cross-validation for one model type.

    Unlike `TrainModelUseCase`, this does not persist a model to disk -- its
    only output is the metric distribution across folds, used to judge how
    stable a model's performance is before committing to a single train/test
    split for the real, persisted training run.
    """

    _METRIC_KEYS = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc")

    def __init__(
        self,
        model_type: ModelType,
        n_splits: int = 5,
        random_state: int = 42,
        **hyperparameters: Any,
    ):
        self.model_type = model_type
        self.n_splits = n_splits
        self.random_state = random_state
        self.hyperparameters = hyperparameters

    def execute(self, X: pd.DataFrame, y: pd.Series) -> CrossValidationResult:
        logger.info(f"Cross-validating {self.model_type.value} ({self.n_splits} folds)...")

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        wrapper_cls = MODEL_WRAPPER_REGISTRY[self.model_type]
        splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        fold_metrics = []
        for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X, y_encoded)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            model = wrapper_cls(random_state=self.random_state, **self.hyperparameters)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            metrics = compute_classification_metrics(y_val, y_pred, label_encoder)
            fold_metrics.append({k: metrics[k] for k in self._METRIC_KEYS})
            logger.info(f"  fold {fold_index + 1}/{self.n_splits}: {fold_metrics[-1]}")

        mean_metrics = {k: float(np.mean([f[k] for f in fold_metrics])) for k in self._METRIC_KEYS}
        std_metrics = {k: float(np.std([f[k] for f in fold_metrics])) for k in self._METRIC_KEYS}

        return CrossValidationResult(
            model_type=self.model_type,
            n_splits=self.n_splits,
            fold_metrics=fold_metrics,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
        )


class CompareModelsUseCase:
    """
    Use Case: train every registered `ModelType` on the same train/test
    split via `TrainModelUseCase`, so their metrics can be compared directly.
    """

    def __init__(
        self,
        output_dir: str,
        random_state: int = 42,
        hyperparameters_by_model: dict[ModelType, dict] | None = None,
    ):
        self.output_dir = output_dir
        self.random_state = random_state
        self.hyperparameters_by_model = hyperparameters_by_model or {}

    def execute(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[ModelType, ModelTrainingResult]:
        results = {}
        for model_type in ModelType:
            hyperparameters = self.hyperparameters_by_model.get(model_type, {})
            use_case = TrainModelUseCase(
                model_type=model_type,
                output_dir=self.output_dir,
                random_state=self.random_state,
                **hyperparameters,
            )
            results[model_type] = use_case.execute(X_train, y_train, X_test, y_test)
        return results


def best_model(results: dict[ModelType, ModelTrainingResult], metric: str = "mcc") -> ModelType:
    """
    Pick the `ModelType` with the highest `metric` from a `CompareModelsUseCase`
    result. Defaults to Matthews Correlation Coefficient (`mcc`) -- unlike
    accuracy or the macro-averaged metrics, it stays meaningful under class
    imbalance, which makes it the more trustworthy tie-breaker here.
    """
    return max(results, key=lambda model_type: results[model_type].metrics[metric])


def save_best_model(
    results: dict[ModelType, ModelTrainingResult],
    output_dir: str,
    metric: str = "mcc",
) -> ModelType:
    """
    Pick the best model from a `CompareModelsUseCase` result (see `best_model()`)
    and persist its artifacts to `output_dir/best/` -- a canonical location a
    downstream consumer (e.g. a future Phase 4 API) can load from without
    needing to know which `ModelType` actually won.

    Both models are already saved under their own `output_dir/<model_type>/`
    by `TrainModelUseCase`; this additionally copies the winner's `model.joblib`
    and `label_encoder.joblib` into `output_dir/best/`, alongside a
    `metadata.json` recording which model won and by what metric/value.

    Returns:
        The winning `ModelType`.
    """
    winner = best_model(results, metric=metric)
    winning_result = results[winner]

    best_dir = Path(output_dir) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winning_result.model_path, best_dir / "model.joblib")
    shutil.copy2(winning_result.label_encoder_path, best_dir / "label_encoder.joblib")

    metadata = {
        "model_type": winner.value,
        "selection_metric": metric,
        "selection_metric_value": winning_result.metrics[metric],
    }
    (best_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    value = metadata["selection_metric_value"]
    logger.info(f"Best model ({winner.value}, {metric}={value:.4f}) saved to {best_dir}")

    return winner

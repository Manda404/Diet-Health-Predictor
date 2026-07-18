"""
Application Layer - Model Training Use Case
==============================================

Trains a single classifier (chosen via `ModelType`) on already-preprocessed
features -- the `PreprocessingResult` produced by Phase 2's
`PreprocessDataUseCase` -- evaluates it on the held-out test set, and persists
the fitted model + label encoder to disk.

Each model type is trained independently, one `TrainModelUseCase` call per
model. See `model_evaluation.py` for cross-validation and the
`CompareModelsUseCase` that trains every registered model type together.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from diet_health_predictor.infrastructure.models import (
    BaseModelWrapper,
    CatBoostWrapper,
    XGBoostWrapper,
)

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """The boosting classifiers this project trains on Health_Status."""

    XGBOOST = "xgboost"
    CATBOOST = "catboost"


MODEL_WRAPPER_REGISTRY: dict[ModelType, type[BaseModelWrapper]] = {
    ModelType.XGBOOST: XGBoostWrapper,
    ModelType.CATBOOST: CatBoostWrapper,
}


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_encoder: LabelEncoder
) -> dict:
    """
    Shared by `TrainModelUseCase` and `CrossValidateModelUseCase` so a single
    train/test evaluation and a cross-validation fold report identical metrics.
    """
    # Pin `labels` to every class the encoder knows about -- otherwise a
    # class absent from this particular y_true/y_pred batch (common with
    # small test sets) would silently shrink the confusion matrix below
    # len(class_labels), making the two inconsistent to zip/plot together.
    all_class_indices = range(len(label_encoder.classes_))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        # Matthews Correlation Coefficient: a single balanced-summary score
        # (in [-1, 1], 0 = random) that stays meaningful under class
        # imbalance, unlike accuracy -- a useful cross-check alongside the
        # macro-averaged metrics above.
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(all_class_indices)
        ).tolist(),
        "class_labels": list(label_encoder.classes_),
    }


@dataclass
class ModelTrainingResult:
    """Output of training + evaluating a single model."""

    model_type: ModelType
    model: BaseModelWrapper
    metrics: dict
    evals_result: dict
    model_path: str
    label_encoder_path: str


class TrainModelUseCase:
    """
    Use Case: train one model type on preprocessed train/test data, evaluate
    it, and persist the fitted model + label encoder.
    """

    def __init__(
        self,
        model_type: ModelType,
        output_dir: str,
        random_state: int = 42,
        **hyperparameters: Any,
    ):
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.hyperparameters = hyperparameters

    def execute(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> ModelTrainingResult:
        logger.info(f"Training {self.model_type.value}...")

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        wrapper_cls = MODEL_WRAPPER_REGISTRY[self.model_type]
        model = wrapper_cls(random_state=self.random_state, **self.hyperparameters)
        # The held-out test set doubles as the eval_set for train/eval curve
        # tracking. Note this means the "validation" curve isn't a truly
        # untouched holdout if you were to use it for early stopping -- fine
        # for the diagnostic-curve use case here, but worth knowing.
        model.fit(X_train, y_train_encoded, X_test, y_test_encoded)

        y_pred = model.predict(X_test)
        metrics = compute_classification_metrics(y_test_encoded, y_pred, label_encoder)
        evals_result = model.get_evals_result()
        logger.info(f"{self.model_type.value} metrics: {metrics}")

        model_dir = self.output_dir / self.model_type.value
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(model_dir / "model.joblib")
        label_encoder_path = str(model_dir / "label_encoder.joblib")
        model.save(model_path)
        joblib.dump(label_encoder, label_encoder_path)

        return ModelTrainingResult(
            model_type=self.model_type,
            model=model,
            metrics=metrics,
            evals_result=evals_result,
            model_path=model_path,
            label_encoder_path=label_encoder_path,
        )

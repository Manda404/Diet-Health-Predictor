"""
Application Layer - Single-Record Prediction Use Case
=======================================================

Phase 4: takes one raw health/diet record and runs it through the exact same
pipeline used to build the training data -- `DataCleaner.transform()` (fit on
the training split), `FeatureEngineer.transform()`, then
`FeatureTransformer.transform()` (also fit on the training split) -- before
handing it to the persisted best model.

Reusing the training-fit `DataCleaner`/`FeatureTransformer` (rather than
re-deriving impute values/scaling from the single incoming record) is what
keeps this consistent with train/test and avoids leaking wall-clock
request-time statistics into the transformation.
"""

import logging
from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd

from diet_health_predictor.infrastructure import (
    BaseModelWrapper,
    DataCleaner,
    FeatureEngineer,
    FeatureTransformer,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Predicted `Health_Status` plus the model's per-class probabilities."""

    predicted_health_status: str
    probabilities: dict[str, float]


class PredictHealthStatusUseCase:
    """
    Use Case: predict `Health_Status` for a single raw record.

    Loads the fitted `DataCleaner`/`FeatureTransformer` (persisted by
    `PreprocessDataUseCase`) and the best model + label encoder (persisted by
    `save_best_model()`), then replays the same clean -> engineer -> encode/
    scale pipeline used during training on the one incoming record.
    """

    def __init__(
        self,
        cleaner_path: str,
        transformer_path: str,
        model_path: str,
        label_encoder_path: str,
    ):
        self.cleaner = DataCleaner.load(cleaner_path)
        self.transformer = FeatureTransformer.load(transformer_path)
        self.model = BaseModelWrapper.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def execute(self, raw_record: dict[str, Any]) -> PredictionResult:
        df = pd.DataFrame([raw_record])

        clean_df = self.cleaner.transform(df)
        engineered_df = FeatureEngineer().transform(clean_df)
        X = self.transformer.transform(engineered_df)

        probabilities = self.model.predict_proba(X)[0]
        predicted_index = probabilities.argmax()
        predicted_label = self.label_encoder.inverse_transform([predicted_index])[0]

        return PredictionResult(
            predicted_health_status=str(predicted_label),
            probabilities={
                str(label): float(probability)
                for label, probability in zip(self.label_encoder.classes_, probabilities)
            },
        )

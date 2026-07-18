"""
Unit tests for diet_health_predictor.application.prediction

Builds a small end-to-end pipeline (clean -> engineer -> encode/scale -> fit
a model) on synthetic data, persists each collaborator to `tmp_path` exactly
as `PreprocessDataUseCase`/`TrainModelUseCase`/`save_best_model()` would, then
drives `PredictHealthStatusUseCase` against it -- the same artifacts a real
deployment would load, just built inline instead of via the full pipeline
(covered separately by the integration test).
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from diet_health_predictor.application.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_NUMERIC_COLUMNS,
)
from diet_health_predictor.application.prediction import PredictHealthStatusUseCase
from diet_health_predictor.infrastructure import DataCleaner, FeatureEngineer, FeatureTransformer
from diet_health_predictor.infrastructure.models import XGBoostWrapper

pytestmark = pytest.mark.unit


@pytest.fixture
def raw_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 40
    genders = rng.choice(["Male", "Female", "Other"], size=n)
    height_cm = rng.uniform(150, 195, size=n)
    weight_kg = rng.uniform(45, 110, size=n)
    return pd.DataFrame(
        {
            "Person_ID": [f"P{i:04d}" for i in range(n)],
            "Age": rng.randint(18, 70, size=n),
            "Gender": genders,
            "Height_cm": height_cm,
            "Weight_kg": weight_kg,
            "BMI": weight_kg / (height_cm / 100) ** 2,
            "Activity_Level": rng.choice(
                ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Athlete"],
                size=n,
            ),
            "Daily_Calorie_Requirement": rng.uniform(1800, 2800, size=n),
            "Daily_Calorie_Consumed": rng.uniform(1800, 2800, size=n),
            "Protein_Intake_g": rng.uniform(50, 160, size=n),
            "Carbohydrate_Intake_g": rng.uniform(150, 320, size=n),
            "Fat_Intake_g": rng.uniform(40, 90, size=n),
            "Water_Intake_Liters": rng.uniform(1.0, 3.5, size=n),
            "Diet_Type": rng.choice(
                ["Balanced", "Keto", "Vegetarian", "Vegan", "High Protein", "Mediterranean"],
                size=n,
            ),
            "Health_Status": rng.choice(["Healthy", "Overweight", "Obese", "Underweight"], size=n),
        }
    )


@pytest.fixture
def pipeline_paths(raw_df, tmp_path) -> dict:
    """Fits every pipeline collaborator on `raw_df` and persists it to `tmp_path`."""
    cleaner = DataCleaner(outlier_columns=RAW_NUMERIC_COLUMNS)
    clean_df = cleaner.fit_transform(cleaner.drop_duplicates(raw_df))

    engineered_df = FeatureEngineer().transform(clean_df)

    transformer = FeatureTransformer(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    X = transformer.fit_transform(engineered_df)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(engineered_df["Health_Status"])

    model = XGBoostWrapper(random_state=42)
    model.fit(X, y_encoded)

    cleaner_path = str(tmp_path / "data_cleaner.joblib")
    transformer_path = str(tmp_path / "feature_transformer.joblib")
    model_path = str(tmp_path / "model.joblib")
    label_encoder_path = str(tmp_path / "label_encoder.joblib")

    cleaner.save(cleaner_path)
    transformer.save(transformer_path)
    model.save(model_path)
    joblib.dump(label_encoder, label_encoder_path)

    return {
        "cleaner_path": cleaner_path,
        "transformer_path": transformer_path,
        "model_path": model_path,
        "label_encoder_path": label_encoder_path,
        "classes": set(label_encoder.classes_),
    }


class TestPredictHealthStatusUseCase:
    def test_execute_predicts_a_known_health_status_class(self, raw_df, pipeline_paths):
        use_case = PredictHealthStatusUseCase(
            cleaner_path=pipeline_paths["cleaner_path"],
            transformer_path=pipeline_paths["transformer_path"],
            model_path=pipeline_paths["model_path"],
            label_encoder_path=pipeline_paths["label_encoder_path"],
        )
        record = raw_df.drop(columns=["Person_ID", "Health_Status"]).iloc[0].to_dict()

        result = use_case.execute(record)

        assert result.predicted_health_status in pipeline_paths["classes"]

    def test_execute_returns_probabilities_for_every_known_class_summing_to_one(
        self, raw_df, pipeline_paths
    ):
        use_case = PredictHealthStatusUseCase(
            cleaner_path=pipeline_paths["cleaner_path"],
            transformer_path=pipeline_paths["transformer_path"],
            model_path=pipeline_paths["model_path"],
            label_encoder_path=pipeline_paths["label_encoder_path"],
        )
        record = raw_df.drop(columns=["Person_ID", "Health_Status"]).iloc[1].to_dict()

        result = use_case.execute(record)

        assert set(result.probabilities) == pipeline_paths["classes"]
        assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)

    def test_execute_does_not_require_person_id_or_target_column(self, raw_df, pipeline_paths):
        """The whole point of the DataCleaner fix: inference records lack columns
        that were present when the cleaner was fit on the full raw training data."""
        use_case = PredictHealthStatusUseCase(
            cleaner_path=pipeline_paths["cleaner_path"],
            transformer_path=pipeline_paths["transformer_path"],
            model_path=pipeline_paths["model_path"],
            label_encoder_path=pipeline_paths["label_encoder_path"],
        )
        record = raw_df.drop(columns=["Person_ID", "Health_Status"]).iloc[0].to_dict()
        assert "Person_ID" not in record and "Health_Status" not in record

        use_case.execute(record)  # must not raise

"""Unit tests for diet_health_predictor.application.model_training"""

import numpy as np
import pandas as pd
import pytest

from diet_health_predictor.application.model_training import (
    MODEL_WRAPPER_REGISTRY,
    ModelType,
    TrainModelUseCase,
)

pytestmark = pytest.mark.unit

# Neither wrapper hardcodes hyperparameter defaults (only random_state) -- an
# unconfigured CatBoost falls back to the library's own verbose=1, printing a
# line per iteration. Passed here explicitly to keep test output readable.
_QUIET_KWARGS = {ModelType.CATBOOST: {"verbose": False}}


def build_use_case(model_type, output_dir):
    return TrainModelUseCase(
        model_type=model_type,
        output_dir=str(output_dir),
        random_state=42,
        **_QUIET_KWARGS.get(model_type, {}),
    )


@pytest.fixture
def toy_train_test_data():
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame({"a": rng.rand(40), "b": rng.rand(40)})
    y_train = pd.Series(["Healthy", "Overweight", "Obese", "Underweight"] * 10)
    X_test = pd.DataFrame({"a": rng.rand(12), "b": rng.rand(12)})
    y_test = pd.Series(["Healthy", "Overweight", "Obese", "Underweight"] * 3)
    return X_train, y_train, X_test, y_test


class TestModelWrapperRegistry:
    def test_every_model_type_has_a_registered_wrapper(self):
        assert set(MODEL_WRAPPER_REGISTRY.keys()) == set(ModelType)


@pytest.mark.parametrize("model_type", list(ModelType))
class TestTrainModelUseCase:
    def test_execute_returns_metrics_in_expected_ranges(
        self, model_type, toy_train_test_data, tmp_path
    ):
        X_train, y_train, X_test, y_test = toy_train_test_data
        use_case = build_use_case(model_type, tmp_path)

        result = use_case.execute(X_train, y_train, X_test, y_test)

        for metric_name in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            assert 0.0 <= result.metrics[metric_name] <= 1.0
        assert -1.0 <= result.metrics["mcc"] <= 1.0

    def test_execute_reports_all_classes_seen_in_training(
        self, model_type, toy_train_test_data, tmp_path
    ):
        X_train, y_train, X_test, y_test = toy_train_test_data
        use_case = build_use_case(model_type, tmp_path)

        result = use_case.execute(X_train, y_train, X_test, y_test)

        assert set(result.metrics["class_labels"]) == {
            "Healthy",
            "Overweight",
            "Obese",
            "Underweight",
        }

    def test_execute_persists_model_and_label_encoder_to_disk(
        self, model_type, toy_train_test_data, tmp_path
    ):
        X_train, y_train, X_test, y_test = toy_train_test_data
        use_case = build_use_case(model_type, tmp_path)

        result = use_case.execute(X_train, y_train, X_test, y_test)

        assert (tmp_path / model_type.value / "model.joblib").exists()
        assert (tmp_path / model_type.value / "label_encoder.joblib").exists()
        assert result.model_path == str(tmp_path / model_type.value / "model.joblib")

    def test_execute_returns_a_fitted_model_that_can_predict(
        self, model_type, toy_train_test_data, tmp_path
    ):
        X_train, y_train, X_test, y_test = toy_train_test_data
        use_case = build_use_case(model_type, tmp_path)

        result = use_case.execute(X_train, y_train, X_test, y_test)

        predictions = result.model.predict(X_test)
        assert predictions.shape == (len(X_test),)

    def test_execute_tracks_train_and_validation_curves_against_the_test_set(
        self, model_type, toy_train_test_data, tmp_path
    ):
        X_train, y_train, X_test, y_test = toy_train_test_data
        use_case = build_use_case(model_type, tmp_path)

        result = use_case.execute(X_train, y_train, X_test, y_test)

        assert set(result.evals_result.keys()) == {"train", "validation"}
        train_curve = next(iter(result.evals_result["train"].values()))
        validation_curve = next(iter(result.evals_result["validation"].values()))
        assert len(train_curve) == len(validation_curve) > 0

    def test_execute_does_not_invent_hyperparameters_beyond_random_state(
        self, model_type, toy_train_test_data, tmp_path
    ):
        # No n_estimators/iterations passed through build_use_case() -> the
        # underlying model must be built with the library's own defaults.
        X_train, y_train, X_test, y_test = toy_train_test_data
        use_case = build_use_case(model_type, tmp_path)

        result = use_case.execute(X_train, y_train, X_test, y_test)

        assert "n_estimators" not in result.model.hyperparameters
        assert "iterations" not in result.model.hyperparameters

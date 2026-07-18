"""Unit tests for diet_health_predictor.application.model_evaluation"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from diet_health_predictor.application.model_evaluation import (
    CompareModelsUseCase,
    CrossValidateModelUseCase,
    best_model,
    save_best_model,
)
from diet_health_predictor.application.model_training import ModelTrainingResult, ModelType

pytestmark = pytest.mark.unit

# Neither wrapper hardcodes hyperparameter defaults (only random_state) -- an
# unconfigured CatBoost falls back to the library's own verbose=1, printing a
# line per iteration. Passed here explicitly to keep test output readable.
_QUIET_KWARGS = {ModelType.CATBOOST: {"verbose": False}}


@pytest.fixture
def toy_full_dataset():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"a": rng.rand(40), "b": rng.rand(40)})
    y = pd.Series(["Healthy", "Overweight", "Obese", "Underweight"] * 10)
    return X, y


@pytest.fixture
def toy_train_test_data():
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame({"a": rng.rand(40), "b": rng.rand(40)})
    y_train = pd.Series(["Healthy", "Overweight", "Obese", "Underweight"] * 10)
    X_test = pd.DataFrame({"a": rng.rand(12), "b": rng.rand(12)})
    y_test = pd.Series(["Healthy", "Overweight", "Obese", "Underweight"] * 3)
    return X_train, y_train, X_test, y_test


@pytest.mark.parametrize("model_type", list(ModelType))
class TestCrossValidateModelUseCase:
    def test_execute_produces_one_fold_metrics_entry_per_split(self, model_type, toy_full_dataset):
        X, y = toy_full_dataset
        use_case = CrossValidateModelUseCase(
            model_type=model_type, n_splits=5, random_state=42, **_QUIET_KWARGS.get(model_type, {})
        )

        result = use_case.execute(X, y)

        assert result.n_splits == 5
        assert len(result.fold_metrics) == 5

    def test_execute_computes_mean_and_std_across_folds(self, model_type, toy_full_dataset):
        X, y = toy_full_dataset
        use_case = CrossValidateModelUseCase(
            model_type=model_type, n_splits=5, random_state=42, **_QUIET_KWARGS.get(model_type, {})
        )

        result = use_case.execute(X, y)

        for metric_name in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            assert 0.0 <= result.mean_metrics[metric_name] <= 1.0
            assert result.std_metrics[metric_name] >= 0.0
        assert -1.0 <= result.mean_metrics["mcc"] <= 1.0
        assert result.std_metrics["mcc"] >= 0.0

    def test_default_n_splits_is_5(self, model_type):
        use_case = CrossValidateModelUseCase(model_type=model_type)
        assert use_case.n_splits == 5


class TestCompareModelsUseCase:
    def test_execute_trains_every_registered_model_type(self, toy_train_test_data, tmp_path):
        X_train, y_train, X_test, y_test = toy_train_test_data
        use_case = CompareModelsUseCase(
            output_dir=str(tmp_path),
            random_state=42,
            hyperparameters_by_model={ModelType.CATBOOST: {"verbose": False}},
        )

        results = use_case.execute(X_train, y_train, X_test, y_test)

        assert set(results.keys()) == set(ModelType)
        for model_type, result in results.items():
            assert result.model_type == model_type
            assert 0.0 <= result.metrics["accuracy"] <= 1.0

    def test_execute_applies_per_model_hyperparameters(self, toy_train_test_data, tmp_path):
        X_train, y_train, X_test, y_test = toy_train_test_data
        use_case = CompareModelsUseCase(
            output_dir=str(tmp_path),
            random_state=42,
            hyperparameters_by_model={
                ModelType.XGBOOST: {"n_estimators": 7},
                ModelType.CATBOOST: {"iterations": 9, "verbose": False},
            },
        )

        results = use_case.execute(X_train, y_train, X_test, y_test)

        assert results[ModelType.XGBOOST].model._model.n_estimators == 7
        assert results[ModelType.CATBOOST].model._model.get_params()["iterations"] == 9


class TestBestModel:
    def test_picks_the_model_type_with_the_highest_metric(self):
        results = {
            ModelType.XGBOOST: SimpleNamespace(metrics={"f1_macro": 0.80}),
            ModelType.CATBOOST: SimpleNamespace(metrics={"f1_macro": 0.92}),
        }
        assert best_model(results, metric="f1_macro") == ModelType.CATBOOST

    def test_respects_the_requested_metric(self):
        results = {
            ModelType.XGBOOST: SimpleNamespace(metrics={"accuracy": 0.95, "f1_macro": 0.60}),
            ModelType.CATBOOST: SimpleNamespace(metrics={"accuracy": 0.80, "f1_macro": 0.90}),
        }
        assert best_model(results, metric="accuracy") == ModelType.XGBOOST
        assert best_model(results, metric="f1_macro") == ModelType.CATBOOST

    def test_defaults_to_mcc(self):
        # mcc favors CATBOOST here even though f1_macro would favor XGBOOST --
        # this only passes if the default metric is really "mcc".
        results = {
            ModelType.XGBOOST: SimpleNamespace(metrics={"mcc": 0.70, "f1_macro": 0.95}),
            ModelType.CATBOOST: SimpleNamespace(metrics={"mcc": 0.85, "f1_macro": 0.60}),
        }
        assert best_model(results) == ModelType.CATBOOST


class TestSaveBestModel:
    @pytest.fixture
    def fake_results(self, tmp_path):
        """
        Real `ModelTrainingResult`s pointing at dummy artifact files --
        `save_best_model()` only needs `model_path`/`label_encoder_path` to
        exist on disk to copy them; the contents don't need to be a real
        joblib-pickled model for this test.
        """
        results = {}
        for model_type, mcc in [(ModelType.XGBOOST, 0.70), (ModelType.CATBOOST, 0.85)]:
            model_dir = tmp_path / model_type.value
            model_dir.mkdir()
            model_path = model_dir / "model.joblib"
            label_encoder_path = model_dir / "label_encoder.joblib"
            model_path.write_text(f"fake {model_type.value} model")
            label_encoder_path.write_text(f"fake {model_type.value} label encoder")

            results[model_type] = ModelTrainingResult(
                model_type=model_type,
                model=SimpleNamespace(),
                metrics={"mcc": mcc, "f1_macro": 1.0 - mcc},
                evals_result={},
                model_path=str(model_path),
                label_encoder_path=str(label_encoder_path),
            )
        return results

    def test_copies_the_winning_models_artifacts_to_best_dir(self, fake_results, tmp_path):
        winner = save_best_model(fake_results, str(tmp_path), metric="mcc")

        assert winner == ModelType.CATBOOST  # higher mcc (0.85 > 0.70)
        best_dir = tmp_path / "best"
        assert (best_dir / "model.joblib").read_text() == "fake catboost model"
        assert (best_dir / "label_encoder.joblib").read_text() == "fake catboost label encoder"

    def test_writes_metadata_describing_the_winner(self, fake_results, tmp_path):
        save_best_model(fake_results, str(tmp_path), metric="mcc")

        metadata = json.loads((tmp_path / "best" / "metadata.json").read_text())
        assert metadata["model_type"] == "catboost"
        assert metadata["selection_metric"] == "mcc"
        assert metadata["selection_metric_value"] == 0.85

    def test_respects_a_different_metric(self, fake_results, tmp_path):
        # By f1_macro (1 - mcc here), XGBOOST wins instead.
        winner = save_best_model(fake_results, str(tmp_path), metric="f1_macro")
        assert winner == ModelType.XGBOOST

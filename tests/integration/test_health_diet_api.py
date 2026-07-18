"""
Integration test: the Presentation layer entry point a real caller would use
(`HealthDietAPI`), driving the full stack down to Infrastructure.

`Settings` is a process-wide singleton (`get_settings()` caches it after the
first call), so `HealthDietAPI()` here loads the real dev config. Rather than
touch that global, the fields this test needs to redirect (`data_path`,
`settings.data.processed_data_path`, `settings.data.sample_size`) are patched
via `monkeypatch`, which restores them automatically after the test -- no
cross-test pollution.
"""

import json

import pandas as pd
import pytest

from diet_health_predictor.application import ModelType, best_model
from diet_health_predictor.infrastructure import DataCleaner
from diet_health_predictor.presentation import HealthDietAPI

pytestmark = pytest.mark.integration


@pytest.fixture
def api(tmp_path, monkeypatch):
    instance = HealthDietAPI()
    monkeypatch.setattr(instance.settings.data, "processed_data_path", str(tmp_path))
    monkeypatch.setattr(instance.settings.data, "sample_size", None)
    return instance


@pytest.fixture
def clean_csv_path(mock_csv_path, tmp_path):
    """
    load_data()/get_health_statistics() have no cleaning step (the real
    dataset has no gaps), so they're exercised here against a cleaned copy of
    the mock dataset rather than the raw one with its deliberate missing
    Activity_Level value.
    """
    raw_df = pd.read_csv(mock_csv_path)
    cleaner = DataCleaner()
    clean_df = cleaner.fit_transform(cleaner.drop_duplicates(raw_df))
    path = tmp_path / "clean_mock_diet_data.csv"
    clean_df.to_csv(path, index=False)
    return path


class TestHealthDietAPIIntegration:
    def test_load_data_returns_diet_assessments_from_the_configured_path(
        self, api, clean_csv_path, monkeypatch
    ):
        monkeypatch.setattr(api, "data_path", clean_csv_path)
        assessments = api.load_data()
        assert len(assessments) == 16

    def test_get_health_statistics_end_to_end(self, api, clean_csv_path, monkeypatch):
        monkeypatch.setattr(api, "data_path", clean_csv_path)
        stats = api.get_health_statistics()
        assert stats["total_assessments"] == 16
        assert set(stats["health_status_distribution"]) == {
            "Healthy",
            "Overweight",
            "Obese",
            "Underweight",
        }

    def test_preprocess_data_runs_the_full_phase_2_pipeline(
        self, api, mock_csv_path, tmp_path, monkeypatch
    ):
        # preprocess_data() runs DataCleaner itself, so it can take the raw
        # mock CSV directly (duplicate + missing values included).
        monkeypatch.setattr(api, "data_path", mock_csv_path)
        result = api.preprocess_data()

        assert len(result.X_train) + len(result.X_test) == 16  # deduplicated by DataCleaner
        assert (tmp_path / "feature_transformer.joblib").exists()
        assert (tmp_path / "X_train.csv").exists()

    def test_analyze_data_drift_reports_the_stratified_split_as_stable(
        self, api, mock_csv_path, monkeypatch
    ):
        # The mock dataset is tiny (16 rows after cleaning), so a stratified
        # split can only be expected to be roughly balanced, not perfectly
        # drift-free -- this just checks the report shape/columns, not that
        # every feature comes back "none" (unlike the real dataset, where it
        # reliably does; see the notebook for that check).
        monkeypatch.setattr(api, "data_path", mock_csv_path)
        preprocessing_result = api.preprocess_data()

        drift = api.analyze_data_drift(preprocessing_result)

        assert drift.n_features_checked == len(preprocessing_result.feature_names)
        assert list(drift.report.columns) == [
            "feature",
            "psi",
            "ks_statistic",
            "ks_pvalue",
            "drift_severity",
        ]
        assert isinstance(drift.has_major_drift, bool)

    @pytest.mark.parametrize("model_type", list(ModelType))
    def test_train_model_runs_the_full_phase_3_pipeline(
        self, model_type, api, mock_csv_path, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(api, "data_path", mock_csv_path)
        monkeypatch.setattr(api.settings.model, "models_output_dir", str(tmp_path / "models"))

        preprocessing_result = api.preprocess_data()
        result = api.train_model(model_type, preprocessing_result)

        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert set(result.evals_result.keys()) == {"train", "validation"}
        assert (tmp_path / "models" / model_type.value / "model.joblib").exists()

    def test_train_model_uses_yaml_configured_hyperparameters(
        self, api, mock_csv_path, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(api, "data_path", mock_csv_path)
        monkeypatch.setattr(api.settings.model, "models_output_dir", str(tmp_path / "models"))
        monkeypatch.setattr(api.settings.model, "xgboost_params", {"n_estimators": 7})

        preprocessing_result = api.preprocess_data()
        result = api.train_model(ModelType.XGBOOST, preprocessing_result)

        assert result.model._model.n_estimators == 7

    def test_train_model_explicit_hyperparameters_override_yaml(
        self, api, mock_csv_path, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(api, "data_path", mock_csv_path)
        monkeypatch.setattr(api.settings.model, "models_output_dir", str(tmp_path / "models"))
        # YAML says 7; the explicit call-time override below should win instead.
        monkeypatch.setattr(api.settings.model, "xgboost_params", {"n_estimators": 7})

        preprocessing_result = api.preprocess_data()
        result = api.train_model(
            ModelType.XGBOOST, preprocessing_result, hyperparameters={"n_estimators": 13}
        )

        assert result.model._model.n_estimators == 13

    @pytest.mark.parametrize("model_type", list(ModelType))
    def test_cross_validate_model_runs_on_the_full_preprocessed_dataset(
        self, model_type, api, mock_csv_path, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(api, "data_path", mock_csv_path)
        monkeypatch.setattr(api.settings.model, "models_output_dir", str(tmp_path / "models"))

        preprocessing_result = api.preprocess_data()
        # The mock dataset's smallest class has only 2 members after cleaning
        # (see tests/fixtures/README.md); n_splits can't exceed that.
        result = api.cross_validate_model(model_type, preprocessing_result, n_splits=2)

        assert result.n_splits == 2
        assert len(result.fold_metrics) == 2
        assert 0.0 <= result.mean_metrics["accuracy"] <= 1.0

    def test_compare_models_trains_every_model_type(
        self, api, mock_csv_path, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(api, "data_path", mock_csv_path)
        monkeypatch.setattr(api.settings.model, "models_output_dir", str(tmp_path / "models"))

        preprocessing_result = api.preprocess_data()
        results = api.compare_models(preprocessing_result)

        assert set(results.keys()) == set(ModelType)
        assert best_model(results, metric="f1_macro") in ModelType

    def test_select_best_model_persists_the_winner_to_the_best_directory(
        self, api, mock_csv_path, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(api, "data_path", mock_csv_path)
        monkeypatch.setattr(api.settings.model, "models_output_dir", str(tmp_path / "models"))

        preprocessing_result = api.preprocess_data()
        comparison = api.compare_models(preprocessing_result)

        winner = api.select_best_model(comparison)

        assert winner in ModelType
        best_dir = tmp_path / "models" / "best"
        assert (best_dir / "model.joblib").exists()
        assert (best_dir / "label_encoder.joblib").exists()

        metadata = json.loads((best_dir / "metadata.json").read_text())
        assert metadata["model_type"] == winner.value
        assert metadata["selection_metric"] == "mcc"  # settings.model.selection_metric default

    def test_select_best_model_respects_an_explicit_metric_override(
        self, api, mock_csv_path, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(api, "data_path", mock_csv_path)
        monkeypatch.setattr(api.settings.model, "models_output_dir", str(tmp_path / "models"))

        preprocessing_result = api.preprocess_data()
        comparison = api.compare_models(preprocessing_result)

        winner = api.select_best_model(comparison, metric="accuracy")

        assert winner == best_model(comparison, metric="accuracy")

    def test_print_summary_prints_the_dataset_overview(
        self, api, clean_csv_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(api, "data_path", clean_csv_path)
        api.print_summary()

        output = capsys.readouterr().out
        assert "DIET-HEALTH-PREDICTOR - DATA SUMMARY" in output
        assert "Total Records: 16" in output
        assert "Health Status Distribution:" in output

    def test_print_summary_prints_an_error_instead_of_raising_for_a_bad_path(
        self, api, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(api, "data_path", tmp_path / "does_not_exist.csv")
        api.print_summary()  # must not raise

        assert "Error:" in capsys.readouterr().out

    def test_predict_health_status_end_to_end(self, api, mock_csv_path, tmp_path, monkeypatch):
        monkeypatch.setattr(api, "data_path", mock_csv_path)
        monkeypatch.setattr(api.settings.model, "models_output_dir", str(tmp_path / "models"))

        preprocessing_result = api.preprocess_data()
        comparison = api.compare_models(preprocessing_result)
        api.select_best_model(comparison)

        record = {
            "Age": 30,
            "Gender": "Male",
            "Height_cm": 175.0,
            "Weight_kg": 70.0,
            "BMI": 22.86,
            "Activity_Level": "Moderately Active",
            "Daily_Calorie_Requirement": 2200,
            "Daily_Calorie_Consumed": 2100,
            "Protein_Intake_g": 90.0,
            "Carbohydrate_Intake_g": 250.0,
            "Fat_Intake_g": 70.0,
            "Water_Intake_Liters": 2.5,
            "Diet_Type": "Balanced",
        }
        result = api.predict_health_status(record)

        assert result.predicted_health_status in {
            "Healthy",
            "Overweight",
            "Obese",
            "Underweight",
        }
        assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)

    def test_predict_health_status_raises_a_clear_error_before_a_model_is_trained(
        self, api, mock_csv_path, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(api, "data_path", mock_csv_path)
        monkeypatch.setattr(api.settings.model, "models_output_dir", str(tmp_path / "models"))
        api.preprocess_data()  # transformer/cleaner exist, but no model trained yet

        with pytest.raises(FileNotFoundError, match="No best model found"):
            api.predict_health_status({"Age": 30})

    def test_get_best_model_info_reports_the_winning_model_and_metric(
        self, api, mock_csv_path, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(api, "data_path", mock_csv_path)
        monkeypatch.setattr(api.settings.model, "models_output_dir", str(tmp_path / "models"))

        preprocessing_result = api.preprocess_data()
        comparison = api.compare_models(preprocessing_result)
        winner = api.select_best_model(comparison)

        info = api.get_best_model_info()

        assert info["model_type"] == winner.value
        assert info["selection_metric"] == "mcc"

    def test_get_best_model_info_raises_a_clear_error_before_a_model_is_trained(
        self, api, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(api.settings.model, "models_output_dir", str(tmp_path / "models"))

        with pytest.raises(FileNotFoundError, match="No best model found"):
            api.get_best_model_info()

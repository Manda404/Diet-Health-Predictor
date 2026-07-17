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

import pandas as pd
import pytest

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
    clean_df = DataCleaner().clean(raw_df)
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

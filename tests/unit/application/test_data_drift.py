"""Unit tests for diet_health_predictor.application.data_drift"""

import numpy as np
import pandas as pd
import pytest

from diet_health_predictor.application.data_drift import AnalyzeDataDriftUseCase

pytestmark = pytest.mark.unit


@pytest.fixture
def rng():
    return np.random.RandomState(42)


class TestAnalyzeDataDriftUseCase:
    def test_no_drift_case_reports_empty_drifted_features(self, rng):
        reference = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.choice([0, 1], 300)})
        current = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.choice([0, 1], 300)})

        result = AnalyzeDataDriftUseCase().execute(reference, current)

        assert result.n_features_checked == 2
        assert result.drifted_features == []
        assert result.has_major_drift is False

    def test_major_drift_case_is_reported(self, rng):
        reference = pd.DataFrame({"a": rng.normal(0, 1, 300)})
        current = pd.DataFrame({"a": rng.normal(6, 1, 300)})

        result = AnalyzeDataDriftUseCase().execute(reference, current)

        assert result.drifted_features == ["a"]
        assert result.has_major_drift is True

    def test_report_is_a_dataframe_with_one_row_per_feature(self, rng):
        reference = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
        current = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})

        result = AnalyzeDataDriftUseCase().execute(reference, current)

        assert len(result.report) == 2
        assert set(result.report["feature"]) == {"a", "b"}

    def test_constructor_thresholds_are_forwarded_to_the_detector(self, rng):
        reference = pd.DataFrame({"a": rng.normal(0, 1, 300)})
        current = pd.DataFrame({"a": rng.normal(1, 1, 300)})  # PSI here is ~1.0

        lenient_result = AnalyzeDataDriftUseCase(
            psi_moderate_threshold=2.0, psi_major_threshold=5.0
        ).execute(reference, current)

        assert lenient_result.has_major_drift is False
        assert lenient_result.drifted_features == []

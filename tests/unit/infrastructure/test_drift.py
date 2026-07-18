"""Unit tests for diet_health_predictor.infrastructure.drift"""

import numpy as np
import pandas as pd
import pytest

from diet_health_predictor.infrastructure.drift import DriftDetector

pytestmark = pytest.mark.unit


@pytest.fixture
def rng():
    return np.random.RandomState(42)


class TestDriftDetectorAnalyze:
    def test_returns_expected_columns(self, rng):
        reference = pd.DataFrame({"a": rng.normal(0, 1, 200)})
        current = pd.DataFrame({"a": rng.normal(0, 1, 200)})

        report = DriftDetector().analyze(reference, current)

        assert list(report.columns) == [
            "feature",
            "psi",
            "ks_statistic",
            "ks_pvalue",
            "drift_severity",
        ]
        assert len(report) == 1

    def test_identical_distributions_show_no_drift(self, rng):
        reference = pd.DataFrame(
            {"numeric": rng.normal(0, 1, 500), "binary": rng.choice([0, 1], 500)}
        )
        current = pd.DataFrame(
            {"numeric": rng.normal(0, 1, 500), "binary": rng.choice([0, 1], 500)}
        )

        report = DriftDetector().analyze(reference, current)

        assert (report["drift_severity"] == "none").all()
        assert (report["psi"] < 0.1).all()

    def test_heavily_shifted_numeric_distribution_is_flagged_major(self, rng):
        reference = pd.DataFrame({"numeric": rng.normal(0, 1, 500)})
        current = pd.DataFrame({"numeric": rng.normal(5, 1, 500)})  # 5 std devs away

        report = DriftDetector().analyze(reference, current)

        row = report.iloc[0]
        assert row["drift_severity"] == "major"
        assert row["psi"] >= 0.25
        assert row["ks_pvalue"] < 0.05

    def test_shifted_categorical_like_column_is_detected(self, rng):
        # 50/50 split in reference, 90/10 in current -> should show real drift
        reference = pd.DataFrame({"flag": rng.choice([0, 1], 500, p=[0.5, 0.5])})
        current = pd.DataFrame({"flag": rng.choice([0, 1], 500, p=[0.9, 0.1])})

        report = DriftDetector().analyze(reference, current)

        assert report.iloc[0]["drift_severity"] in ("moderate", "major")

    def test_report_is_sorted_by_psi_descending(self, rng):
        reference = pd.DataFrame(
            {
                "stable": rng.normal(0, 1, 500),
                "shifted": rng.normal(0, 1, 500),
            }
        )
        current = pd.DataFrame(
            {
                "stable": rng.normal(0, 1, 500),
                "shifted": rng.normal(4, 1, 500),
            }
        )

        report = DriftDetector().analyze(reference, current)

        assert report.iloc[0]["feature"] == "shifted"
        assert report["psi"].is_monotonic_decreasing

    def test_only_shared_columns_are_compared(self, rng):
        reference = pd.DataFrame({"a": rng.normal(0, 1, 100), "only_in_reference": range(100)})
        current = pd.DataFrame({"a": rng.normal(0, 1, 100), "only_in_current": range(100)})

        report = DriftDetector().analyze(reference, current)

        assert list(report["feature"]) == ["a"]

    def test_custom_thresholds_change_the_severity_verdict(self, rng):
        reference = pd.DataFrame({"numeric": rng.normal(0, 1, 500)})
        current = pd.DataFrame({"numeric": rng.normal(1, 1, 500)})  # PSI here is ~1.0

        lenient = DriftDetector(psi_moderate_threshold=2.0, psi_major_threshold=5.0)
        strict = DriftDetector(psi_moderate_threshold=0.001, psi_major_threshold=0.01)

        lenient_report = lenient.analyze(reference, current)
        strict_report = strict.analyze(reference, current)

        assert lenient_report.iloc[0]["drift_severity"] == "none"
        assert strict_report.iloc[0]["drift_severity"] == "major"

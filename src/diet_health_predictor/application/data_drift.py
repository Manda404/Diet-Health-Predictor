"""
Application Layer - Data Drift Analysis Use Case
====================================================

Wraps `DriftDetector` (PSI + KS test) to answer a specific question: does the
train/test split (or train vs. a future production dataset) show meaningful
distribution drift? A healthy stratified split should show little to none --
if it doesn't, metrics computed on the "test" side are less trustworthy,
since train and test are no longer really comparable populations.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from diet_health_predictor.infrastructure import DriftDetector

logger = logging.getLogger(__name__)


@dataclass
class DataDriftResult:
    """Per-feature drift report plus a quick pass/fail-style summary."""

    report: pd.DataFrame
    n_features_checked: int
    drifted_features: list[str]
    has_major_drift: bool


class AnalyzeDataDriftUseCase:
    """
    Use Case: compare a reference DataFrame (typically train) against a
    current one (typically test) and flag features whose distribution has
    shifted.
    """

    def __init__(
        self,
        buckets: int = 10,
        psi_moderate_threshold: float = 0.1,
        psi_major_threshold: float = 0.25,
    ):
        self.detector = DriftDetector(
            buckets=buckets,
            psi_moderate_threshold=psi_moderate_threshold,
            psi_major_threshold=psi_major_threshold,
        )

    def execute(self, reference: pd.DataFrame, current: pd.DataFrame) -> DataDriftResult:
        logger.info(
            f"Analyzing data drift: {len(reference)} reference rows vs. {len(current)} current rows"
        )
        report = self.detector.analyze(reference, current)
        drifted_features = report.loc[report["drift_severity"] != "none", "feature"].tolist()
        has_major_drift = bool((report["drift_severity"] == "major").any())

        if drifted_features:
            logger.warning(
                f"Drift detected in {len(drifted_features)} feature(s): {drifted_features}"
            )
        else:
            logger.info("No drift detected in any feature")

        return DataDriftResult(
            report=report,
            n_features_checked=len(report),
            drifted_features=drifted_features,
            has_major_drift=has_major_drift,
        )

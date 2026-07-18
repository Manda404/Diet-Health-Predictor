"""
Infrastructure Layer - Data Drift Detection
==============================================

Compares feature distributions between a reference dataset (typically train)
and a current one (typically test, or a future production dataset) using two
complementary signals per feature:

- **PSI** (Population Stability Index) - the standard industry metric for
  drift monitoring; bins the reference distribution and measures how much
  the current distribution's bin proportions have shifted. Thresholds of
  0.1 / 0.25 (moderate / major) are the commonly used industry defaults.
- **KS test** (Kolmogorov-Smirnov, two-sample) - a statistical test that the
  two samples come from the same distribution; its p-value is a
  independent cross-check on PSI's verdict.

Low-cardinality numeric columns (e.g. one-hot encoded 0/1 columns, or a
binary flag) are treated as categorical -- binning them into quantile
buckets like a continuous column would produce meaningless, collapsed bins.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


class DriftDetector:
    """Compares feature distributions between a reference and a current DataFrame."""

    def __init__(
        self,
        buckets: int = 10,
        psi_moderate_threshold: float = 0.1,
        psi_major_threshold: float = 0.25,
        categorical_max_unique: int = 10,
    ):
        self.buckets = buckets
        self.psi_moderate_threshold = psi_moderate_threshold
        self.psi_major_threshold = psi_major_threshold
        self.categorical_max_unique = categorical_max_unique

    def analyze(self, reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        """
        Per-feature drift report, one row per column shared by both
        DataFrames, sorted by PSI descending (most-drifted first).

        Returns:
            DataFrame with columns: feature, psi, ks_statistic, ks_pvalue,
            drift_severity ("none" / "moderate" / "major").
        """
        shared_columns = [column for column in reference.columns if column in current.columns]
        rows = []
        for column in shared_columns:
            psi = self._population_stability_index(reference[column], current[column])
            ks_statistic, ks_pvalue = ks_2samp(reference[column], current[column])
            rows.append(
                {
                    "feature": column,
                    "psi": psi,
                    "ks_statistic": float(ks_statistic),
                    "ks_pvalue": float(ks_pvalue),
                    "drift_severity": self._severity(psi),
                }
            )

        report = pd.DataFrame(
            rows, columns=["feature", "psi", "ks_statistic", "ks_pvalue", "drift_severity"]
        )
        return report.sort_values("psi", ascending=False).reset_index(drop=True)

    def _severity(self, psi: float) -> str:
        if psi >= self.psi_major_threshold:
            return "major"
        if psi >= self.psi_moderate_threshold:
            return "moderate"
        return "none"

    def _population_stability_index(self, reference: pd.Series, current: pd.Series) -> float:
        if reference.nunique() <= self.categorical_max_unique:
            categories = sorted(set(reference.unique()) | set(current.unique()))
            reference_pct = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
            current_pct = current.value_counts(normalize=True).reindex(categories, fill_value=0)
        else:
            breakpoints = np.unique(np.quantile(reference, np.linspace(0, 1, self.buckets + 1)))
            breakpoints[0], breakpoints[-1] = -np.inf, np.inf
            reference_binned = pd.cut(reference, bins=breakpoints, duplicates="drop")
            current_binned = pd.cut(current, bins=breakpoints, duplicates="drop")
            reference_pct = reference_binned.value_counts(normalize=True, sort=False)
            current_pct = current_binned.value_counts(normalize=True, sort=False)

        # Floor at a small epsilon so an empty bin doesn't divide by zero or
        # take log(0); an unseen category/bin is exactly the kind of shift
        # PSI is meant to flag, not something to silently skip.
        epsilon = 1e-6
        reference_pct = reference_pct.clip(lower=epsilon)
        current_pct = current_pct.clip(lower=epsilon)

        return float(((current_pct - reference_pct) * np.log(current_pct / reference_pct)).sum())

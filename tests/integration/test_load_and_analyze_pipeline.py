"""
Integration test: load -> analyze, using the real HealthDietDataLoader and
both application use cases together.

`LoadHealthDietDataUseCase` builds domain objects directly from raw values
with no imputation (the real dataset has no gaps), so it's exercised here
against a cleaned copy of the mock dataset written to a temp CSV — this still
crosses real file I/O + infra + application layers, it just doesn't ask this
use case to also do DataCleaner's job.
"""

import pandas as pd
import pytest

from diet_health_predictor.application.use_cases import (
    AnalyzeHealthStatsUseCase,
    LoadHealthDietDataUseCase,
)
from diet_health_predictor.infrastructure import DataCleaner, HealthDietDataLoader

pytestmark = pytest.mark.integration


@pytest.fixture
def clean_csv_path(mock_csv_path, tmp_path):
    raw_df = pd.read_csv(mock_csv_path)
    clean_df = DataCleaner().clean(raw_df)
    path = tmp_path / "clean_mock_diet_data.csv"
    clean_df.to_csv(path, index=False)
    return path


class TestLoadAndAnalyzePipeline:
    def test_end_to_end_produces_expected_health_status_distribution(self, clean_csv_path):
        loader = HealthDietDataLoader(str(clean_csv_path))
        assessments = LoadHealthDietDataUseCase(loader).execute()
        stats = AnalyzeHealthStatsUseCase().execute(assessments)

        assert stats["total_assessments"] == 16
        assert stats["health_status_distribution"] == {
            "Healthy": 8,
            "Overweight": 4,
            "Obese": 2,
            "Underweight": 2,
        }

    def test_intervention_percentage_matches_overweight_plus_obese_share(self, clean_csv_path):
        loader = HealthDietDataLoader(str(clean_csv_path))
        assessments = LoadHealthDietDataUseCase(loader).execute()
        stats = AnalyzeHealthStatsUseCase().execute(assessments)

        # Overweight (4) + Obese (2) out of 16
        assert stats["people_needing_intervention"] == 6
        assert stats["intervention_percentage"] == pytest.approx(6 / 16 * 100)

    def test_sample_size_is_applied_through_the_full_stack(self, clean_csv_path):
        loader = HealthDietDataLoader(str(clean_csv_path))
        assessments = LoadHealthDietDataUseCase(loader).execute(sample_size=5)
        assert len(assessments) == 5

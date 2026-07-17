"""
Unit tests for diet_health_predictor.application.use_cases

`LoadHealthDietDataUseCase` only calls `.load(sample_size)` on its data_loader
collaborator, so a minimal stub is used here instead of a real
`HealthDietDataLoader` — this keeps the test a true unit test (no file I/O) and
independent from tests/unit/infrastructure/test_data_loader.py, which already
covers the loader itself.
"""

import pandas as pd
import pytest

from diet_health_predictor.application.use_cases import (
    AnalyzeHealthStatsUseCase,
    LoadHealthDietDataUseCase,
)
from diet_health_predictor.domain import (
    ActivityLevel,
    DietAssessment,
    DietType,
    HealthStatus,
    NutritionData,
    Person,
)

pytestmark = pytest.mark.unit


class StubDataLoader:
    """Minimal stand-in for HealthDietDataLoader — records the sample_size it was called with."""

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.last_sample_size = "not called"

    def load(self, sample_size=None):
        self.last_sample_size = sample_size
        return self._df


def make_assessment(health_status, diet_type=DietType.BALANCED, bmi=24.7, calorie_balance=-200.0):
    return DietAssessment(
        person=Person(
            person_id="P0001",
            age=30,
            gender="Male",
            height_cm=180.0,
            weight_kg=80.0,
            bmi=bmi,
            activity_level=ActivityLevel.MODERATELY_ACTIVE,
        ),
        nutrition=NutritionData(
            daily_calorie_requirement=2500.0,
            daily_calorie_consumed=2500.0 + calorie_balance,
            protein_intake_g=150.0,
            carbohydrate_intake_g=250.0,
            fat_intake_g=70.0,
            water_intake_liters=2.5,
        ),
        diet_type=diet_type,
        health_status=health_status,
    )


class TestLoadHealthDietDataUseCase:
    """
    `LoadHealthDietDataUseCase` builds domain objects directly from raw field
    values with no imputation, so it expects already-clean data (as the real
    dataset is). The mock fixture has intentional gaps (see
    tests/fixtures/README.md), so it's run through `DataCleaner` first here —
    that also collapses the deliberate duplicate `Person_ID` down to one row.
    """

    @pytest.fixture
    def clean_mock_df(self, mock_raw_df):
        from diet_health_predictor.infrastructure import DataCleaner

        return DataCleaner().clean(mock_raw_df)

    def test_converts_every_row_to_a_diet_assessment(self, clean_mock_df):
        use_case = LoadHealthDietDataUseCase(StubDataLoader(clean_mock_df))
        assessments = use_case.execute()
        assert len(assessments) == len(clean_mock_df) == 16
        assert all(isinstance(a, DietAssessment) for a in assessments)

    def test_maps_fields_correctly_for_the_first_row(self, clean_mock_df):
        use_case = LoadHealthDietDataUseCase(StubDataLoader(clean_mock_df))
        first = use_case.execute()[0]

        assert first.person.person_id == "P0001"
        assert first.person.age == 45
        assert first.person.bmi == pytest.approx(24.7)
        assert first.diet_type == DietType.BALANCED
        assert first.health_status == HealthStatus.HEALTHY

    def test_passes_sample_size_through_to_the_loader(self, clean_mock_df):
        loader = StubDataLoader(clean_mock_df)
        use_case = LoadHealthDietDataUseCase(loader)
        use_case.execute(sample_size=5)
        assert loader.last_sample_size == 5


class TestAnalyzeHealthStatsUseCase:
    def test_empty_assessments_returns_empty_dict(self):
        assert AnalyzeHealthStatsUseCase().execute([]) == {}

    def test_computes_distributions_and_averages(self):
        assessments = [
            make_assessment(HealthStatus.HEALTHY, bmi=22.0, calorie_balance=0.0),
            make_assessment(HealthStatus.OVERWEIGHT, bmi=27.0, calorie_balance=400.0),
            make_assessment(HealthStatus.OBESE, bmi=32.0, calorie_balance=800.0),
        ]
        stats = AnalyzeHealthStatsUseCase().execute(assessments)

        assert stats["total_assessments"] == 3
        assert stats["health_status_distribution"] == {"Healthy": 1, "Overweight": 1, "Obese": 1}
        assert stats["people_needing_intervention"] == 2  # Overweight + Obese
        assert stats["intervention_percentage"] == pytest.approx(200 / 3)
        assert stats["average_bmi"] == pytest.approx((22.0 + 27.0 + 32.0) / 3)
        assert stats["average_calorie_balance"] == pytest.approx((0.0 + 400.0 + 800.0) / 3)

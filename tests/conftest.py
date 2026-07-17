"""
Shared pytest fixtures.

Test strategy
-------------
- Formula-level unit tests (domain business rules, `FeatureEngineer` columns)
  use small, inline DataFrames/objects built directly in the test — the
  expected values are computed independently in the test itself, not copied
  from a fixture, so they stay easy to verify by hand.
- Everything else (loader validation, cleaning, splitting, encoding, use case
  orchestration, full pipeline integration) is exercised against the single
  shared mock dataset in `tests/fixtures/mock_diet_data.csv`. See
  `tests/fixtures/README.md` for what each row is there to cover.
"""

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MOCK_CSV_PATH = FIXTURES_DIR / "mock_diet_data.csv"


@pytest.fixture
def mock_csv_path() -> Path:
    """Path to the shared mock dataset (see tests/fixtures/README.md)."""
    return MOCK_CSV_PATH


@pytest.fixture
def mock_raw_df() -> pd.DataFrame:
    """
    A fresh copy of the mock dataset as a DataFrame.

    Function-scoped and read fresh each time (rather than session-scoped) so
    that a test mutating the DataFrame in place can't leak state into another
    test.
    """
    return pd.read_csv(MOCK_CSV_PATH)


@pytest.fixture
def sample_person():
    from diet_health_predictor.domain import ActivityLevel, Person

    return Person(
        person_id="P0001",
        age=30,
        gender="Male",
        height_cm=180.0,
        weight_kg=80.0,
        bmi=24.7,
        activity_level=ActivityLevel.MODERATELY_ACTIVE,
    )


@pytest.fixture
def sample_nutrition_data():
    from diet_health_predictor.domain import NutritionData

    return NutritionData(
        daily_calorie_requirement=2500.0,
        daily_calorie_consumed=2300.0,
        protein_intake_g=150.0,
        carbohydrate_intake_g=250.0,
        fat_intake_g=70.0,
        water_intake_liters=2.5,
    )


@pytest.fixture
def sample_diet_assessment(sample_person, sample_nutrition_data):
    from diet_health_predictor.domain import DietAssessment, DietType, HealthStatus

    return DietAssessment(
        person=sample_person,
        nutrition=sample_nutrition_data,
        diet_type=DietType.BALANCED,
        health_status=HealthStatus.HEALTHY,
    )

"""Unit tests for diet_health_predictor.domain.DietAssessment"""

import pytest

from diet_health_predictor.domain import (
    ActivityLevel,
    DietAssessment,
    DietType,
    HealthStatus,
    NutritionData,
    Person,
)

pytestmark = pytest.mark.unit


def make_assessment(health_status=HealthStatus.HEALTHY, **person_overrides) -> DietAssessment:
    person_defaults = dict(
        person_id="P0001",
        age=30,
        gender="Male",
        height_cm=180.0,
        weight_kg=80.0,
        bmi=24.7,
        activity_level=ActivityLevel.MODERATELY_ACTIVE,
    )
    person_defaults.update(person_overrides)

    return DietAssessment(
        person=Person(**person_defaults),
        nutrition=NutritionData(
            daily_calorie_requirement=2500.0,
            daily_calorie_consumed=2300.0,
            protein_intake_g=150.0,
            carbohydrate_intake_g=250.0,
            fat_intake_g=70.0,
            water_intake_liters=2.5,
        ),
        diet_type=DietType.BALANCED,
        health_status=health_status,
    )


class TestNeedsIntervention:
    @pytest.mark.parametrize(
        "health_status, expected",
        [
            (HealthStatus.UNDERWEIGHT, False),
            (HealthStatus.HEALTHY, False),
            (HealthStatus.OVERWEIGHT, True),
            (HealthStatus.OBESE, True),
        ],
    )
    def test_only_overweight_and_obese_need_intervention(self, health_status, expected):
        assert make_assessment(health_status=health_status).needs_intervention() is expected


class TestActivityCalorieMultiplier:
    def test_matches_requirement_over_bmr(self):
        assessment = make_assessment()
        expected = assessment.nutrition.daily_calorie_requirement / assessment.person.bmr()
        assert assessment.activity_calorie_multiplier() == pytest.approx(expected)

    def test_zero_bmr_returns_zero_instead_of_dividing_by_zero(self):
        # base = 10*0 + 6.25*0 - 5*1 = -5; +5 (male offset) => bmr == 0
        assessment = make_assessment(gender="Male", weight_kg=0.0, height_cm=0.0, age=1)
        assert assessment.person.bmr() == 0.0
        assert assessment.activity_calorie_multiplier() == 0.0


class TestProteinPerKgBodyweight:
    def test_matches_protein_over_weight(self):
        assessment = make_assessment(weight_kg=80.0)
        assert assessment.protein_per_kg_bodyweight() == pytest.approx(150.0 / 80.0)

    def test_zero_weight_returns_zero_instead_of_dividing_by_zero(self):
        assessment = make_assessment(weight_kg=0.0)
        assert assessment.protein_per_kg_bodyweight() == 0.0


class TestWaterIntakeMlPerKg:
    def test_matches_water_liters_times_1000_over_weight(self):
        assessment = make_assessment(weight_kg=80.0)
        assert assessment.water_intake_ml_per_kg() == pytest.approx(2.5 * 1000 / 80.0)

    def test_zero_weight_returns_zero_instead_of_dividing_by_zero(self):
        assessment = make_assessment(weight_kg=0.0)
        assert assessment.water_intake_ml_per_kg() == 0.0

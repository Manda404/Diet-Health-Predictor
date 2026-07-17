"""Unit tests for diet_health_predictor.domain.NutritionData"""

import pytest

from diet_health_predictor.domain import NutritionData

pytestmark = pytest.mark.unit


def make_nutrition(**overrides) -> NutritionData:
    defaults = dict(
        daily_calorie_requirement=2500.0,
        daily_calorie_consumed=2300.0,
        protein_intake_g=150.0,
        carbohydrate_intake_g=250.0,
        fat_intake_g=70.0,
        water_intake_liters=2.5,
    )
    defaults.update(overrides)
    return NutritionData(**defaults)


class TestCalorieBalance:
    def test_deficit_is_negative(self):
        nutrition = make_nutrition(daily_calorie_requirement=2500.0, daily_calorie_consumed=2300.0)
        assert nutrition.calorie_balance() == -200.0

    def test_surplus_is_positive(self):
        nutrition = make_nutrition(daily_calorie_requirement=2000.0, daily_calorie_consumed=2200.0)
        assert nutrition.calorie_balance() == 200.0


class TestCalorieDeviationPct:
    def test_matches_balance_over_requirement(self):
        nutrition = make_nutrition(daily_calorie_requirement=2000.0, daily_calorie_consumed=2200.0)
        assert nutrition.calorie_deviation_pct() == pytest.approx(10.0)

    def test_zero_requirement_returns_zero_instead_of_dividing_by_zero(self):
        nutrition = make_nutrition(daily_calorie_requirement=0.0, daily_calorie_consumed=500.0)
        assert nutrition.calorie_deviation_pct() == 0.0


class TestIsAdequateWaterIntake:
    @pytest.mark.parametrize(
        "liters, expected",
        [(1.9, False), (2.0, True), (2.1, True)],
    )
    def test_two_liter_boundary(self, liters, expected):
        assert make_nutrition(water_intake_liters=liters).is_adequate_water_intake() is expected


class TestMacrosCalories:
    def test_uses_atwater_factors(self):
        nutrition = make_nutrition(
            protein_intake_g=100.0, carbohydrate_intake_g=200.0, fat_intake_g=50.0
        )
        protein_kcal, carb_kcal, fat_kcal = nutrition.macros_calories()
        assert protein_kcal == pytest.approx(400.0)
        assert carb_kcal == pytest.approx(800.0)
        assert fat_kcal == pytest.approx(450.0)


class TestMacroRatios:
    def test_ratios_sum_to_one(self):
        nutrition = make_nutrition(
            protein_intake_g=150.0, carbohydrate_intake_g=250.0, fat_intake_g=70.0
        )
        protein_ratio, carb_ratio, fat_ratio = nutrition.macro_ratios()
        assert protein_ratio + carb_ratio + fat_ratio == pytest.approx(1.0)

    def test_zero_intake_returns_zeros_instead_of_dividing_by_zero(self):
        nutrition = make_nutrition(
            protein_intake_g=0.0, carbohydrate_intake_g=0.0, fat_intake_g=0.0
        )
        assert nutrition.macro_ratios() == (0.0, 0.0, 0.0)

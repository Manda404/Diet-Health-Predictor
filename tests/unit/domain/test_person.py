"""Unit tests for diet_health_predictor.domain.Person"""

import pytest

from diet_health_predictor.domain import ActivityLevel, Person

pytestmark = pytest.mark.unit


def make_person(**overrides) -> Person:
    defaults = dict(
        person_id="P0001",
        age=30,
        gender="Male",
        height_cm=180.0,
        weight_kg=80.0,
        bmi=24.7,
        activity_level=ActivityLevel.MODERATELY_ACTIVE,
    )
    defaults.update(overrides)
    return Person(**defaults)


class TestIsHealthy:
    @pytest.mark.parametrize(
        "bmi, expected",
        [
            (18.4, False),  # just below the lower bound
            (18.5, True),  # lower bound is inclusive
            (24.9, True),
            (25.0, False),  # upper bound is exclusive
            (30.0, False),
        ],
    )
    def test_bmi_boundaries(self, bmi, expected):
        assert make_person(bmi=bmi).is_healthy() is expected


class TestGetAgeGroup:
    @pytest.mark.parametrize(
        "age, expected_group",
        [
            (0, "Child"),
            (17, "Child"),
            (18, "Young Adult"),
            (29, "Young Adult"),
            (30, "Adult"),
            (59, "Adult"),
            (60, "Senior"),
            (100, "Senior"),
        ],
    )
    def test_age_boundaries(self, age, expected_group):
        assert make_person(age=age).get_age_group() == expected_group


class TestBmr:
    def test_male_uses_mifflin_st_jeor_male_offset(self):
        person = make_person(gender="Male", weight_kg=80.0, height_cm=180.0, age=30)
        expected = 10 * 80.0 + 6.25 * 180.0 - 5 * 30 + 5
        assert person.bmr() == pytest.approx(expected)

    def test_female_uses_mifflin_st_jeor_female_offset(self):
        person = make_person(gender="Female", weight_kg=60.0, height_cm=165.0, age=30)
        expected = 10 * 60.0 + 6.25 * 165.0 - 5 * 30 - 161
        assert person.bmr() == pytest.approx(expected)

    def test_other_gender_uses_average_of_both_offsets(self):
        person = make_person(gender="Other", weight_kg=70.0, height_cm=170.0, age=50)
        base = 10 * 70.0 + 6.25 * 170.0 - 5 * 50
        male_bmr = base + 5
        female_bmr = base - 161
        assert person.bmr() == pytest.approx((male_bmr + female_bmr) / 2)


class TestIdealWeightKg:
    def test_male_devine_formula(self):
        person = make_person(gender="Male", height_cm=180.0)
        height_inches = 180.0 / 2.54
        expected = 50 + 2.3 * (height_inches - 60)
        assert person.ideal_weight_kg() == pytest.approx(expected)

    def test_female_devine_formula(self):
        person = make_person(gender="Female", height_cm=165.0)
        height_inches = 165.0 / 2.54
        expected = 45.5 + 2.3 * (height_inches - 60)
        assert person.ideal_weight_kg() == pytest.approx(expected)

    def test_other_gender_uses_average_of_both_offsets(self):
        person = make_person(gender="Other", height_cm=170.0)
        height_inches = 170.0 / 2.54
        male_ibw = 50 + 2.3 * (height_inches - 60)
        female_ibw = 45.5 + 2.3 * (height_inches - 60)
        assert person.ideal_weight_kg() == pytest.approx((male_ibw + female_ibw) / 2)


class TestWeightDeviationKg:
    def test_matches_weight_minus_ideal_weight(self):
        person = make_person(gender="Male", height_cm=180.0, weight_kg=90.0)
        assert person.weight_deviation_kg() == pytest.approx(
            person.weight_kg - person.ideal_weight_kg()
        )

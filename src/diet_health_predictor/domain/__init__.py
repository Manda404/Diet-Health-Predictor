"""
Domain Layer - Core Business Entities and Rules
===============================================

The Domain layer contains:
- Business entities (no external dependencies)
- Business rules and validations
- Domain-specific value objects

This layer is independent of any framework or external library.
It represents the "heart" of the application.
"""

from dataclasses import dataclass
from enum import Enum


class HealthStatus(str, Enum):
    """Enumeration of possible health statuses"""

    UNDERWEIGHT = "Underweight"
    HEALTHY = "Healthy"
    OVERWEIGHT = "Overweight"
    OBESE = "Obese"


class DietType(str, Enum):
    """Enumeration of diet types"""

    BALANCED = "Balanced"
    KETO = "Keto"
    VEGETARIAN = "Vegetarian"
    VEGAN = "Vegan"
    HIGH_PROTEIN = "High Protein"
    MEDITERRANEAN = "Mediterranean"


class ActivityLevel(str, Enum):
    """Enumeration of activity levels"""

    SEDENTARY = "Sedentary"
    LIGHTLY_ACTIVE = "Lightly Active"
    MODERATELY_ACTIVE = "Moderately Active"
    VERY_ACTIVE = "Very Active"
    ATHLETE = "Athlete"


@dataclass
class Person:
    """
    Core domain entity representing a person in our system.

    This is a pure business entity with no external dependencies.
    """

    person_id: str
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    bmi: float
    activity_level: ActivityLevel

    def is_healthy(self) -> bool:
        """Determine if person is healthy based on BMI"""
        return 18.5 <= self.bmi < 25.0

    def get_age_group(self) -> str:
        """Categorize person into age group"""
        if self.age < 18:
            return "Child"
        elif self.age < 30:
            return "Young Adult"
        elif self.age < 60:
            return "Adult"
        else:
            return "Senior"

    def bmr(self) -> float:
        """
        Basal Metabolic Rate in kcal/day (Mifflin-St Jeor equation).

        "Other"/unspecified gender uses the average of the male and female
        offsets, since the equation has no third branch.
        """
        base = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age
        if self.gender == "Male":
            return base + 5
        elif self.gender == "Female":
            return base - 161
        return base - 78  # average of +5 and -161

    def ideal_weight_kg(self) -> float:
        """
        Ideal body weight in kg (Devine formula), for comparison against actual weight.

        "Other"/unspecified gender uses the average of the male and female offsets.
        """
        height_inches = self.height_cm / 2.54
        inches_over_5ft = height_inches - 60
        if self.gender == "Male":
            return 50 + 2.3 * inches_over_5ft
        elif self.gender == "Female":
            return 45.5 + 2.3 * inches_over_5ft
        return 47.75 + 2.3 * inches_over_5ft  # average of 50 and 45.5

    def weight_deviation_kg(self) -> float:
        """Actual weight minus ideal weight (Devine formula); positive means above ideal"""
        return self.weight_kg - self.ideal_weight_kg()


@dataclass
class NutritionData:
    """Represents nutrition information for a person"""

    daily_calorie_requirement: float
    daily_calorie_consumed: float
    protein_intake_g: float
    carbohydrate_intake_g: float
    fat_intake_g: float
    water_intake_liters: float

    # Standard Atwater factors: kcal per gram of each macronutrient
    _KCAL_PER_G_PROTEIN = 4.0
    _KCAL_PER_G_CARB = 4.0
    _KCAL_PER_G_FAT = 9.0

    def calorie_balance(self) -> float:
        """Calculate calorie balance (consumed - requirement)"""
        return self.daily_calorie_consumed - self.daily_calorie_requirement

    def calorie_deviation_pct(self) -> float:
        """Calorie balance as a percentage of the required intake"""
        if self.daily_calorie_requirement == 0:
            return 0.0
        return (self.calorie_balance() / self.daily_calorie_requirement) * 100

    def is_adequate_water_intake(self) -> bool:
        """Check if water intake meets minimum (2L recommended)"""
        return self.water_intake_liters >= 2.0

    def macros_calories(self) -> tuple[float, float, float]:
        """Calories contributed by (protein, carbohydrate, fat) intake"""
        return (
            self.protein_intake_g * self._KCAL_PER_G_PROTEIN,
            self.carbohydrate_intake_g * self._KCAL_PER_G_CARB,
            self.fat_intake_g * self._KCAL_PER_G_FAT,
        )

    def macro_ratios(self) -> tuple[float, float, float]:
        """Share of total macro calories from (protein, carbohydrate, fat), each in [0, 1]"""
        protein_kcal, carb_kcal, fat_kcal = self.macros_calories()
        total_kcal = protein_kcal + carb_kcal + fat_kcal
        if total_kcal == 0:
            return (0.0, 0.0, 0.0)
        return (protein_kcal / total_kcal, carb_kcal / total_kcal, fat_kcal / total_kcal)


@dataclass
class DietAssessment:
    """Represents a person's diet and health assessment"""

    person: Person
    nutrition: NutritionData
    diet_type: DietType
    health_status: HealthStatus

    def needs_intervention(self) -> bool:
        """Determine if person needs dietary intervention"""
        return self.health_status in [HealthStatus.OBESE, HealthStatus.OVERWEIGHT]

    def activity_calorie_multiplier(self) -> float:
        """
        Ratio of the stated daily calorie requirement to BMR.

        Cross-checks the declared Activity_Level against actual metabolic demand
        (e.g. ~1.2 for sedentary, ~1.9 for very active, per standard activity
        multiplier tables).
        """
        bmr = self.person.bmr()
        if bmr == 0:
            return 0.0
        return self.nutrition.daily_calorie_requirement / bmr

    def protein_per_kg_bodyweight(self) -> float:
        """Grams of protein per kg of bodyweight (standard nutrition sizing metric)"""
        if self.person.weight_kg == 0:
            return 0.0
        return self.nutrition.protein_intake_g / self.person.weight_kg

    def water_intake_ml_per_kg(self) -> float:
        """Water intake in mL per kg of bodyweight (hydration adequacy relative to size)"""
        if self.person.weight_kg == 0:
            return 0.0
        return (self.nutrition.water_intake_liters * 1000) / self.person.weight_kg

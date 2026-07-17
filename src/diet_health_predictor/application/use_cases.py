"""
Application Layer - Load & Analyze Use Cases
=============================================

Use cases for loading raw data into domain objects and computing summary
statistics over them.
"""

import logging
from typing import Optional

import pandas as pd

from diet_health_predictor.domain import (
    ActivityLevel,
    DietAssessment,
    DietType,
    HealthStatus,
    NutritionData,
    Person,
)
from diet_health_predictor.infrastructure import HealthDietDataLoader

logger = logging.getLogger(__name__)


class LoadHealthDietDataUseCase:
    """
    Use Case: Load and prepare health diet data.

    Orchestrates loading data from infrastructure and transforming it
    into domain objects.
    """

    def __init__(self, data_loader: HealthDietDataLoader):
        self.data_loader = data_loader

    def execute(self, sample_size: Optional[int] = None) -> list[DietAssessment]:
        """
        Execute the use case.

        Args:
            sample_size: Optional sample size

        Returns:
            List of DietAssessment domain objects
        """
        logger.info("Loading health diet data...")
        df = self.data_loader.load(sample_size)

        assessments = []
        for _, row in df.iterrows():
            assessment = self._row_to_assessment(row)
            assessments.append(assessment)

        logger.info(f"Successfully loaded {len(assessments)} assessments")
        return assessments

    @staticmethod
    def _row_to_assessment(row: pd.Series) -> DietAssessment:
        """Convert a DataFrame row to a DietAssessment domain object"""
        person = Person(
            person_id=row["Person_ID"],
            age=int(row["Age"]),
            gender=row["Gender"],
            height_cm=float(row["Height_cm"]),
            weight_kg=float(row["Weight_kg"]),
            bmi=float(row["BMI"]),
            activity_level=ActivityLevel(row["Activity_Level"]),
        )

        nutrition = NutritionData(
            daily_calorie_requirement=float(row["Daily_Calorie_Requirement"]),
            daily_calorie_consumed=float(row["Daily_Calorie_Consumed"]),
            protein_intake_g=float(row["Protein_Intake_g"]),
            carbohydrate_intake_g=float(row["Carbohydrate_Intake_g"]),
            fat_intake_g=float(row["Fat_Intake_g"]),
            water_intake_liters=float(row["Water_Intake_Liters"]),
        )

        return DietAssessment(
            person=person,
            nutrition=nutrition,
            diet_type=DietType(row["Diet_Type"]),
            health_status=HealthStatus(row["Health_Status"]),
        )


class AnalyzeHealthStatsUseCase:
    """Use Case: Analyze health statistics from assessments"""

    def execute(self, assessments: list[DietAssessment]) -> dict:
        """
        Analyze health statistics.

        Returns:
            Dictionary with analysis results
        """
        if not assessments:
            return {}

        health_status_counts: dict[str, int] = {}
        diet_type_counts: dict[str, int] = {}
        people_needing_intervention = 0
        avg_bmi = 0.0
        avg_calorie_balance = 0.0

        for assessment in assessments:
            # Count health statuses
            status = assessment.health_status.value
            health_status_counts[status] = health_status_counts.get(status, 0) + 1

            # Count diet types
            diet = assessment.diet_type.value
            diet_type_counts[diet] = diet_type_counts.get(diet, 0) + 1

            # Count interventions needed
            if assessment.needs_intervention():
                people_needing_intervention += 1

            # Averages
            avg_bmi += assessment.person.bmi
            avg_calorie_balance += assessment.nutrition.calorie_balance()

        n = len(assessments)
        return {
            "total_assessments": n,
            "health_status_distribution": health_status_counts,
            "diet_type_distribution": diet_type_counts,
            "people_needing_intervention": people_needing_intervention,
            "intervention_percentage": (people_needing_intervention / n * 100) if n > 0 else 0,
            "average_bmi": avg_bmi / n if n > 0 else 0,
            "average_calorie_balance": avg_calorie_balance / n if n > 0 else 0,
        }

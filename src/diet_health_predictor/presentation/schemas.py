"""
Presentation Layer - FastAPI Request/Response Schemas
========================================================

Pydantic models for the HTTP layer (`api_app.py`). These validate what comes
over the wire before it ever reaches `HealthDietAPI` -- field constraints
here (e.g. `gt=0`, enum membership) reject malformed requests with a 422
before any use case runs.
"""

from typing import Literal

from pydantic import BaseModel, Field

from diet_health_predictor.domain import ActivityLevel, DietType


class PredictionRequest(BaseModel):
    """
    One raw health/diet record -- the same fields as the source CSV, minus
    `Person_ID` and `Health_Status` (the target being predicted). `BMI` is
    derived server-side from `Height_cm`/`Weight_kg` rather than being asked
    of the caller, since it's a deterministic function of the two.
    """

    age: int = Field(gt=0, le=120, description="Age in years")
    gender: Literal["Male", "Female", "Other"]
    height_cm: float = Field(gt=0, description="Height in centimeters")
    weight_kg: float = Field(gt=0, description="Weight in kilograms")
    activity_level: ActivityLevel
    diet_type: DietType
    daily_calorie_requirement: float = Field(gt=0)
    daily_calorie_consumed: float = Field(gt=0)
    protein_intake_g: float = Field(ge=0)
    carbohydrate_intake_g: float = Field(ge=0)
    fat_intake_g: float = Field(ge=0)
    water_intake_liters: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 30,
                "gender": "Male",
                "height_cm": 175.0,
                "weight_kg": 70.0,
                "activity_level": "Moderately Active",
                "diet_type": "Balanced",
                "daily_calorie_requirement": 2200,
                "daily_calorie_consumed": 2100,
                "protein_intake_g": 90.0,
                "carbohydrate_intake_g": 250.0,
                "fat_intake_g": 70.0,
                "water_intake_liters": 2.5,
            }
        }
    }

    def to_raw_record(self) -> dict:
        """Map to the raw CSV column names `PredictHealthStatusUseCase` expects."""
        bmi = self.weight_kg / (self.height_cm / 100) ** 2
        return {
            "Age": self.age,
            "Gender": self.gender,
            "Height_cm": self.height_cm,
            "Weight_kg": self.weight_kg,
            "BMI": round(bmi, 2),
            "Activity_Level": self.activity_level.value,
            "Daily_Calorie_Requirement": self.daily_calorie_requirement,
            "Daily_Calorie_Consumed": self.daily_calorie_consumed,
            "Protein_Intake_g": self.protein_intake_g,
            "Carbohydrate_Intake_g": self.carbohydrate_intake_g,
            "Fat_Intake_g": self.fat_intake_g,
            "Water_Intake_Liters": self.water_intake_liters,
            "Diet_Type": self.diet_type.value,
        }


class PredictionResponse(BaseModel):
    predicted_health_status: str
    probabilities: dict[str, float]


class ModelInfoResponse(BaseModel):
    model_type: str
    selection_metric: str
    selection_metric_value: float


class HealthCheckResponse(BaseModel):
    status: str
    environment: str

"""
Presentation Layer - FastAPI HTTP Application (Phase 4)
==========================================================

Thin HTTP controllers over `HealthDietAPI`: each route validates its request
via a Pydantic schema, delegates to the API facade, and maps the result (or
a missing-artifact error) onto an HTTP response. No business logic lives
here -- that's `HealthDietAPI` and the use cases underneath it.

Run locally with:
    poetry run uvicorn diet_health_predictor.presentation.api_app:app --reload
"""

import logging

from fastapi import FastAPI, HTTPException

from diet_health_predictor.presentation import HealthDietAPI
from diet_health_predictor.presentation.schemas import (
    HealthCheckResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Diet-Health-Predictor API",
    description="Predicts Health_Status from diet and biometric data.",
    version="0.1.0",
)

api = HealthDietAPI()


@app.get("/health", response_model=HealthCheckResponse, tags=["health"])
def health_check() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok", environment=api.settings.environment)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    try:
        metadata = api.get_best_model_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ModelInfoResponse(**metadata)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        result = api.predict_health_status(request.to_raw_record())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PredictionResponse(
        predicted_health_status=result.predicted_health_status,
        probabilities=result.probabilities,
    )

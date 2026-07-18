"""
Integration test: the FastAPI HTTP layer (Phase 4), driven with a real
`TestClient` against a real preprocess -> compare -> select-best pipeline on
the mock dataset -- no mocking of the ASGI app or `HealthDietAPI`.

`api_app.api` is a module-level `HealthDietAPI()` singleton (built once at
import time, like `get_settings()`); the fields these tests need to redirect
are patched via `monkeypatch`, same as `test_health_diet_api.py`.
"""

import pytest
from fastapi.testclient import TestClient

from diet_health_predictor.presentation import api_app

pytestmark = pytest.mark.integration

_VALID_PAYLOAD = {
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


@pytest.fixture
def client(mock_csv_path, tmp_path, monkeypatch):
    monkeypatch.setattr(api_app.api, "data_path", mock_csv_path)
    monkeypatch.setattr(
        api_app.api.settings.data, "processed_data_path", str(tmp_path / "processed")
    )
    monkeypatch.setattr(api_app.api.settings.model, "models_output_dir", str(tmp_path / "models"))
    return TestClient(api_app.app)


@pytest.fixture
def trained_client(client):
    """A client whose backing HealthDietAPI already has a selected best model."""
    preprocessing_result = api_app.api.preprocess_data()
    comparison = api_app.api.compare_models(preprocessing_result)
    api_app.api.select_best_model(comparison)
    return client


class TestHealthEndpoint:
    def test_health_check_reports_ok_and_the_current_environment(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "environment": "development"}


class TestModelInfoEndpoint:
    def test_returns_404_before_a_model_is_selected(self, client):
        response = client.get("/model/info")
        assert response.status_code == 404

    def test_returns_the_winning_model_after_selection(self, trained_client):
        response = trained_client.get("/model/info")
        assert response.status_code == 200
        body = response.json()
        assert body["model_type"] in {"xgboost", "catboost"}
        assert body["selection_metric"] == "mcc"


class TestPredictEndpoint:
    def test_returns_404_before_a_model_is_selected(self, client):
        response = client.post("/predict", json=_VALID_PAYLOAD)
        assert response.status_code == 404

    def test_predicts_a_known_health_status_after_training(self, trained_client):
        response = trained_client.post("/predict", json=_VALID_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_health_status"] in {
            "Healthy",
            "Overweight",
            "Obese",
            "Underweight",
        }
        assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)

    def test_rejects_invalid_age_with_422(self, trained_client):
        response = trained_client.post("/predict", json={**_VALID_PAYLOAD, "age": -5})
        assert response.status_code == 422

    def test_rejects_unknown_gender_with_422(self, trained_client):
        response = trained_client.post("/predict", json={**_VALID_PAYLOAD, "gender": "Robot"})
        assert response.status_code == 422

    def test_rejects_unknown_activity_level_with_422(self, trained_client):
        response = trained_client.post(
            "/predict", json={**_VALID_PAYLOAD, "activity_level": "Superhuman"}
        )
        assert response.status_code == 422

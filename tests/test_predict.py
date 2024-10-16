import os
import pytest
import asyncio
from fastapi.testclient import TestClient
from main import app, lifespan

client = TestClient(app)

@pytest.mark.parametrize("model_type", ["pyspark", "pickle"])
def test_predict_missing_field(monkeypatch, model_type):
    # Set the environment variable for MODEL_TYPE
    monkeypatch.setenv('MODEL_TYPE', model_type)

    # Use asyncio to run the async lifespan
    async def run_lifespan():
        async with lifespan(app):
            client = TestClient(app)
            response = client.post(
                "/model/predict/",
                json={
                    "dep_delay": -1.0,
                    "origin_wind_speed": 5.8,
                    "distance": 212.0,
                    "carrier": "EV"
                }
            )
            assert response.status_code == 422  # 422 indicates a validation error

    asyncio.run(run_lifespan())

@pytest.mark.parametrize("model_type", ["pyspark", "pickle"])
def test_predict_invalid_data_type(monkeypatch, model_type):
    # Set the environment variable for MODEL_TYPE
    monkeypatch.setenv('MODEL_TYPE', model_type)

    async def run_lifespan():
        async with lifespan(app):
            client = TestClient(app)
            response = client.post(
                "/model/predict/",
                json={
                    "dep_delay": "not_a_number",
                    "origin_wind_speed": 5.8,
                    "dest_wind_speed": 1.9,
                    "distance": 212.0,
                    "carrier": "EV"
                }
            )
            assert response.status_code == 422  # Validation should fail

    asyncio.run(run_lifespan())

@pytest.mark.parametrize("model_type", ["pyspark", "pickle"])
def test_predict_edge_case_data(monkeypatch, model_type):
    # Set the environment variable for MODEL_TYPE
    monkeypatch.setenv('MODEL_TYPE', model_type)

    async def run_lifespan():
        async with lifespan(app):
            client = TestClient(app)
            response = client.post(
                "/model/predict/",
                json={
                    "dep_delay": 1000.0,  # Extreme delay
                    "origin_wind_speed": 100.0,  # Extreme wind speed
                    "dest_wind_speed": 0.0,
                    "distance": 10000.0,  # Extreme distance
                    "carrier": "EV"
                }
            )
            assert response.status_code == 200
            assert "prediction" in response.json()

    asyncio.run(run_lifespan())

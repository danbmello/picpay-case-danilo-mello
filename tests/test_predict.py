import asyncio

import pytest

from fastapi.testclient import TestClient

from main import app, lifespan

# Initialize the test client for FastAPI
client = TestClient(app)


# Test case for missing required fields in the prediction request
@pytest.mark.parametrize(
    "model_type", ["pyspark", "pickle"]
)  # Parameterize the test to run with both 'pyspark' and 'pickle' model types
def test_predict_missing_field(monkeypatch, model_type):
    # Mock the environment variable for MODEL_TYPE
    monkeypatch.setenv("MODEL_TYPE", model_type)

    # Define an async function to handle the lifespan context
    async def run_lifespan():
        async with lifespan(app):
            client = TestClient(app)
            # Send a POST request to /model/predict/ with missing fields (no 'dest_wind_speed')
            response = client.post(
                "/model/predict/",
                json={
                    "dep_delay": -1.0,
                    "origin_wind_speed": 5.8,
                    "distance": 212.0,
                    "carrier": "EV",
                },
            )
            # Assert that the response status is 422, which indicates a validation error
            assert response.status_code == 422

    # Run the async lifespan using asyncio
    asyncio.run(run_lifespan())


# Test case for invalid data types in the prediction request
@pytest.mark.parametrize("model_type", ["pyspark", "pickle"])
def test_predict_invalid_data_type(monkeypatch, model_type):
    # Mock the environment variable for MODEL_TYPE
    monkeypatch.setenv("MODEL_TYPE", model_type)

    # Define an async function to handle the lifespan context
    async def run_lifespan():
        async with lifespan(app):
            client = TestClient(app)
            # Send a POST request to /model/predict/ with an invalid data type for 'dep_delay'
            response = client.post(
                "/model/predict/",
                json={
                    "dep_delay": "not_a_number",  # Invalid data type (string instead of float)
                    "origin_wind_speed": 5.8,
                    "dest_wind_speed": 1.9,
                    "distance": 212.0,
                    "carrier": "EV",
                },
            )
            # Assert that the response status is 422, indicating a validation error
            assert response.status_code == 422

    # Run the async lifespan using asyncio
    asyncio.run(run_lifespan())


# Test case for edge-case data values in the prediction request
@pytest.mark.parametrize("model_type", ["pyspark", "pickle"])
def test_predict_edge_case_data(monkeypatch, model_type):
    # Mock the environment variable for MODEL_TYPE
    monkeypatch.setenv("MODEL_TYPE", model_type)

    # Define an async function to handle the lifespan context
    async def run_lifespan():
        async with lifespan(app):
            client = TestClient(app)
            # Send a POST request to /model/predict/ with extreme values for the input fields
            response = client.post(
                "/model/predict/",
                json={
                    "dep_delay": 1000.0,  # Extreme delay value
                    "origin_wind_speed": 100.0,  # Extreme wind speed
                    "dest_wind_speed": 0.0,
                    "distance": 10000.0,  # Extreme distance value
                    "carrier": "EV",
                },
            )
            # Assert that the response status is 200, indicating a successful prediction
            assert response.status_code == 200
            # Assert that the prediction result is present in the response
            assert "prediction" in response.json()

    # Run the async lifespan using asyncio
    asyncio.run(run_lifespan())

import asyncio

import pytest

from fastapi.testclient import TestClient

from main import app, lifespan

# Initialize the test client for the FastAPI app
client = TestClient(app)


# Test case for handling unsupported model types
@pytest.mark.parametrize("model_type", ["unsupported_model"])
def test_unsupported_model_type(monkeypatch, model_type):
    # Mock the environment variable for MODEL_TYPE with an unsupported value
    monkeypatch.setenv("MODEL_TYPE", model_type)

    # Define an async function to handle the lifespan context
    async def run_lifespan():
        async with lifespan(app):  # Use the app's lifespan to ensure model loading
            client = TestClient(app)
            # Send a POST request to /model/predict/ with flight data
            response = client.post(
                "/model/predict/",
                json={
                    "dep_delay": -1.0,
                    "origin_wind_speed": 5.8,
                    "dest_wind_speed": 1.9,
                    "distance": 212.0,
                    "carrier": "EV",
                },
            )
            # Assert that the response status is 400 (bad request) due to unsupported model type
            assert response.status_code == 400
            # Assert that the correct error message is returned for the unsupported model type
            assert response.json()["detail"] == f"Unknown model type: {model_type}"

    # Run the async lifespan using asyncio
    asyncio.run(run_lifespan())


# Test case for retrieving prediction history
def test_get_prediction_history(monkeypatch):
    # Mock the environment variable for MODEL_TYPE with 'pyspark'
    monkeypatch.setenv("MODEL_TYPE", "pyspark")

    # Define an async function to handle lifespan context
    async def run_lifespan():
        async with lifespan(app):  # Ensure the app loads the model correctly
            client = TestClient(app)

            # First, make a prediction using the /model/predict/ endpoint
            response = client.post(
                "/model/predict/",
                json={
                    "dep_delay": -1.0,
                    "origin_wind_speed": 5.8,
                    "dest_wind_speed": 1.9,
                    "distance": 212.0,
                    "carrier": "EV",
                },
            )

            # Assert that the response is either 200 (success) or 400 (error due to model issues)
            assert response.status_code == 200 or response.status_code == 400
            if response.status_code == 200:
                # If successful, ensure the response contains a prediction
                assert "prediction" in response.json()

                # Retrieve the prediction history from the /model/history/ endpoint
                response = client.get("/model/history/")
                assert response.status_code == 200  # Ensure the history request is successful
                assert "history" in response.json()  # Ensure the history key exists in the response
                assert isinstance(response.json()["history"], list)  # Ensure the history is a list
                assert len(response.json()["history"]) > 0  # Ensure the history has entries

            elif response.status_code == 400:
                # If the response is 400, ensure the error is due to model unavailability
                assert response.json()["detail"] == "Service Temporarily Unavailable! Please try again later."

    # Run the async lifespan using asyncio
    asyncio.run(run_lifespan())

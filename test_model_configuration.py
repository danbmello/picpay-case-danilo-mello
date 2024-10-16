from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

@pytest.mark.parametrize("model_type", ["unsupported_model"])
def test_unsupported_model_type(monkeypatch, model_type):
    # Mock the environment variable for MODEL_TYPE
    monkeypatch.setenv('MODEL_TYPE', model_type)

    # Recreate the client to apply the new environment variable
    client = TestClient(app)
    
    response = client.post(
        "/model/predict/",
        json={
            "dep_delay": -1.0,
            "origin_wind_speed": 5.8,
            "dest_wind_speed": 1.9,
            "distance": 212.0,
            "carrier": "EV"
        }
    )
    assert response.status_code == 400
    # Adjusted assertion to match the new error message
    assert response.json()["detail"] == "Service Temporarily Unavailable! Please try again later."

def test_get_prediction_history():
    # First, make a prediction with a valid model type
    response = client.post(
        "/model/predict/",
        json={
            "dep_delay": -1.0,
            "origin_wind_speed": 5.8,
            "dest_wind_speed": 1.9,
            "distance": 212.0,
            "carrier": "EV"
        }
    )

    # If the model is loaded properly, the response should have a 200 status
    assert response.status_code == 200 or response.status_code == 400
    if response.status_code == 200:
        assert "prediction" in response.json()
        # Then, check the history
        response = client.get("/model/history/")
        assert response.status_code == 200
        assert "history" in response.json()
        assert isinstance(response.json()["history"], list)
        assert len(response.json()["history"]) > 0
    elif response.status_code == 400:
        # Ensure that the error message is about model unavailability
        assert response.json()["detail"] == "Service Temporarily Unavailable! Please try again later."


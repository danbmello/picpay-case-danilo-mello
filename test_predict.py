from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

print(client)

def test_model_is_loaded():
    assert model is not None, "Model should be loaded"

def test_predict_valid_data():
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
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)
    
def test_predict_missing_field():
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
    
def test_predict_invalid_data_type():
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

def test_predict_edge_case_data():
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


#import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_welcome():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ML Model API!"}

def test_post_predict_valid_data():
    data = {
        "feature_1": 1.0,
        "feature_2": 2.0,
        "feature_3": 3.0
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    response_data = response.json()
    assert "prediction" in response_data
    assert response_data["prediction"] == 6.0  # Expecting 1.0 + 2.0 + 3.0


def test_post_predict_invalid_data():
    # Missing required field "feature_3"
    data = {
        "feature_1": 1.0,
        "feature_2": 2.0
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 422  # 422 is Unprocessable Entity (validation error)

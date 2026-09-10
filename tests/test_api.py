from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_valid_input():
    payload = {
        "OverallQual": 7,
        "GrLivArea": 1710,
        "GarageCars": 2,
        "GarageArea": 548,
        "TotalBsmtSF": 856,
        "FullBath": 2,
        "YearBuilt": 2003,
        "Neighborhood": "CollgCr"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price" in response.json()


def test_predict_invalid_input():
    payload = {
        "OverallQual": 50,   # sengaja salah, maksimal cuma 10
        "GrLivArea": 1710,
        "GarageCars": 2,
        "GarageArea": 548,
        "TotalBsmtSF": 856,
        "FullBath": 2,
        "YearBuilt": 2003,
        "Neighborhood": "CollgCr"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
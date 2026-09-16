from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "1.0.0"


def test_predict_valid_payload():
    payload = {
        "tenure": 3,
        "MonthlyCharges": 75.0,
        "TotalCharges": 225.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_tier" in data
    assert "model_version" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in (0, 1)
    assert data["risk_tier"] in ("Low", "Medium", "High")


def test_predict_invalid_payload():
    invalid_payload = {
        "tenure": -5,
        "MonthlyCharges": 75.0,
        "TotalCharges": 225.0,
        "Contract": "InvalidContractType",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

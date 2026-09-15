import pandas as pd
from telco.predict import load_artifact, determine_risk_tier, predict_single, predict_batch


def test_load_artifact():
    artifact = load_artifact()
    assert "pipeline" in artifact
    assert "threshold" in artifact
    assert "features" in artifact
    assert "version" in artifact
    assert artifact["threshold"] == 0.25
    assert len(artifact["features"]) == 9


def test_determine_risk_tier():
    assert determine_risk_tier(0.10, 0.25) == "Low"
    assert determine_risk_tier(0.24, 0.25) == "Low"
    assert determine_risk_tier(0.25, 0.25) == "Medium"
    assert determine_risk_tier(0.55, 0.25) == "Medium"
    assert determine_risk_tier(0.60, 0.25) == "High"
    assert determine_risk_tier(0.95, 0.25) == "High"


def test_predict_single():
    sample = {
        "tenure": 2,
        "MonthlyCharges": 85.5,
        "TotalCharges": 171.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
    }
    res = predict_single(sample)

    assert "churn_probability" in res
    assert "churn_prediction" in res
    assert "risk_tier" in res
    assert "model_version" in res
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["churn_prediction"] in (0, 1)
    assert res["risk_tier"] in ("Low", "Medium", "High")


def test_predict_batch():
    samples = pd.DataFrame(
        [
            {
                "tenure": 70,
                "MonthlyCharges": 20.0,
                "TotalCharges": 1400.0,
                "Contract": "Two year",
                "InternetService": "No",
                "OnlineSecurity": "No internet service",
                "TechSupport": "No internet service",
                "PaymentMethod": "Bank transfer (automatic)",
                "PaperlessBilling": "No",
            },
            {
                "tenure": 1,
                "MonthlyCharges": 90.0,
                "TotalCharges": 90.0,
                "Contract": "Month-to-month",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "TechSupport": "No",
                "PaymentMethod": "Electronic check",
                "PaperlessBilling": "Yes",
            },
        ]
    )
    res = predict_batch(samples)

    assert len(res) == 2
    assert "churn_probability" in res.columns
    assert "churn_prediction" in res.columns
    assert "risk_tier" in res.columns
    assert res["churn_probability"].iloc[0] < res["churn_probability"].iloc[1]

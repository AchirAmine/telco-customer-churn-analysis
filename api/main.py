from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from telco.predict import load_artifact, predict_single

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    version="1.0.0",
)

artifact = None


def get_artifact():
    global artifact
    if artifact is None:
        artifact = load_artifact()
    return artifact


class CustomerPayload(BaseModel):
    tenure: int = Field(..., ge=0, le=120)
    MonthlyCharges: float = Field(..., ge=0.0, le=500.0)
    TotalCharges: float = Field(..., ge=0.0)
    Contract: Literal["Month-to-month", "One year", "Two year"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    PaperlessBilling: Literal["Yes", "No"]


class HealthResponse(BaseModel):
    status: str
    model_version: str


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_tier: str
    model_version: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    art = get_artifact()
    return {
        "status": "healthy",
        "model_version": art["version"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerPayload):
    try:
        art = get_artifact()
        data = payload.model_dump()
        result = predict_single(data, artifact=art)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

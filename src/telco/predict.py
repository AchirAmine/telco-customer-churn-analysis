from pathlib import Path
import joblib
import pandas as pd
import numpy as np

DEFAULT_ARTIFACT_PATH = Path("models/model_v1.joblib")


def load_artifact(artifact_path: str | Path | None = None) -> dict:
    if artifact_path is None:
        artifact_path = DEFAULT_ARTIFACT_PATH
    path = Path(artifact_path)
    if not path.exists():
        fallback = Path(__file__).resolve().parent.parent.parent / "models" / "model_v1.joblib"
        if fallback.exists():
            path = fallback
    return joblib.load(path)


def determine_risk_tier(probability: float, threshold: float = 0.25) -> str:
    if probability < threshold:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    return "High"


def predict_single(
    customer_data: dict,
    artifact: dict | None = None,
) -> dict:
    if artifact is None:
        artifact = load_artifact()

    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]
    features = artifact["features"]
    version = artifact["version"]

    df = pd.DataFrame([customer_data])
    for col in features:
        if col not in df.columns:
            df[col] = np.nan
    df = df[features]

    proba = float(pipeline.predict_proba(df)[0, 1])
    pred = int(proba >= threshold)
    risk = determine_risk_tier(proba, threshold=threshold)

    return {
        "churn_probability": round(proba, 4),
        "churn_prediction": pred,
        "risk_tier": risk,
        "model_version": version,
    }


def predict_batch(
    df: pd.DataFrame,
    artifact: dict | None = None,
) -> pd.DataFrame:
    if artifact is None:
        artifact = load_artifact()

    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]
    features = artifact["features"]
    version = artifact["version"]

    input_df = df.copy()
    for col in features:
        if col not in input_df.columns:
            input_df[col] = np.nan
    input_df = input_df[features]

    probas = pipeline.predict_proba(input_df)[:, 1]
    preds = (probas >= threshold).astype(int)
    tiers = [determine_risk_tier(p, threshold) for p in probas]

    result = df.copy()
    result["churn_probability"] = np.round(probas, 4)
    result["churn_prediction"] = preds
    result["risk_tier"] = tiers
    result["model_version"] = version

    return result

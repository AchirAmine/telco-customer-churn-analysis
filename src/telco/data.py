from pathlib import Path
import pandas as pd


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    return df


def get_feature_target_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    data = df.copy()
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])
    target = (data["Churn"] == "Yes").astype(int)
    features = data.drop(columns=["Churn"])
    return features, target

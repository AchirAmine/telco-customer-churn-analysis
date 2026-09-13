from sklearn.compose import ColumnTransformer
from telco.pipeline import build_preprocessor

REDUCED_NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

REDUCED_CATEGORICAL_FEATURES = [
    "Contract",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaymentMethod",
    "PaperlessBilling",
]

REDUCED_FEATURES = REDUCED_NUMERIC_FEATURES + REDUCED_CATEGORICAL_FEATURES


def get_reduced_feature_columns() -> list[str]:
    return list(REDUCED_FEATURES)


def build_reduced_preprocessor() -> ColumnTransformer:
    return build_preprocessor(
        numeric_features=REDUCED_NUMERIC_FEATURES,
        categorical_features=REDUCED_CATEGORICAL_FEATURES,
    )

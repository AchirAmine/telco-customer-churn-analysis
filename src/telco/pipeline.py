from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    if categorical_features is None:
        transformers = [
            ("num", numeric_transformer, numeric_features),
            (
                "cat",
                categorical_transformer,
                lambda df: [c for c in df.columns if c not in numeric_features],
            ),
        ]
    else:
        transformers = [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]

    return ColumnTransformer(transformers=transformers)


def create_full_pipeline(
    model,
    preprocessor: ColumnTransformer | None = None,
) -> Pipeline:
    if preprocessor is None:
        preprocessor = build_preprocessor()

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

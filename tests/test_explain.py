from telco.data import load_raw_data, get_feature_target_split
from telco.explain import (
    get_reduced_feature_columns,
    build_reduced_preprocessor,
)
from telco.pipeline import create_full_pipeline
from sklearn.linear_model import LogisticRegression


def test_reduced_features_count():
    features = get_reduced_feature_columns()
    assert len(features) == 9
    assert "Contract" in features
    assert "tenure" in features
    assert "MonthlyCharges" in features


def test_reduced_pipeline_execution():
    df = load_raw_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv").head(100)
    X, y = get_feature_target_split(df)
    features = get_reduced_feature_columns()
    X_reduced = X[features]

    preprocessor = build_reduced_preprocessor()
    pipe = create_full_pipeline(LogisticRegression(), preprocessor=preprocessor)
    pipe.fit(X_reduced, y)

    preds = pipe.predict(X_reduced)
    probs = pipe.predict_proba(X_reduced)

    assert len(preds) == 100
    assert probs.shape == (100, 2)

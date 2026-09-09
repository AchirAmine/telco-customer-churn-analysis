import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from telco.data import load_raw_data, get_feature_target_split
from telco.pipeline import build_preprocessor, create_full_pipeline


def test_preprocessor_shape():
    df = load_raw_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X, _ = get_feature_target_split(df)
    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X)

    assert X_trans.shape[0] == len(X)
    assert X_trans.shape[1] == 46
    assert not np.isnan(X_trans).any()


def test_unseen_category_handling():
    df = load_raw_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X, _ = get_feature_target_split(df)
    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    unseen_row = X.iloc[[0]].copy()
    unseen_row.loc[0, "InternetService"] = "QuantumFiber"
    unseen_row.loc[0, "Contract"] = "Decade-to-Decade"

    trans = preprocessor.transform(unseen_row)
    assert trans.shape == (1, 46)
    assert not np.isnan(trans).any()


def test_missing_imputation():
    df = load_raw_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X, _ = get_feature_target_split(df)
    preprocessor = build_preprocessor()

    X_with_nan = X.head(50).copy()
    X_with_nan.loc[0, "TotalCharges"] = np.nan
    X_with_nan.loc[1, "MonthlyCharges"] = np.nan
    X_with_nan.loc[2, "PaymentMethod"] = np.nan

    trans = preprocessor.fit_transform(X_with_nan)
    assert not np.isnan(trans).any()


def test_pipeline_fit_predict():
    df = load_raw_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv").head(200)
    X, y = get_feature_target_split(df)
    model = LogisticRegression(max_iter=200)
    pipe = create_full_pipeline(model)
    pipe.fit(X, y)

    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)

    assert len(preds) == 200
    assert set(preds).issubset({0, 1})
    assert probs.shape == (200, 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

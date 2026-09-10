from telco.data import load_raw_data, get_feature_target_split
from telco.train import get_candidate_models
from telco.evaluate import evaluate_model_cv, compare_all_models


def test_candidate_models_presence():
    models = get_candidate_models()
    assert "Dummy Baseline" in models
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    assert "HistGradientBoosting" in models
    assert "LightGBM" in models


def test_evaluate_model_cv_output():
    df = load_raw_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv").head(150)
    X, y = get_feature_target_split(df)
    models = get_candidate_models()

    metrics = evaluate_model_cv(models["Logistic Regression"], X, y, n_splits=2)
    assert "roc_auc_mean" in metrics
    assert "pr_auc_mean" in metrics
    assert "f1_mean" in metrics
    assert "brier_score_mean" in metrics
    assert 0.0 <= metrics["roc_auc_mean"] <= 1.0


def test_compare_all_models_dataframe():
    df = load_raw_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv").head(150)
    X, y = get_feature_target_split(df)
    res = compare_all_models(X, y, n_splits=2)

    assert len(res) == 5
    assert "Model" in res.columns
    assert "ROC-AUC" in res.columns
    assert "PR-AUC" in res.columns

import numpy as np
import pandas as pd
from telco.data import load_raw_data, get_feature_target_split
from telco.tune import find_optimal_threshold, evaluate_threshold_costs, tune_logistic_regression


def test_find_optimal_threshold_logic():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.6, 0.2, 0.8, 0.1, 0.7])
    opt_t, opt_cost, def_cost = find_optimal_threshold(
        probs, y_true, cost_fp=10.0, cost_fn=100.0
    )

    assert 0.05 <= opt_t <= 0.95
    assert opt_cost <= def_cost


def test_evaluate_threshold_costs_structure():
    y_true = np.array([0, 1, 0, 1])
    probs = np.array([0.2, 0.7, 0.4, 0.8])
    res = evaluate_threshold_costs(probs, y_true, threshold=0.5)

    assert "total_cost" in res
    assert "false_positives" in res
    assert "false_negatives" in res
    assert res["total_cost"] >= 0


def test_tuning_fast_execution():
    df = load_raw_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv").head(100)
    X, y = get_feature_target_split(df)
    search = tune_logistic_regression(X, y, n_iter=2)

    assert hasattr(search, "best_estimator_")
    assert hasattr(search, "best_score_")
    assert search.best_score_ > 0.5

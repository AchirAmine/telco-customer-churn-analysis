import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from telco.pipeline import create_full_pipeline


def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 10,
    random_state: int = 42,
) -> RandomizedSearchCV:
    pipe = create_full_pipeline(
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
    )
    param_dist = {
        "classifier__C": np.logspace(-2, 2, 20),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=n_iter,
        cv=skf,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def tune_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 15,
    random_state: int = 42,
) -> RandomizedSearchCV:
    pipe = create_full_pipeline(
        HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=random_state,
        )
    )
    param_dist = {
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "classifier__max_depth": [3, 4, 5, 6, None],
        "classifier__min_samples_leaf": [15, 20, 30, 50],
        "classifier__l2_regularization": [0.0, 0.1, 1.0, 5.0],
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=n_iter,
        cv=skf,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def find_optimal_threshold(
    probabilities: np.ndarray,
    y_true: pd.Series | np.ndarray,
    cost_fp: float = 15.0,
    cost_fn: float = 100.0,
    num_thresholds: int = 91,
) -> tuple[float, float, float]:
    thresholds = np.linspace(0.05, 0.95, num_thresholds)
    costs = []

    for t in thresholds:
        preds = (probabilities >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = (fp * cost_fp) + (fn * cost_fn)
        costs.append(cost)

    best_idx = int(np.argmin(costs))
    optimal_t = float(thresholds[best_idx])
    optimal_cost = float(costs[best_idx])

    default_preds = (probabilities >= 0.50).astype(int)
    tn5, fp5, fn5, tp5 = confusion_matrix(y_true, default_preds).ravel()
    default_cost = float((fp5 * cost_fp) + (fn5 * cost_fn))

    return optimal_t, optimal_cost, default_cost


def evaluate_threshold_costs(
    probabilities: np.ndarray,
    y_true: pd.Series | np.ndarray,
    threshold: float = 0.5,
    cost_fp: float = 15.0,
    cost_fn: float = 100.0,
) -> dict[str, float]:
    preds = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    total_cost = float((fp * cost_fp) + (fn * cost_fn))
    return {
        "threshold": float(threshold),
        "total_cost": total_cost,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }

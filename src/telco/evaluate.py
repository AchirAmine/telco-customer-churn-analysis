import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
)
from telco.pipeline import create_full_pipeline
from telco.train import get_candidate_models


def evaluate_model_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    pipe = create_full_pipeline(model)
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    roc_aucs, pr_aucs, f1s, briers = [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe.fit(X_train, y_train)

        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_val)[:, 1]
        else:
            probs = np.full(len(X_val), y_train.mean())

        preds = pipe.predict(X_val)

        try:
            roc_aucs.append(roc_auc_score(y_val, probs))
        except ValueError:
            roc_aucs.append(0.5)

        pr_aucs.append(average_precision_score(y_val, probs))
        f1s.append(f1_score(y_val, preds, zero_division=0))
        briers.append(brier_score_loss(y_val, probs))

    return {
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "f1_mean": float(np.mean(f1s)),
        "brier_score_mean": float(np.mean(briers)),
    }


def compare_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    models = get_candidate_models(random_state=random_state)
    records = []

    for name, model in models.items():
        metrics = evaluate_model_cv(
            model=model,
            X=X,
            y=y,
            n_splits=n_splits,
            random_state=random_state,
        )
        records.append(
            {
                "Model": name,
                "ROC-AUC": metrics["roc_auc_mean"],
                "ROC-AUC Std": metrics["roc_auc_std"],
                "PR-AUC": metrics["pr_auc_mean"],
                "F1-Score": metrics["f1_mean"],
                "Brier Score": metrics["brier_score_mean"],
            }
        )

    return pd.DataFrame(records).sort_values(by="ROC-AUC", ascending=False)

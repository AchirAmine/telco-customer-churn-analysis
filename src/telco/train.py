from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier


def get_candidate_models(random_state: int = 42) -> dict:
    return {
        "Dummy Baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
        "LightGBM": LGBMClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_estimators=100,
            verbose=-1,
            n_jobs=-1,
        ),
    }

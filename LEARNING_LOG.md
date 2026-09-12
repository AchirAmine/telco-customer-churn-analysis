# Learning Log — Telco Customer Churn Prediction

## S1 — Project Setup (2026-09-05)

**What I did:** Created the production src layout, configured a dedicated Python 3.11 virtual environment with all required dependencies, set up .gitignore to isolate data and models, connected the remote GitHub repository, and documented the project foundation.

**What I learned:** Isolating dependencies in a dedicated virtual environment prevents package version conflicts. A clean src package layout separates reusable production code from notebooks and deployment interfaces, while .gitignore safeguards repository performance and data privacy.

## S2 — Exploratory Data Analysis & Anomaly Resolution (2026-09-07)

**What I did:** Inspected the 7,043 customer records across 21 columns, resolved a subtle data quirk where 11 rows in TotalCharges stored whitespace strings for brand new customers (tenure=0), quantified the target class imbalance (73.46% retained vs 26.54% churned), and visualized key churn relationships across contract types, tenure, and monthly charges in an executed notebook.

**What I learned:** Raw accuracy is fundamentally deceptive on imbalanced datasets (~26.5% minority class) since a naive non-churn baseline scores 73.5% while finding zero churners. Converting string-encoded missing values via domain knowledge (zero tenure implies zero total charges) prevents data leakage while preserving valid customer records.

## S3 — Leakage-Safe ColumnTransformer & Preprocessing Pipeline (2026-09-09)

**What I did:** Built a modular ColumnTransformer encapsulating median imputation and standard scaling for numeric features alongside most-frequent imputation and ignore-unknown one-hot encoding for categorical features, integrated into an end-to-end Pipeline validated by 4 automated unit tests.

**What I learned:** Encapsulating feature transformations strictly inside the Pipeline ensures that encoders and scalers are fitted exclusively on training folds during cross-validation, completely eliminating data leakage. Setting handle_unknown="ignore" on OneHotEncoder ensures robust inference when unseen categories arrive at serving time.

## S4 — Stratified 5-Fold Cross-Validation Model Comparison (2026-09-10)

**What I did:** Benchmarked 5 candidate models (Dummy Baseline, Logistic Regression, Random Forest, HistGradientBoosting, LightGBM) across 5 stratified folds with class_weight="balanced", measuring ROC-AUC, PR-AUC, F1-Score, and Brier Score.

**What I learned:** Logistic Regression achieved the top ranking discrimination (ROC-AUC: 0.8449, PR-AUC: 0.6554) outperforming tree ensembles under standard scaling, while the Dummy Baseline's PR-AUC (0.2654) matched the exact class prevalence. HistGradientBoosting and LightGBM demonstrated superior probability calibration with lower Brier scores (0.1587 and 0.1597).

## S5 — Hyperparameter Tuning, Cost-Sensitive Threshold Optimization & Test Evaluation (2026-09-12)

**What I did:** Tuned hyperparameters via RandomizedSearchCV, swept classification thresholds against a realistic business retention cost matrix ($C_{FP}=\$15$, $C_{FN}=\$100$), and evaluated the optimized model on an untouched 20% hold-out test set (1,409 customers).

**What I learned:** Lowering the decision threshold from 0.50 to 0.25 reduced validation business cost by 15.93% and produced $1,220 in net financial savings (10.72% reduction) on the hold-out test set, proving that optimizing decision thresholds against asymmetric business costs provides far greater economic value than pursuing raw accuracy.

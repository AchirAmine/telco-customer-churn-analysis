# Production Telco Customer Churn Prediction & Retention Service

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.165.0-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/pytest-19%20passed-brightgreen.svg)](https://docs.pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade machine learning system to predict telecommunication customer churn, quantify financial retention risk, and optimize decision thresholds against asymmetric business costs. Delivered with a leakage-safe scikit-learn pipeline, SHAP tree explainability, a containerized FastAPI service with strict Pydantic validation, and an interactive Streamlit operations dashboard.

---

## 1. Problem Statement

In the telecommunications sector, customer acquisition costs 5x to 7x more than customer retention. Customer churn datasets are inherently imbalanced (~26.5% churn prevalence), making standard accuracy metrics misleading and default decision thresholds ($T = 0.5$) economically suboptimal. 

A production retention service must solve three core challenges:
1. **Prevent Data Leakage:** Ensure zero information leakage across cross-validation folds during preprocessing and encoding.
2. **Optimize Financial Economics:** Calibrate decision thresholds using a business cost matrix where failing to identify a churner ($C_{FN}$) is far costlier than calling a satisfied customer ($C_{FP}$).
3. **Operationalize Inference:** Deliver low-latency predictions via a typed REST API and an explainable, reduced-feature operations dashboard.

---

## 2. Dataset & Data Quirks

* **Source:** Standard IBM / Kaggle Telco Customer Churn dataset (7,043 customer records across 21 raw columns).
* **Dataset Availability:** The raw CSV is not committed to version control per ML engineering best practices. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
* **Identified Data Quirks:**
  * **Hidden Whitespace Values:** The `TotalCharges` feature is stored as `object/string` due to 11 records containing whitespace characters (`" "`). Inspecting these records reveals they all correspond to brand new customers with `tenure = 0` who have not completed their first billing cycle. Converting `" "` to `NaN` and imputing `0.0` based on domain knowledge correctly resolves the issue without dropping valid customers.
  * **Identifier Leakage:** The `customerID` column consists of 7,043 unique alphanumeric strings and is dropped prior to feature transformation to avoid memorization.
  * **Class Imbalance:** `Churn = 'No'` accounts for 5,174 records (73.46%), while `Churn = 'Yes'` accounts for 1,869 records (26.54%).

---

## 3. Leakage-Safe Pipeline Architecture

All preprocessing transformations are strictly encapsulated inside a `ColumnTransformer` embedded within a unified `Pipeline` to eliminate data leakage during cross-validation:

* **Numeric Features (`tenure`, `MonthlyCharges`, `TotalCharges`):**
  * `SimpleImputer(strategy="median")`: Median imputation robust to skewed monetary distributions.
  * `StandardScaler()`: Zero-mean, unit-variance scaling required for regularized linear models and gradient convergence.
* **Categorical Features (Contract, InternetService, PaymentMethod, etc.):**
  * `SimpleImputer(strategy="most_frequent")`: Mode imputation for missing values.
  * `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`: Dense one-hot encoding with graceful fallback for unseen categories during production serving.

---

## 4. Model Comparison (Stratified 5-Fold Cross-Validation)

Five candidate models were benchmarked across 5 stratified folds using fixed random seeds and `class_weight="balanced"` to address class imbalance without synthetic data distortion:

| Model | ROC-AUC (Mean) | ROC-AUC (Std) | PR-AUC (Mean) | F1-Score | Brier Score (Lower is better) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | **0.8449** | ±0.0134 | **0.6554** | **0.6260** | 0.1661 |
| **HistGradientBoosting** | **0.8365** | ±0.0095 | **0.6467** | 0.6191 | **0.1587** |
| **LightGBM** | **0.8365** | ±0.0105 | **0.6474** | 0.6124 | **0.1597** |
| **Random Forest** | 0.8215 | ±0.0145 | 0.6018 | 0.5950 | 0.1562 |
| **Dummy Baseline (Prior)** | 0.5000 | ±0.0000 | 0.2654 | 0.0000 | 0.1949 |

### Key Benchmark Insights
* The Dummy Baseline's PR-AUC of **0.2654** matches the exact empirical positive class ratio ($1,869 / 7,043 = 26.54\%$), demonstrating why Average Precision is the primary baseline for imbalanced classification.
* Logistic Regression achieved the strongest ranking discrimination (ROC-AUC: 0.8449), while gradient boosting models achieved superior probability calibration (Brier Score: 0.1587).

---

## 5. Decision Threshold & Business Cost Optimization

In customer retention, decision costs are asymmetric:
* **Cost of False Positive ($C_{FP}$):** Offering a retention incentive or calling a loyal customer: **$15.00**.
* **Cost of False Negative ($C_{FN}$):** Losing customer lifetime value to a competitor: **$100.00**.

Using validation fold probabilities, the total cost was swept across thresholds $T \in [0.05, 0.95]$:

$$\text{Total Cost} = (FP \times \$15) + (FN \times \$100)$$

* **Validation Cost at default $T = 0.50$:** $46,235.00
* **Validation Cost at optimal $T = 0.25$:** **$38,870.00** (**15.93% cost reduction**)

### Final Evaluation on Untouched Hold-Out Test Set (20% Split, 1,409 Customers)
* **Test ROC-AUC:** `0.8460`
* **Test PR-AUC:** `0.6574`
* **Test Brier Score:** `0.1627`
* **Test Cost at $T = 0.50$:** $11,380.00
* **Test Cost at $T = 0.25$:** **$10,160.00**
* **Net Financial Savings:** **$1,220.00 (10.72% cost savings on test cohort alone)**

---

## 6. SHAP Explainability & Reduced Serving Model

Using `shap.TreeExplainer`, feature contributions were quantified globally and locally:
* **Top Churn Drivers:** `Contract` (Month-to-month, +0.87 SHAP impact), `tenure`, `MonthlyCharges`, `TotalCharges`, `OnlineSecurity`, `TechSupport`, and `PaymentMethod` (Electronic check).
* **Reduced Serving Pipeline:** To streamline live API payloads and operator forms, the feature set was pruned from 19 down to the **9 highest-impact features**.

### Full vs Reduced Model Comparison (Hold-Out Test Set)
* **Full Model (19 features):** ROC-AUC: `0.8334` | PR-AUC: `0.6462`
* **Reduced Model (9 features):** ROC-AUC: **`0.8353`** | PR-AUC: `0.6437`
* **Conclusion:** The reduced model retains **99.6%** of the PR-AUC while eliminating noise, reducing inference payload size, and improving test ROC-AUC.

---

## 7. Project Architecture

```text
telco-customer-churn-prediction/
├── api/
│   └── main.py                   # FastAPI REST service with Pydantic validation
├── app/
│   └── streamlit_app.py          # Interactive operations dashboard with SHAP insights
├── data/                         # Isolated dataset directory (.gitignored)
├── models/
│   └── model_v1.joblib           # Atomic production artifact (pipeline + threshold)
├── notebooks/
│   └── 01_eda.ipynb              # Executed EDA and quirk inspection notebook
├── reports/
│   └── figures/                  # Generated figures (SHAP beeswarm, waterfall, KDE)
├── src/
│   └── telco/
│       ├── __init__.py
│       ├── data.py               # Data loading, typing, and target extraction
│       ├── pipeline.py           # Leakage-safe ColumnTransformer & Pipeline
│       ├── train.py              # Candidate model definitions
│       ├── evaluate.py           # Stratified 5-fold cross-validation engine
│       ├── tune.py               # Hyperparameter tuning & cost matrix optimization
│       ├── explain.py            # Reduced feature specification & SHAP explainers
│       └── predict.py            # Unified production inference engine
├── tests/
│   ├── test_api.py               # FastAPI TestClient endpoints & 422 validations
│   ├── test_explain.py           # Reduced pipeline tests
│   ├── test_pipeline.py          # ColumnTransformer shapes & unknown categories
│   ├── test_predict.py           # Single/batch inference & risk tier logic
│   ├── test_train.py             # Candidate model & CV verification
│   └── test_tune.py              # Cost optimization logic tests
├── .dockerignore
├── .gitignore
├── Dockerfile                    # Multi-stage python:3.11-slim container with non-root user
├── LEARNING_LOG.md               # Chronological engineering progression log
├── README.md                     # Comprehensive project documentation
└── requirements.txt              # Pinned production dependencies
```

---

## 8. How to Run (PowerShell)

### Environment Setup
```powershell
git clone https://github.com/AchirAmine/telco-customer-churn-analysis.git
cd telco-customer-churn-analysis

# Create and activate Python 3.11 virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Run Test Suite (19 Automated Tests)
```powershell
python -m pytest -v
```

### Run FastAPI Service Locally
```powershell
uvicorn api.main:app --reload --port 8000
```
* Interactive Swagger Docs: `http://localhost:8000/docs`
* Health Check: `http://localhost:8000/health`

### Run Streamlit Operations Dashboard
```powershell
streamlit run app/streamlit_app.py
```
* Dashboard URL: `http://localhost:8501`

---

## 9. Docker Containerization

The production Docker image is built on `python:3.11-slim` with layered dependency caching and runs under an unprivileged `appuser` (UID 1000) for container security:

```powershell
# Build Docker image
docker build -t telco-churn-api:v1 .

# Run container
docker run -d -p 8000:8000 --name telco-api telco-churn-api:v1

# Verify service health
curl http://localhost:8000/health

# Stop and remove container
docker rm -f telco-api
```

---

## 10. API Specification & Example Payload

### `POST /predict`
**Request Headers:** `Content-Type: application/json`

**Sample Payload:**
```json
{
  "tenure": 2,
  "MonthlyCharges": 85.50,
  "TotalCharges": 171.00,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "TechSupport": "No",
  "PaymentMethod": "Electronic check",
  "PaperlessBilling": "Yes"
}
```

**Response (`HTTP 200 OK`):**
```json
{
  "churn_probability": 0.7612,
  "churn_prediction": 1,
  "risk_tier": "High",
  "model_version": "1.0.0"
}
```

---

## 11. Limitations & Future Improvements

1. **Static Cost Assumptions:** The current cost model uses static values ($C_{FP}=\$15$, $C_{FN}=\$100$). A production extension would dynamically compute customer-specific lifetime value (CLV) based on `tenure` and `MonthlyCharges`.
2. **Survival Analysis:** Incorporating survival models (Cox Proportional Hazards) would estimate *when* a customer is expected to churn, rather than a binary snapshot.
3. **Automated Monitoring:** Adding data drift detection (Evidently AI) and OpenTelemetry request logging.

---

## 12. Author & Attribution

* **Author:** Developed by **Achir Mohamed Amine** — AI & Data Science Engineering Student at ESI-SBA (Algeria). Built for production ML engineering portfolio and international internship applications.
* **Dataset Attribution:** IBM Cognos Analytics sample dataset, distributed via Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

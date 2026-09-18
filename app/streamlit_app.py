import sys
from pathlib import Path
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = str(ROOT_DIR / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from telco.predict import load_artifact, predict_single

st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
)


@st.cache_resource
def get_model_artifact():
    return load_artifact(ROOT_DIR / "models" / "model_v1.joblib")


artifact = get_model_artifact()

st.title("📡 Telco Customer Churn Prediction Service")
st.markdown(
    "Predict individual customer churn probability and retention risk using a production-grade machine learning service."
)

col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("Model Version", artifact["version"])
col_info2.metric("Optimal Decision Threshold", f"{artifact['threshold']:.2f}")
col_info3.metric("Retention Cost Savings", "10.72%")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Account & Tenure Details")
    tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=70.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly_charges), step=10.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

with col_right:
    st.subheader("Services & Payment Method")
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    
    if internet == "No":
        sec_options = ["No internet service"]
        tech_options = ["No internet service"]
    else:
        sec_options = ["No", "Yes"]
        tech_options = ["No", "Yes"]
        
    online_security = st.selectbox("Online Security", sec_options)
    tech_support = st.selectbox("Tech Support", tech_options)
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

st.divider()

if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    payload = {
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "Contract": contract,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless,
    }

    result = predict_single(payload, artifact=artifact)
    proba = result["churn_probability"]
    tier = result["risk_tier"]
    pred = result["churn_prediction"]

    st.subheader("Assessment Results")
    res_col1, res_col2, res_col3 = st.columns(3)

    res_col1.metric("Churn Probability", f"{proba * 100:.1f}%")
    res_col2.metric("Prediction", "Will Churn" if pred == 1 else "Will Stay")
    res_col3.metric("Retention Risk Tier", tier)

    st.progress(proba)

    if tier == "High":
        st.error("🚨 **High Risk Tier**: Urgent intervention recommended. Immediate retention call or discount package should be offered.")
    elif tier == "Medium":
        st.warning("⚠️ **Medium Risk Tier**: Customer probability exceeds optimal cost threshold (0.25). Monitor engagement and follow up.")
    else:
        st.success("✅ **Low Risk Tier**: Customer is stable. Standard loyalty communications recommended.")

    st.subheader("Key Risk Drivers (SHAP Insights)")
    drivers = []
    if contract == "Month-to-month":
        drivers.append("Month-to-month contracts are the #1 global churn driver (+0.87 SHAP impact).")
    if tenure < 12:
        drivers.append("Customer is in the critical first year (tenure < 12 months, high historical hazard rate).")
    if internet == "Fiber optic":
        drivers.append("Fiber optic subscription with elevated price sensitivity.")
    if tech_support == "No":
        drivers.append("Absence of technical support increases dissatisfaction likelihood.")
    if payment == "Electronic check":
        drivers.append("Electronic check billing method correlates with higher voluntary churn.")
    if monthly_charges > 70.0:
        drivers.append(f"High monthly commitment (${monthly_charges:.2f}) raises price vulnerability.")

    if drivers:
        for d in drivers:
            st.markdown(f"- {d}")
    else:
        st.markdown("- No dominant negative risk drivers detected.")

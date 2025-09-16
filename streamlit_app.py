import os
import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = os.path.join("models", "model.joblib")

st.set_page_config(page_title="US Visa Outcome Classifier", layout="centered")
st.title("US Visa Application Outcome Classifier")
st.write("Upload a CSV of applications or enter a single applicant to get a sample prediction.")

# Sidebar: single input demo
st.sidebar.header("Single applicant demo")
applicant_wage = st.sidebar.number_input("Prevailing wage (USD)", min_value=0.0, value=50000.0)
if st.sidebar.button("Predict single"):
    # simple rule-based fallback
    pred = "Certified" if applicant_wage >= 50000 else "Denied"
    st.sidebar.success(f"Prediction: {pred}")

# File upload
uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.subheader("Preview")
        st.write(df.head())
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            preds = model.predict(df)
            st.subheader("Predictions")
            st.write(pd.DataFrame({"prediction": preds}))
        else:
            st.warning("No trained model found. Using a simple sample rule for predictions.")
            if "PREVAILING_WAGE" in df.columns:
                preds = ["Certified" if w >= 50000 else "Denied" for w in df["PREVAILING_WAGE"]]
            else:
                preds = ["Certified" for _ in range(len(df))]
            st.subheader("Sample predictions")
            st.write(pd.DataFrame({"prediction": preds}))
    except Exception as e:
        st.error(f"Failed to read file: {e}")

st.markdown("---")
st.info("Place a trained model at `models/model.joblib` for real predictions. Training scripts live in `src/` in future commits.")

# app.py

import streamlit as st
import pandas as pd
import joblib
import boto3
import os
from preprocessing import preprocess_inference

# ------------------- S3 Settings -------------------
BUCKET = "ml-rvce-us-east-1"
ARTIFACTS = {
    "model": "fraud-detection/models/fraud_model.pkl",
    "scaler": "fraud-detection/models/scaler.pkl",
    "hourly": "fraud-detection/models/hourly_stats.pkl"
}

LOCAL_PATHS = {
    "model": "fraud_model.pkl",
    "scaler": "scaler.pkl",
    "hourly": "hourly_stats.pkl"
}

# ------------------- Helpers -------------------
def download_from_s3():
    """Download model + preprocessing artifacts from S3 if not already present."""
    s3 = boto3.client("s3")
    for key, s3_key in ARTIFACTS.items():
        if not os.path.exists(LOCAL_PATHS[key]):
            s3.download_file(BUCKET, s3_key, LOCAL_PATHS[key])

@st.cache_resource
def load_artifacts():
    """Ensure artifacts are downloaded and return loaded objects."""
    download_from_s3()
    model = joblib.load(LOCAL_PATHS["model"])
    # Preprocessing artifacts will be loaded inside preprocess_inference
    return model

# ------------------- Load Model -------------------
model = load_artifacts()

# ------------------- Streamlit UI -------------------
st.title("💳 Fraud Detection App (S3-powered)")

st.markdown(
    """
    This app predicts whether a given credit card transaction is **fraudulent or legitimate**  
    using a model trained in **Amazon SageMaker** and deployed via **Streamlit Cloud**.
    """
)

# User input form
with st.form("transaction_form"):
    amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0)
    time = st.number_input("Transaction Time (seconds since start)", min_value=0, step=1)
    submitted = st.form_submit_button("Predict Fraud?")

if submitted:
    # Construct dataframe
    transaction = pd.DataFrame([{"Amount": amount, "Time": time}])

    # Apply preprocessing
    transaction = preprocess_inference(transaction)

    # Make prediction
    pred = model.predict(transaction)[0]
    prob = model.predict_proba(transaction)[0, 1]

    # Display result
    if pred == 1:
        st.error(f"Fraudulent Transaction Detected (prob={prob:.2f})")
    else:
        st.success(f"Legitimate Transaction (prob={prob:.2f})")

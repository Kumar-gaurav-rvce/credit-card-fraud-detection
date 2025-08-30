import streamlit as st
import joblib
from preprocessing import preprocess_inference
import boto3
import os

# ------------------- Model S3 download -------------------
BUCKET = "ml-rvce-us-east-1"
MODEL_KEY = "fraud-detection/models/model.pkl"
LOCAL_MODEL_PATH = "artifacts/model.pkl"

os.makedirs("artifacts", exist_ok=True)
s3 = boto3.client("s3")
if not os.path.exists(LOCAL_MODEL_PATH):
    s3.download_file(BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)

# ------------------- Load model -------------------
model = joblib.load(LOCAL_MODEL_PATH)

# ------------------- Streamlit UI -------------------
st.title("Fraud Detection App (Full Features)")
st.write("Predict fraud using Time + Amount (other features filled automatically)")

# Inputs
amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0)
time = st.number_input("Transaction Time (seconds since start)", min_value=0, step=1)

if st.button("Predict"):
    transaction = {"Time": time, "Amount": amount}
    X = preprocess_inference(transaction)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0, 1]

    if pred == 1:
        st.error(f"Fraudulent Transaction Detected (prob={prob:.2f})")
    else:
        st.success(f"Legitimate Transaction (prob={prob:.2f})")

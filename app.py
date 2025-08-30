# app.py
import streamlit as st
import joblib
from preprocessing import preprocess_inference
import boto3
import os

MODEL_KEY = "fraud-detection/models/model.pkl"
LOCAL_MODEL_PATH = "artifacts/model.pkl"
BUCKET = "ml-rvce-us-east-1"

# Download model from S3 if missing
os.makedirs("artifacts", exist_ok=True)
s3 = boto3.client("s3")
if not os.path.exists(LOCAL_MODEL_PATH):
    s3.download_file(BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)

model = joblib.load(LOCAL_MODEL_PATH)

st.title("💳 Fraud Detection App (S3-powered)")
st.write("Predict fraud using Amount + Time only")

amount = st.text_input("Transaction Amount")
time = st.text_input("Transaction Time (seconds since start)")

if st.button("Predict"):
    try:
        transaction = {"Amount": float(amount), "Time": float(time)}
        X = preprocess_inference(transaction)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0, 1]

        if pred == 1:
            st.error(f"Fraudulent Transaction Detected (prob={prob:.2f})")
        else:
            st.success(f"Legitimate Transaction (prob={prob:.2f})")

    except Exception as e:
        st.error(f"Error: {e}")

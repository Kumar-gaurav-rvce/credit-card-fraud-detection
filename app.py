# app.py
import streamlit as st
import joblib
from preprocessing import preprocess_inference

MODEL_PATH = "artifacts/model.pkl"

st.title("Fraud Detection App (S3-powered)")
st.write("Predict fraud using Amount + Time only")

# Inputs
amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0)
time = st.number_input("Transaction Time (seconds since start)", min_value=0, step=1)

# Load model
model = joblib.load(MODEL_PATH)

if st.button("Predict"):
    transaction = {"Amount": amount, "Time": time}
    X = preprocess_inference(transaction)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0, 1]

    if pred == 1:
        st.error(f"Fraudulent Transaction Detected (prob={prob:.2f})")
    else:
        st.success(f"Legitimate Transaction (prob={prob:.2f})")

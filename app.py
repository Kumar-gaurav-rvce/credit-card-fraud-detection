import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_inference

MODEL_PATH = "artifacts/model.pkl"

st.title("Fraud Detection App (S3-powered)")
st.write("Predict fraud for single or multiple transactions (CSV upload supported)")

# Option 1: Single transaction input
st.subheader("Single Transaction Input")
amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0)
time = st.number_input("Transaction Time (seconds since start)", min_value=0, step=1)

# Option 2: CSV upload
st.subheader("Or Upload CSV")
uploaded_file = st.file_uploader("Upload CSV with same columns as training features", type=["csv"])

# Load model
model = joblib.load(MODEL_PATH)

# Prepare data
transactions = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    transactions = df
else:
    transactions = {"Time": time, "Amount": amount}  # Can add V1-V28 if needed

# Predict
if st.button("Predict"):
    try:
        X = preprocess_inference(transactions)
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        # Display results
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            if uploaded_file is not None:
                st.write(f"Transaction {i+1}: {'Fraudulent' if pred==1 else 'Legitimate'} (prob={prob:.2f})")
            else:
                st.write(f"Transaction: {'Fraudulent' if pred==1 else 'Legitimate'} (prob={prob:.2f})")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

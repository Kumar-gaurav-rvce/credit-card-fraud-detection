# app.py
import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_inference

MODEL_PATH = "artifacts/model.pkl"

st.title("Fraud Detection App ")
st.write("Predict fraud using raw transaction data (supports single or multiple rows)")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Or input a single transaction manually
st.subheader("Or enter a single transaction manually")
amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0, key="amount")
time = st.number_input("Transaction Time (seconds since start)", min_value=0, step=1, key="time")

# Load model
model = joblib.load(MODEL_PATH)

if st.button("Predict"):
    try:
        if uploaded_file is not None:
            # Read uploaded CSV
            df = pd.read_csv(uploaded_file)
            X = preprocess_inference(df)
        else:
            # Single transaction
            transaction = {"Amount": amount, "Time": time}
            X = preprocess_inference(transaction)

        # Predict
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        # Display results
        if uploaded_file is not None:
            st.subheader("Predictions for uploaded CSV")
            results = df.copy()
            results['Prediction'] = preds
            results['Fraud_Prob'] = probs
            st.dataframe(results)
        else:
            if preds[0] == 1:
                st.error(f"Fraudulent Transaction Detected (prob={probs[0]:.2f})")
            else:
                st.success(f"Legitimate Transaction (prob={probs[0]:.2f})")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

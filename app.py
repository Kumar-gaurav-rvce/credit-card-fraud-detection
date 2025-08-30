# app.py
import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_inference

MODEL_PATH = "artifacts/model.pkl"

st.title("💳 Fraud Detection App (CSV Upload)")

st.markdown("""
Upload a CSV file with credit card transactions.  
The CSV must include the following columns: **Time, Amount, V1–V28**.  
The app will predict whether each transaction is **fraudulent** or **legitimate**.
""")

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Preprocess
    X = preprocess_inference(df)

    # Load model
    model = joblib.load(MODEL_PATH)

    # Predict
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    # Add predictions to dataframe
    df["Prediction"] = preds
    df["Fraud_Probability"] = probs

    st.subheader("Predictions")
    st.dataframe(df)

    # Option to download predictions
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

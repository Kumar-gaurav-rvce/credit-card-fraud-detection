import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_inference
from inference import predict_transaction

MODEL_PATH = "artifacts/model.pkl"

st.title("Fraud Detection App (S3-powered)")
st.write("Upload a CSV file with transactions to predict fraud.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    # Read CSV normally
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", df)

    try:
        # Pass the DataFrame directly to inference
        preds, probs = predict_transaction(df)

        results = df.copy()
        results["Prediction"] = preds
        results["Fraud_Prob"] = probs
        st.write("Prediction Results", results)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

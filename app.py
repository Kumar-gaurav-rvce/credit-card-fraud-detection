# app.py
import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_inference

MODEL_PATH = "artifacts/model.pkl"

st.title("Fraud Detection App (S3-powered)")
st.write("Upload a CSV file with raw transaction data (Time, V1-V28, Amount)")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Preprocess
    try:
        X = preprocess_inference(df)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        st.stop()

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Predict
    try:
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Display results
    results = df.copy()
    results['Prediction'] = preds
    results['Fraud_Probability'] = probs
    st.write(results)

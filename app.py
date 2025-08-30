import streamlit as st
import pandas as pd
from preprocessing import preprocess_inference
import joblib

MODEL_PATH = "artifacts/model.pkl"
model = joblib.load(MODEL_PATH)

st.title("Fraud Detection App (S3-powered)")
st.write("Upload CSV with transactions to predict fraud")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    try:
        X = preprocess_inference(df)  # pass DataFrame directly, no wrapping!
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        # Show results
        results = df.copy()
        results['Prediction'] = preds
        results['Fraud_Prob'] = probs
        st.dataframe(results)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

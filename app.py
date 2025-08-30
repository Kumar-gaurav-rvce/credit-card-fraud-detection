import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_inference

MODEL_PATH = "artifacts/model.pkl"
model = joblib.load(MODEL_PATH)

st.title("Fraud Detection App (S3-powered)")
st.write("Upload a CSV file with transactions to predict fraud.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", df)

    try:
        # preprocess to 2D numpy array
        X = preprocess_inference(df)
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        results = df.copy()
        results["Prediction"] = preds
        results["Fraud_Prob"] = probs
        st.write("Prediction Results", results)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

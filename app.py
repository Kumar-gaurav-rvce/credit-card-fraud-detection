import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_inference

MODEL_PATH = "artifacts/model.pkl"

st.title("Fraud Detection App (S3-powered)")
st.write("Upload a CSV file with transaction(s) to predict fraud")

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(df)

        # Preprocess
        X = preprocess_inference(df)

        # Load model
        model = joblib.load(MODEL_PATH)

        # Predict
        pred = model.predict(X)
        prob = model.predict_proba(X)[:, 1]

        # Display results
        results = df.copy()
        results["Prediction"] = pred
        results["Fraud_Probability"] = prob
        results["Prediction"] = results["Prediction"].map({0: "Legitimate", 1: "Fraudulent"})

        st.write("Prediction Results:")
        st.dataframe(results)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

import streamlit as st
import pandas as pd
from inference import predict_transaction

st.title("Fraud Detection App")
st.write("Upload a CSV file with transactions to predict fraud.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", df)

    try:
        preds, probs = predict_transaction(df)
        results = df.copy()
        results["Prediction"] = preds
        results["Fraud_Prob"] = probs
        st.write("Prediction Results", results)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

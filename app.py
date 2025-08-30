import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_inference

MODEL_PATH = "artifacts/model.pkl"

st.title("Fraud Detection App (S3-powered)")
st.write("Predict fraud using uploaded CSV (1 or more rows)")

# CSV upload
uploaded_file = st.file_uploader(
    "Upload CSV with the same columns as training features (Time, V1–V28, Amount)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(df)

        # Load model
        model = joblib.load(MODEL_PATH)

        # Preprocess
        X = preprocess_inference(df)

        # Predict
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        # Display results
        results = df.copy()
        results["Prediction"] = preds
        results["Fraud_Prob"] = probs
        results["Prediction_Label"] = results["Prediction"].apply(lambda x: "Fraudulent" if x==1 else "Legitimate")

        st.subheader("Prediction Results")
        st.dataframe(results)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload a CSV file to make predictions.")

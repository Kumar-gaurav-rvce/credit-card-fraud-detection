import streamlit as st
import pandas as pd
from inference import predict_transaction

# ----------------------------------------
# Streamlit app title and description
# ----------------------------------------
st.title("Fraud Detection App")
st.write("Upload a CSV file containing transactions to predict whether they are fraudulent or legitimate.")

# ----------------------------------------
# File uploader widget
# ----------------------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Only run if a file has been uploaded
if uploaded_file is not None:
    # Read CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", df)  # Display the uploaded data

    try:
        # ----------------------------------------
        # Predict fraud using the inference module
        # ----------------------------------------
        preds, probs = predict_transaction(df)

        # ----------------------------------------
        # Add predictions to the original DataFrame
        # ----------------------------------------
        results = df.copy()
        results["Prediction"] = preds       # Predicted class: 0 = Legitimate, 1 = Fraud
        results["Fraud_Prob"] = probs       # Predicted fraud probability

        # Display the results
        st.write("Prediction Results", results)

    except Exception as e:
        # If something goes wrong, display the error
        st.error(f"Error during prediction: {e}")

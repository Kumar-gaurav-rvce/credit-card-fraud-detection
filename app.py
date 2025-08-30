import streamlit as st
from inference import predict_transaction

st.title("💳 Credit Card Fraud Detection")
st.write("Predict fraud using raw transaction data")

amount = st.number_input("Transaction Amount", min_value=0.0)
time = st.number_input("Transaction Time (seconds since start)", min_value=0)

if st.button("Predict"):
    transaction = {"Amount": amount, "Time": time}
    pred, prob = predict_transaction(transaction)
    
    if pred == 1:
        st.error(f"Fraudulent Transaction Detected (prob={prob:.2f})")
    else:
        st.success(f"Legitimate Transaction (prob={prob:.2f})")

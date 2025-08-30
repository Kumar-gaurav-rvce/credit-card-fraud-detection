#inference.py
import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")  # or download from S3

def predict_transaction(transaction_dict):
    """
    transaction_dict: {"Amount": 100, "Time": 123456, ...}
    Returns: prediction (0=normal, 1=fraud) and probability
    """
    df = pd.DataFrame([transaction_dict])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0,1]
    return pred, prob

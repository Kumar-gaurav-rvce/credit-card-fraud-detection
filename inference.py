import joblib
import pandas as pd
from preprocessing import preprocess_inference

MODEL_PATH = "artifacts/fraud_model.pkl"
model = joblib.load(MODEL_PATH)

def predict_transaction(transaction_dict):
    """
    Predict fraud for a single transaction.
    Returns: prediction (0=legit,1=fraud) and probability
    """
    X = preprocess_inference(transaction_dict)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0, 1]
    return pred, prob

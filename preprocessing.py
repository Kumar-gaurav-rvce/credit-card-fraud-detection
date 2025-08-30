# preprocessing.py
import pandas as pd
import joblib
import numpy as np

SCALER_PATH = "artifacts/scaler.pkl"

def preprocess_inference(transaction):
    """
    transaction: either a dict (single transaction) or a DataFrame with all features
    Returns a 2D NumPy array suitable for model.predict
    """
    scaler = joblib.load(SCALER_PATH)

    # If transaction is a dict, convert to DataFrame
    if isinstance(transaction, dict):
        transaction_df = pd.DataFrame([transaction])
    elif isinstance(transaction, pd.DataFrame):
        transaction_df = transaction.copy()
    else:
        raise ValueError("transaction must be dict or pd.DataFrame")

    # Ensure all columns match scaler's expected features
    expected_cols = scaler.feature_names_in_
    missing_cols = set(expected_cols) - set(transaction_df.columns)
    for col in missing_cols:
        transaction_df[col] = 0  # fill missing columns with 0

    # Reorder columns
    transaction_df = transaction_df[expected_cols]

    # Scale
    features_scaled = scaler.transform(transaction_df)

    return features_scaled  # always 2D

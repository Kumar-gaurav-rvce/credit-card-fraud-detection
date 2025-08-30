# preprocessing.py
import pandas as pd
import joblib
import numpy as np

SCALER_PATH = "artifacts/scaler.pkl"

def preprocess_inference(transaction_input):
    """
    Accepts:
        - dict (single transaction)
        - list of dicts
        - pandas DataFrame (from CSV upload)
    Returns:
        2D numpy array: shape (n_samples, n_features)
    """
    # Convert input to DataFrame
    if isinstance(transaction_input, dict):
        df = pd.DataFrame([transaction_input])  # single row
    elif isinstance(transaction_input, list):
        df = pd.DataFrame(transaction_input)    # multiple rows
    elif isinstance(transaction_input, pd.DataFrame):
        df = transaction_input.copy()          # already DataFrame
    else:
        raise ValueError("Input must be dict, list of dicts, or DataFrame")

    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Ensure columns match scaler's feature names
    feature_cols = getattr(scaler, "feature_names_in_", df.columns)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0  # fill missing features
    df = df[feature_cols].fillna(0)

    # Convert to 2D numpy array
    X = df.to_numpy()
    # Ensure shape is exactly 2D (n_samples, n_features)
    if X.ndim == 1:
        X = X.reshape(1, -1)  # single row
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)  # flatten extra dimensions

    return X

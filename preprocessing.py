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
        df = pd.DataFrame([transaction_input])
    elif isinstance(transaction_input, list):
        df = pd.DataFrame(transaction_input)
    elif isinstance(transaction_input, pd.DataFrame):
        df = transaction_input.copy()
    else:
        raise ValueError("Input must be dict, list of dicts, or DataFrame")

    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Make sure DataFrame has all features expected by the scaler
    feature_cols = getattr(scaler, "feature_names_in_", df.columns)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols].fillna(0)

    # Convert to 2D array
    X = df.to_numpy()
    # If somehow it ends up 3D, squeeze it
    if X.ndim > 2:
        X = np.squeeze(X, axis=0)
    return X

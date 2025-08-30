# preprocessing.py
import pandas as pd
import joblib
import numpy as np

SCALER_PATH = "artifacts/scaler.pkl"

def preprocess_inference(transaction_input):
    if isinstance(transaction_input, dict):
        df = pd.DataFrame([transaction_input])
    elif isinstance(transaction_input, list):
        df = pd.DataFrame(transaction_input)
    elif isinstance(transaction_input, pd.DataFrame):
        df = transaction_input.copy()
    else:
        raise ValueError("Input must be dict, list of dicts, or DataFrame")

    scaler = joblib.load(SCALER_PATH)

    feature_cols = getattr(scaler, "feature_names_in_", df.columns)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols].fillna(0)

    X = df.to_numpy()

    # Flatten if mistakenly 3D
    if X.ndim == 3:
        X = X.reshape(X.shape[1], X.shape[2])
    elif X.ndim == 1:
        X = X.reshape(1, -1)

    return X


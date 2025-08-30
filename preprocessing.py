# preprocessing.py
import pandas as pd
import joblib
import numpy as np

SCALER_PATH = "artifacts/scaler.pkl"

def preprocess_inference(transaction_input):
    import pandas as pd
    import joblib

    # Convert to DataFrame
    if isinstance(transaction_input, dict):
        df = pd.DataFrame([transaction_input])
    elif isinstance(transaction_input, list):
        df = pd.DataFrame(transaction_input)
    elif isinstance(transaction_input, pd.DataFrame):
        df = transaction_input.copy()
    else:
        raise ValueError("Input must be dict, list of dicts, or DataFrame")

    # Drop target column if present
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])

    # Load scaler
    scaler = joblib.load("artifacts/scaler.pkl")

    # Ensure all required feature columns exist
    feature_cols = getattr(scaler, "feature_names_in_", df.columns)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols].fillna(0)

    # Convert to 2D numpy array
    X = df.to_numpy()
    if X.ndim == 1:
        X = X.reshape(1, -1)  # single row
    # **Do not reshape 3D arrays** — DataFrame to_numpy already gives 2D
    return X



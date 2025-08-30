import pandas as pd
import joblib

SCALER_PATH = "artifacts/scaler.pkl"

def preprocess_inference(transaction_input):
    """
    Preprocess transaction(s) for inference.
    Accepts:
        - dict (single transaction)
        - list of dicts
        - pandas DataFrame
    Returns: 2D numpy array (n_samples, n_features)
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

    # Ensure feature order matches scaler
    feature_cols = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else df.columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0  # fill missing features

    df = df[feature_cols].fillna(0)

    # Convert to 2D numpy array
    X = df.to_numpy()
    return X

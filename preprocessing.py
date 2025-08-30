import pandas as pd
import joblib

SCALER_PATH = "artifacts/scaler.pkl"

def preprocess_inference(transaction_input):
    """
    Accepts:
        - dict (single transaction)
        - list of dicts (multiple transactions)
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

    # Drop target column if present
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])

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

    # ✅ Make absolutely sure it's 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim == 3:
        X = X.reshape(X.shape[1], X.shape[2])  # (1, n_rows, n_features) → (n_rows, n_features)

    return X


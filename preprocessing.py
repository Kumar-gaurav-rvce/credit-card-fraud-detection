# preprocessing.py
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
        df = pd.DataFrame([transaction_input])  # single row
    elif isinstance(transaction_input, list):
        df = pd.DataFrame(transaction_input)    # multiple rows
    elif isinstance(transaction_input, pd.DataFrame):
        df = transaction_input.copy()          # already DataFrame
    else:
        raise ValueError("Input must be dict, list of dicts, or DataFrame")

    # Drop target column if present
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])

    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Align DataFrame columns with scaler’s feature names
    feature_cols = list(scaler.feature_names_in_)
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Convert to 2D numpy array
    X = df.to_numpy()
    return X

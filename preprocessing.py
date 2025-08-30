import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

SCALER_PATH = "artifacts/scaler.pkl"

def preprocess_training(df):
    """
    Preprocess the raw dataframe for training:
    - Fill missing values
    - Scale numeric columns
    """
    feature_cols = [c for c in df.columns if c != 'Class']

    df[feature_cols] = df[feature_cols].fillna(0)

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    joblib.dump(scaler, SCALER_PATH)
    print("Scaler saved to", SCALER_PATH)

    return df

def preprocess_inference(transaction_input):
    """
    Preprocess transaction(s) for inference.
    Accepts:
    - pandas DataFrame directly (from CSV upload)
    - dict (single transaction)
    - list of dicts
    Returns: 2D numpy array (n_samples, n_features)
    """
    # Convert input to DataFrame
    if isinstance(transaction_input, dict):
        df = pd.DataFrame([transaction_input])  # single row
    elif isinstance(transaction_input, list):
        df = pd.DataFrame(transaction_input)    # multiple rows
    elif isinstance(transaction_input, pd.DataFrame):
        df = transaction_input.copy()          # CSV upload
    else:
        raise ValueError("Input must be dict, list of dicts, or DataFrame")

    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Ensure DataFrame columns match the scaler's features
    feature_cols = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else df.columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0  # fill missing columns with zeros

    df = df[feature_cols].fillna(0)

    # Ensure output is 2D: (n_samples, n_features)
    X = df.to_numpy()
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X

# preprocessing.py
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
    - pandas DataFrame directly
    - dict (single transaction)
    - list of dicts
    Returns: 2D numpy array
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
            df[col] = 0  # fill missing features with zeros

    df = df[feature_cols].fillna(0)

    # Make sure shape is 2D: (n_samples, n_features)
    X = df.values
    if len(X.shape) != 2:
        X = X.reshape(X.shape[0], -1)

    return X

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

    # Fill missing values
    df[feature_cols] = df[feature_cols].fillna(0)

    # Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print("Scaler saved to", SCALER_PATH)

    return df

def preprocess_inference(transaction_input):
    """
    Preprocess transaction(s) for inference.
    Accepts:
    - dict: single transaction
    - list of dicts
    - pandas DataFrame
    Returns: scaled numpy array
    """
    # Convert to DataFrame if input is dict or list
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

    # Ensure all scaler features are present
    feature_cols = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else df.columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0  # fill missing columns with zeros

    df = df[feature_cols].fillna(0)

    return scaler.transform(df)

# preprocessing.py
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

SCALER_PATH = "artifacts/scaler.pkl"

def preprocess_training(df):
    """
    Preprocess the raw dataframe for training:
    - Fill missing values (if any)
    - Scale numeric columns
    """
    # Assuming raw CSV has 'Time', 'Amount', 'V1'..'V28', 'Class'
    feature_cols = [c for c in df.columns if c != 'Class']

    # Fill missing values
    df[feature_cols] = df[feature_cols].fillna(0)

    # Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Save scaler for inference
    joblib.dump(scaler, SCALER_PATH)
    print("Scaler saved to", SCALER_PATH)

    return df

def preprocess_inference(transaction):
    """
    Preprocess a transaction for inference
    - transaction: dict (single transaction) OR pd.DataFrame (multiple rows)
    """
    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Single transaction dict
    if isinstance(transaction, dict):
        features = np.array([[transaction["Time"], transaction["Amount"]]])
    else:
        # DataFrame with columns matching training features
        features = transaction.values  # shape (n_samples, n_features)

    # Scale
    features_scaled = scaler.transform(features)
    return features_scaled

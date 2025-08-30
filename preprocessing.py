import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

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

def preprocess_inference(transaction_df):
    """
    Preprocess a transaction dataframe for inference
    - transaction_df: pd.DataFrame or pd.Series (single transaction)
    """
    scaler = joblib.load(SCALER_PATH)

    # If user passed a single row dict/series, convert to 2D DataFrame
    if isinstance(transaction_df, pd.Series):
        transaction_df = transaction_df.to_frame().T

    elif isinstance(transaction_df, dict):
        transaction_df = pd.DataFrame([transaction_df])

    # Scale features
    features_scaled = scaler.transform(transaction_df)
    return pd.DataFrame(features_scaled, columns=transaction_df.columns)

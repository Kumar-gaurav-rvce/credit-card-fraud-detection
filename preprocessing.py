# preprocessing.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

SCALER_PATH = "artifacts/scaler.pkl"

def preprocess_training(df):
    """
    Preprocess the raw dataframe for training:
    - Fill missing values (if any)
    - Scale numeric columns
    """
    feature_cols = [c for c in df.columns if c != 'Class']

    # Fill missing values
    df[feature_cols] = df[feature_cols].fillna(0)

    # Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Save scaler for inference
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print("Scaler saved to", SCALER_PATH)

    return df

def preprocess_inference(transaction_df):
    """
    Preprocess a transaction dataframe for inference
    - Handles missing columns by filling with 0
    - Ensures correct column order
    """
    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Get training feature order from scaler
    feature_cols = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else transaction_df.columns

    # Ensure all training features exist in transaction_df
    for col in feature_cols:
        if col not in transaction_df.columns:
            transaction_df[col] = 0  # fill missing columns with 0

    # Reorder columns to match training
    transaction_df = transaction_df[feature_cols]

    # Fill any remaining NaNs
    transaction_df = transaction_df.fillna(0)

    # Scale
    features_scaled = scaler.transform(transaction_df)

    return features_scaled

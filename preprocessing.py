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

def preprocess_inference(transaction_df):
    """
    Preprocess a transaction DataFrame or dict for inference.
    Returns a 2D NumPy array suitable for model.predict.
    """

    scaler = joblib.load("artifacts/scaler.pkl")
    
    # Convert dict to DataFrame if needed
    if isinstance(transaction_df, dict):
        transaction_df = pd.DataFrame([transaction_df])

    # Ensure columns match scaler's training columns
    expected_cols = scaler.feature_names_in_  # sklearn stores feature names in 1.0+
    missing_cols = set(expected_cols) - set(transaction_df.columns)
    if missing_cols:
        # Fill missing columns with zeros
        for col in missing_cols:
            transaction_df[col] = 0

    # Reorder columns to match training
    transaction_df = transaction_df[expected_cols]

    # Scale features
    features_scaled = scaler.transform(transaction_df)

    return features_scaled  # 2D array, ready for model.predict



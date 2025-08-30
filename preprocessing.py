# preprocessing.py
import pandas as pd
import joblib
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
    Preprocess a transaction dataframe for inference
    - transaction_df: pd.DataFrame with same columns as training features
    """
    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    features_scaled = scaler.transform(transaction_df)
    return features_scaled

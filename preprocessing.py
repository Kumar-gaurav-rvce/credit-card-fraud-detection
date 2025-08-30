import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")

def preprocess_training(df):
    """
    Preprocess raw CSV:
    - Scale 'Amount' and 'Time'
    - Feature engineering (hour of day, ratios)
    """
    # Scale 'Amount' and 'Time'
    scaler = StandardScaler()
    df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])
    
    # Feature engineering: example
    df['Hour'] = (df['Time'] // 3600) % 24
    df['MeanAmountByHour'] = df.groupby('Hour')['Amount'].transform('mean')
    df['Amount_vs_HourlyMean'] = df['Amount'] / (df['MeanAmountByHour'] + 1e-6)

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)

    return df

def preprocess_inference(transaction):
    """
    Preprocess a single transaction for prediction
    transaction: dict {'Amount': ..., 'Time': ...}
    """
    scaler = joblib.load(SCALER_PATH)
    df = pd.DataFrame([transaction])
    df[['Amount', 'Time']] = scaler.transform(df[['Amount', 'Time']])
    
    # Feature engineering same as training
    df['Hour'] = (df['Time'] // 3600) % 24
    # Use hourly mean from training if available, else 0
    df['MeanAmountByHour'] = 0
    df['Amount_vs_HourlyMean'] = df['Amount']
    
    return df

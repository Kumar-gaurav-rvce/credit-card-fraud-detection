import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# File paths for artifacts
SCALER_PATH = "artifacts/scaler.pkl"
HOURLY_STATS_PATH = "artifacts/hourly_stats.pkl"

def preprocess_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing pipeline for training:
    - Standardize 'Amount'
    - Extract 'Hour' from 'Time'
    - Compute mean transaction amount per hour
    - Save scaler + hourly stats for inference use
    """
    # Scale Amount
    scaler = StandardScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
    joblib.dump(scaler, SCALER_PATH)  # save scaler
    
    # Create hour feature
    df['Hour'] = (df['Time'] // 3600) % 24
    
    # Compute mean by hour
    hourly_means = df.groupby('Hour')['Amount'].mean().to_dict()
    joblib.dump(hourly_means, HOUR_STATS_PATH)  # save dict
    
    # Add new features
    df['MeanAmountByHour'] = df['Hour'].map(hourly_means)
    df['Amount_vs_HourlyMean'] = df['Amount'] / (df['MeanAmountByHour'] + 1e-6)
    
    return df


def preprocess_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing pipeline for inference:
    - Load scaler + hourly stats
    - Apply same transformations as training
    """
    # Load scaler + stats
    scaler = joblib.load(SCALER_PATH)
    hourly_means = joblib.load(HOUR_STATS_PATH)
    
    # Apply scaler
    df['Amount_Scaled'] = scaler.transform(df[['Amount']])
    
    # Extract Hour
    df['Hour'] = (df['Time'] // 3600) % 24
    
    # Map stats
    df['MeanAmountByHour'] = df['Hour'].map(hourly_means).fillna(0)
    df['Amount_vs_HourlyMean'] = df['Amount'] / (df['MeanAmountByHour'] + 1e-6)
    
    return df

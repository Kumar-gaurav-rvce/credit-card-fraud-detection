# preprocessing.py
import joblib
import numpy as np
import pandas as pd

SCALER_PATH = "artifacts/scaler.pkl"
HOUR_STATS_PATH = "artifacts/hourly_stats.pkl"

def preprocess_inference(transaction):
    """
    transaction: dict with keys 'Amount' and 'Time'
    Returns: pd.DataFrame with scaled features (V1–V28 + Amount + Time)
    """
    # Load artifacts
    scaler = joblib.load(SCALER_PATH)
    hourly_means = joblib.load(HOUR_STATS_PATH)

    # Extract user input
    amount = float(transaction["Amount"])
    time = int(transaction["Time"])

    # Derive "hour of day" from time
    hour = (time // 3600) % 24

    # Get default V1–V28 for that hour
    v_features = hourly_means.get(hour, np.zeros(28))

    # Build feature vector
    features = [time] + list(v_features) + [amount]

    # Create DataFrame with correct columns
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df = pd.DataFrame([features], columns=cols)

    # Scale with pre-fitted scaler
    df_scaled = pd.DataFrame(scaler.transform(df), columns=cols)

    return df_scaled

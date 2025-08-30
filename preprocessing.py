import joblib
import numpy as np
import os
import boto3
import pandas as pd

BUCKET = "ml-rvce-us-east-1"
SCALER_KEY = "fraud-detection/models/scaler.pkl"
LOCAL_SCALER_PATH = "artifacts/scaler.pkl"

s3 = boto3.client("s3")

def get_scaler():
    os.makedirs("artifacts", exist_ok=True)
    if not os.path.exists(LOCAL_SCALER_PATH):
        s3.download_file(BUCKET, SCALER_KEY, LOCAL_SCALER_PATH)
    scaler = joblib.load(LOCAL_SCALER_PATH)
    return scaler

def preprocess_inference(transaction):
    """
    transaction: dict with keys 'Time' and 'Amount'
    Returns: numpy array with shape (1, 30) matching training features
    """
    # Fill missing features (V1-V28) with zeros or median values
    v_features = {f"V{i}": 0.0 for i in range(1, 29)}

    # Combine Time + Amount + V1-V28
    all_features = {"Time": transaction["Time"], "Amount": transaction["Amount"], **v_features}

    # Order features exactly as during training: V1–V28, Time, Amount
    feature_order = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    X = np.array([[all_features[f] for f in feature_order]])

    # Scale
    scaler = get_scaler()
    X_scaled = scaler.transform(X)
    return X_scaled

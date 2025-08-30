# preprocessing.py
import joblib
import numpy as np
import boto3
import os

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

def preprocess_inference(transaction: dict):
    # Convert dict -> array
    features = np.array([[transaction["Time"], transaction["Amount"]]])
    scaler = get_scaler()
    features_scaled = scaler.transform(features)
    return features_scaled

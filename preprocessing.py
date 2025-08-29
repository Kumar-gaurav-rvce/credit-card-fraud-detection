# preprocessing.py
import joblib
import boto3
import os
import numpy as np

BUCKET = "ml-rvce-us-east-1"
SCALER_KEY = "fraud-detection/models/scaler.pkl"
LOCAL_SCALER_PATH = "artifacts/scaler.pkl"

s3 = boto3.client("s3")

def get_scaler():
    # Always refresh scaler from S3
    os.makedirs("artifacts", exist_ok=True)
    s3.download_file(BUCKET, SCALER_KEY, LOCAL_SCALER_PATH)
    scaler = joblib.load(LOCAL_SCALER_PATH)
    print("Scaler loaded with n_features_in_ =", scaler.n_features_in_)
    return scaler

def preprocess_inference(transaction):
    # Convert dict → feature array
    features = np.array([[transaction["Time"], transaction["Amount"]]])
    scaler = get_scaler()
    features_scaled = scaler.transform(features)
    return features_scaled


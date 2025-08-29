# preprocessing.py
import numpy as np
import joblib

SCALER_PATH = "artifacts/scaler.pkl"

def preprocess_inference(transaction):
    # transaction = {"Amount": value, "Time": value}
    amount = transaction["Amount"]
    time = transaction["Time"]

    # load scaler
    scaler = joblib.load(SCALER_PATH)

    # build feature vector
    features = np.array([[time, amount]])

    # scale
    features_scaled = scaler.transform(features)

    return features_scaled

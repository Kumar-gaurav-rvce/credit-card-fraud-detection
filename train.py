import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import boto3
import os

from preprocessing import preprocess_training, SCALER_PATH, HOUR_STATS_PATH

BUCKET = "ml-rvce-us-east-1"
MODEL_PATH = "fraud-detection/models/fraud_model.pkl"
SCALER_S3_PATH = "fraud-detection/models/scaler.pkl"
HOUR_STATS_S3_PATH = "fraud-detection/models/hourly_stats.pkl"

def train_model():
    # Load raw data
    df = pd.read_csv("data/creditcard.csv")
    
    # Preprocess (this step saves scaler.pkl + hourly_stats.pkl locally)
    df = preprocess_training(df)
    
    # Features & Target
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train XGBoost model
    model = XGBClassifier(
        scale_pos_weight=10,  # imbalance handling
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save model locally
    joblib.dump(model, "fraud_model.pkl")
    print("Model saved as fraud_model.pkl")
    
    # Upload model + preprocessing artifacts to S3
    try:
        s3 = boto3.client("s3")
        
        s3.upload_file("fraud_model.pkl", BUCKET, MODEL_PATH)
        print(f"Model uploaded to s3://{BUCKET}/{MODEL_PATH}")
        
        if os.path.exists(SCALER_PATH):
            s3.upload_file(SCALER_PATH, BUCKET, SCALER_S3_PATH)
            print(f"Scaler uploaded to s3://{BUCKET}/{SCALER_S3_PATH}")
        
        if os.path.exists(HOUR_STATS_PATH):
            s3.upload_file(HOUR_STATS_PATH, BUCKET, HOUR_STATS_S3_PATH)
            print(f"Hourly stats uploaded to s3://{BUCKET}/{HOUR_STATS_S3_PATH}")
        
    except Exception as e:
        print("Skipping S3 upload:", e)

if __name__ == "__main__":
    train_model()

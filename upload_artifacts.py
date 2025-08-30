import boto3
import os

# Replace with your bucket name
BUCKET_NAME = "ml-rvce-us-east-1"
S3_PREFIX = "fraud-detection/models/"   # path inside bucket

# Local artifacts (after training)
ARTIFACTS = ["fraud_model.pkl", "scaler.pkl"]

def upload_to_s3():
    s3 = boto3.client("s3")

    for artifact in ARTIFACTS:
        if not os.path.exists(artifact):
            print(f"{artifact} not found locally, skipping...")
            continue

        s3_key = f"{S3_PREFIX}{artifact}"
        print(f"Uploading {artifact} → s3://{BUCKET_NAME}/{s3_key}")

        s3.upload_file(artifact, BUCKET_NAME, s3_key)
        print(f"Uploaded {artifact} to {s3_key}")

if __name__ == "__main__":
    upload_to_s3()

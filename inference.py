import joblib
from preprocessing import preprocess_inference

MODEL_PATH = "artifacts/model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)

def predict_transaction(transaction_input):
    """
    Predict fraud for given transaction(s).

    Accepts:
    - dict: single transaction
    - list of dicts: multiple transactions
    - pandas DataFrame: multiple transactions
    Returns:
    - preds: array of 0/1 predictions
    - probs: array of fraud probabilities
    """
    # Preprocess input
    X = preprocess_inference(transaction_input)

    # Predict
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    return preds, probs

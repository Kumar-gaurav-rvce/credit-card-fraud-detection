import joblib   # For saving/loading trained models and scalers
from preprocessing import preprocess_inference

# ----------------------------------------
# Load trained model
# ----------------------------------------
MODEL_PATH = "artifacts/model.pkl"
model = joblib.load(MODEL_PATH)

# ----------------------------------------
# Predict function
# ----------------------------------------
def predict_transaction(transaction_input):
    """
    Predict fraud for transaction(s)

    Accepts:
        - dict: single transaction
        - list of dicts: multiple transactions
        - pandas DataFrame: multiple transactions

    Returns:
        - preds: 0/1 predictions
        - probs: fraud probabilities
    """
    # Preprocess input
    X = preprocess_inference(transaction_input)

    # Make predictions
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    return preds, probs

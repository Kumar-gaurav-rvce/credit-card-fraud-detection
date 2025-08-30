import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

SCALER_PATH = "artifacts/scaler.pkl"

# ----------------------------------------
# Preprocessing for training
# ----------------------------------------
def preprocess_training(df):
    """
    Preprocess raw data for training.
    - Fill missing values with 0
    - Scale numeric features
    - Save the scaler for inference
    """
    # Extract feature columns (exclude target)
    feature_cols = [c for c in df.columns if c != 'Class']

    # Fill missing values
    df[feature_cols] = df[feature_cols].fillna(0)

    # Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Save the scaler
    joblib.dump(scaler, SCALER_PATH)
    print("Scaler saved to", SCALER_PATH)

    return df

# ----------------------------------------
# Preprocessing for inference
# ----------------------------------------
def preprocess_inference(transaction_input):
    """
    Preprocess transaction(s) for inference.
    Accepts:
        - dict (single transaction)
        - list of dicts (multiple transactions)
        - pandas DataFrame (multiple transactions)
    Returns:
        2D numpy array ready for model prediction
    """
    # Convert input to DataFrame
    if isinstance(transaction_input, dict):
        df = pd.DataFrame([transaction_input])  # single row
    elif isinstance(transaction_input, list):
        df = pd.DataFrame(transaction_input)    # multiple rows
    elif isinstance(transaction_input, pd.DataFrame):
        df = transaction_input.copy()          # already DataFrame
    else:
        raise ValueError("Input must be dict, list of dicts, or DataFrame")

    # Drop target column if present
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])

    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Align DataFrame columns with scaler’s feature names
    feature_cols = list(scaler.feature_names_in_)
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Convert to 2D numpy array
    X = df.to_numpy()
    return X

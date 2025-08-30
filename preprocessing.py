import pandas as pd
import joblib

SCALER_PATH = "artifacts/scaler.pkl"

def preprocess_training(df):
    """
    Preprocess raw dataframe for training:
    - Fill missing values
    - Scale numeric columns
    """
    feature_cols = [c for c in df.columns if c != 'Class']

    df[feature_cols] = df[feature_cols].fillna(0)

    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Save scaler for inference
    joblib.dump(scaler, SCALER_PATH)
    print("Scaler saved to", SCALER_PATH)

    return df

def preprocess_inference(transaction_input):
    """
    Preprocess transaction(s) for inference.
    Accepts:
    - dict: single transaction
    - list of dicts: multiple transactions
    - pandas DataFrame: multiple transactions
    Returns: 2D numpy array (n_samples, n_features)
    """
    # Convert input to DataFrame
    if isinstance(transaction_input, dict):
        df = pd.DataFrame([transaction_input])
    elif isinstance(transaction_input, list):
        df = pd.DataFrame(transaction_input)
    elif isinstance(transaction_input, pd.DataFrame):
        df = transaction_input.copy()
    else:
        raise ValueError("Input must be dict, list of dicts, or DataFrame")

    # Drop target column if present
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])

    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Ensure all feature columns exist
    feature_cols = getattr(scaler, "feature_names_in_", df.columns)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols].fillna(0)

    # Convert to 2D array
    X = df.to_numpy()
    return X

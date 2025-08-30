# train.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("creditcard.csv")

# Features & target
X = df[["Time", "Amount"]]  # simple model
y = df["Class"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Compute scale_pos_weight for imbalanced data
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# Train XGBoost
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Saved model.pkl and scaler.pkl (2 features, imbalance handled)")

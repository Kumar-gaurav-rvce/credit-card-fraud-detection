import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

# Legitimate transactions
amount_legit = np.random.normal(50, 30, size=n_samples)
time_legit = np.random.normal(30000, 10000, size=n_samples)
y_legit = np.zeros(n_samples)

# Fraudulent transactions (extreme values)
amount_fraud = np.random.normal(2000, 500, size=50)
time_fraud = np.random.normal(60000, 5000, size=50)
y_fraud = np.ones(50)

# Combine
X = pd.DataFrame({
    "Amount": np.concatenate([amount_legit, amount_fraud]),
    "Time": np.concatenate([time_legit, time_fraud])
})
y = np.concatenate([y_legit, y_fraud])

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=10)
model.fit(X_scaled, y)

# Save artifacts
import os
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")

print("Toy model and scaler saved!")

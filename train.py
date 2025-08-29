# train_simple.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# 1. Load data
df = pd.read_csv("creditcard.csv")  # original Kaggle dataset

# 2. Select only the features you want
X = df[["Time", "Amount"]]
y = df["Class"]

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train model
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=(len(y) - sum(y)) / sum(y),  # handle imbalance
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

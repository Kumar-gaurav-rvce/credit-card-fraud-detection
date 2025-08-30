import pandas as pd
from preprocessing import preprocess_training
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import joblib

# Load raw data
df = pd.read_csv("data/creditcard_raw.csv")

# Preprocess
df = preprocess_training(df)

# Features & target
X = df.drop(columns=['Class'])
y = df['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Handle imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train_res, y_train_res)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "artifacts/fraud_model.pkl")
print("Model and scaler saved in artifacts/")

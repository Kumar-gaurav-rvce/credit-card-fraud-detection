import pandas as pd
import joblib
from preprocessing import preprocess_training
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# ----------------------------------------
# Paths
# ----------------------------------------
RAW_CSV_PATH = "data/creditcard_raw.csv"
ARTIFACTS_DIR = "artifacts/"
MODEL_PATH = ARTIFACTS_DIR + "model.pkl"

# ----------------------------------------
# Load raw CSV data
# ----------------------------------------
df = pd.read_csv(RAW_CSV_PATH)
print("Raw data loaded:", df.shape)

# ----------------------------------------
# Preprocess the data
# ----------------------------------------
df_processed = preprocess_training(df)

# ----------------------------------------
# Split features and target
# ----------------------------------------
X = df_processed.drop(columns=["Class"])
y = df_processed["Class"]

# ----------------------------------------
# Train/Test split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train/Test split:", X_train.shape, X_test.shape)

# ----------------------------------------
# Handle class imbalance using SMOTE
# ----------------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE:", X_train_res.shape, y_train_res.shape)

# ----------------------------------------
# Train XGBoost classifier
# ----------------------------------------
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train_res, y_train_res)

# ----------------------------------------
# Evaluate the model
# ----------------------------------------
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------------
# Save trained model
# ----------------------------------------
joblib.dump(model, MODEL_PATH)
print("Model saved to:", MODEL_PATH)

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from pytorch_tabnet.tab_model import TabNetClassifier


# ================= 路徑設定 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "loan_test_9000.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "henry/fe_models_4f")

MODEL_PATH = os.path.join(MODEL_DIR, "fe_4f_model.zip")
META_PATH = os.path.join(MODEL_DIR, "features.json")
ENCODER_PATH = os.path.join(MODEL_DIR, "cat_encoders.pkl")

TARGET = "loan_status"


# ================= 讀資料 =================
df = pd.read_csv(DATA_PATH)


# ================= engineered features（與訓練完全一致） =================
df["log_income"] = np.log1p(df["person_income"])
df["log_loan_amnt"] = np.log1p(df["loan_amnt"])
df["interest_pressure"] = df["loan_int_rate"] * df["loan_percent_income"]
df["age_bucket"] = pd.cut(
    df["person_age"],
    bins=[0, 30, 100],
    labels=["young", "mid"]
)


# ================= 載入特徵結構 =================
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

FEATURES = meta["features"]
categorical_cols = meta["categorical_cols"]

X = df[FEATURES]
y = df[TARGET].values


# ================= 類別編碼（與訓練一致） =================
cat_encoders = joblib.load(ENCODER_PATH)

for col in categorical_cols:
    le = cat_encoders[col]
    X[col] = le.transform(X[col].astype(str))


# ================= 載入模型 =================
model = TabNetClassifier()
model.load_model(MODEL_PATH)


# ================= 預測 =================
y_pred = model.predict(X.values)
y_proba = model.predict_proba(X.values)[:, 1]


# ================= 評估 =================
print("\n=== Test Set Metrics ===")
print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall   :", recall_score(y, y_pred))
print("F1-score :", f1_score(y, y_pred))
print("ROC-AUC  :", roc_auc_score(y, y_proba))
print("PR-AUC   :", average_precision_score(y, y_proba))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y, y_pred, digits=4))

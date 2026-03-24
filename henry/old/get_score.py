import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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


# ==================================================
# 1. 路徑設定（完全對齊你現在的資料夾結構）
# ==================================================
# henry/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 專案根目錄 loan-approval-ml/
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "loan_train_36000.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "tabnet_loan_default.zip")

TARGET = "loan_status"


# ==================================================
# 2. 讀資料
# ==================================================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET].values


# ==================================================
# 3. Train / Validation split（與訓練一致）
# ==================================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X.values,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==================================================
# 4. 載入已訓練 TabNet 模型
# ==================================================
model = TabNetClassifier()
model.load_model(MODEL_PATH)


# ==================================================
# 5. 預測
# ==================================================
y_pred = model.predict(X_valid)
y_proba = model.predict_proba(X_valid)[:, 1]


# ==================================================
# 6. 評估指標
# ==================================================
acc = accuracy_score(y_valid, y_pred)
prec = precision_score(y_valid, y_pred)
rec = recall_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred)
roc_auc = roc_auc_score(y_valid, y_proba)
pr_auc = average_precision_score(y_valid, y_proba)

print("\n=== Validation Metrics ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_valid, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_valid, y_pred, digits=4))

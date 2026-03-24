import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

# =========================
# 路徑
# =========================
MODEL_PATH = "lgbm_best_model.zip"
FEATURE_PATH = "lgbm_best_model_features.json"
TEST_PATH = "loan_test_9000.csv"

TARGET = "loan_status"

# =========================
# 載入模型
# =========================
model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH) as f:
    features = json.load(f)

# =========================
# 讀取測試資料
# =========================
df = pd.read_csv(TEST_PATH)

# 建立 engineered features
df["log_income"] = np.log1p(df["person_income"])
df["interest_pressure"] = df["loan_int_rate"] * df["loan_percent_income"]

# 類別欄位轉型
categorical_cols = [
    "person_home_ownership",
    "loan_intent",
    "person_gender"
]

for col in categorical_cols:
    df[col] = df[col].astype("category")

# 取特徵與標籤
X_test = df[features]
y_test = df[TARGET]

# =========================
# 預測
# =========================
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# =========================
# 評估指標
# =========================
print("=== TEST SET EVALUATION ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("AUC      :", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

plt.figure(figsize=(6,6))
disp.plot(values_format='d')
plt.title("Confusion Matrix (Test Set)")
plt.show()

# =========================
# ROC Curve
# =========================
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.4f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend()
plt.show()
import shap
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# =========================
# 路徑
# =========================
MODEL_PATH = "lgbm_best_model.zip"
FEATURE_PATH = "lgbm_best_model_features.json"
DATA_PATH = "loan_train_36000.csv"   # 用 train 或 test 都可以

TARGET = "loan_status"

# =========================
# 載入模型
# =========================
model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH) as f:
    features = json.load(f)

# =========================
# 讀資料 + 重建特徵
# =========================
df = pd.read_csv(DATA_PATH)

df["log_income"] = np.log1p(df["person_income"])
df["interest_pressure"] = df["loan_int_rate"] * df["loan_percent_income"]

categorical_cols = [
    "person_home_ownership",
    "loan_intent",
    "person_gender"
]

for col in categorical_cols:
    df[col] = df[col].astype("category")

X = df[features]

# =========================
# 建立 SHAP Explainer
# =========================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 如果是二分類，取 class 1（違約）
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# =========================
# 1️⃣ SHAP Summary Plot（最重要）
# =========================
plt.figure()
shap.summary_plot(shap_values, X, show=True)

# =========================
# 2️⃣ SHAP Bar Plot（全局重要性）
# =========================
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=True)

# =========================
# 3️⃣ 單一樣本解釋
# =========================
sample_index = 0  # 你可以改成任何 index

plt.figure()
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_index],
    X.iloc[sample_index],
    matplotlib=True
)
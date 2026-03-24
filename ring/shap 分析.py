# !pip uninstall scikit-learn -y
# !pip install scikit-learn==1.8.0

import sklearn
import sys

# 確認版本，要跟zip檔一致
print("sklearn version:", sklearn.__version__)
print("python path:", sys.executable)

import shap
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# =========================
# 路徑
# =========================
MODEL_PATH = "/content/lgbm_best_model.zip"
FEATURE_PATH = "/content/lgbm_best_model_features.json"
DATA_PATH = "/content/loan_train_36000.csv"

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
# SHAP 計算（抽樣）
# =========================
X_sample = X.sample(800, random_state=42)

explainer = shap.TreeExplainer(model.booster_)
shap_values = explainer.shap_values(X_sample)

# 二分類取 class 1
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# 最終版
# ==========================================
# 1️⃣ Summary Plot（點圖）美化版
# ==========================================
plt.figure(figsize=(9,6))

shap.summary_plot(
    shap_values,
    X_sample,
    show=False,
    color_bar=True
)

ax = plt.gca()

ax.set_title("SHAP Summary Plot (Feature Impact on Log-Odds of Default)",
             fontsize=14,
             fontweight="bold",
             color="#2C3E50")

ax.set_xlabel("SHAP Value (Impact on Log-Odds)",
              fontsize=12,
              fontweight="bold")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("/content/shap_summary_dot_pretty2.png",
            dpi=300,
            bbox_inches="tight")
plt.close()


# ==========================================
# 2️⃣ Bar Plot（全局重要性）美化版
# ==========================================
plt.figure(figsize=(8,6))

shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False,
    color="#94B0A8"
)

ax = plt.gca()

ax.set_title("Global Feature Importance (Mean |SHAP Value| in Log-Odds Space)",
             fontsize=14,
             fontweight="bold",
             color="#2C3E50")

ax.set_xlabel("Mean |SHAP Value| (Log-Odds)",
              fontsize=12,
              fontweight="bold")

ax.tick_params(axis='y', labelsize=11)
ax.tick_params(axis='x', labelsize=11)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("/content/shap_summary_bar_pretty2.png",
            dpi=300,
            bbox_inches="tight")
plt.close()


# ==========================================
# 3️⃣ Waterfall Plot（單筆解釋）美化版
# ==========================================
sample_index = 0

exp = shap.Explanation(
    values=shap_values[sample_index],
    base_values=explainer.expected_value,
    data=X_sample.iloc[sample_index],
    feature_names=X_sample.columns
)

fig = plt.figure(figsize=(9,6))

shap.plots.waterfall(
    exp,
    show=False
)

ax = plt.gca()

ax.set_title("Local Explanation - Log-Odds Contribution to Default Risk",
             fontsize=14,
             fontweight="bold",
             color="#2C3E50")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig("/content/shap_waterfall_pretty2.png",
            dpi=300,
            bbox_inches="tight")
plt.close(fig)
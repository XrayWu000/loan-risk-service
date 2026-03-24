import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================
# 設定
# =========================
TRAIN_PATH = "loan_train_36000.csv"
TEST_PATH  = "loan_test_9000.csv"

MODEL_PATH = "lgbm_best_model.zip"
FEATURE_PATH = "lgbm_best_model_features.json"

TARGET = "loan_status"

FINAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_int_rate",
    "cb_person_cred_hist_length",
    "interest_pressure",
    "person_emp_exp",
    "person_age",
    "person_gender",
    "loan_amnt",
    "log_income",
]

CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "person_gender",
]

# =========================
# 讀 Train
# =========================
train_df = pd.read_csv(TRAIN_PATH)

train_df["log_income"] = np.log1p(train_df["person_income"])
train_df["interest_pressure"] = train_df["loan_int_rate"] * train_df["loan_percent_income"]

for col in CATEGORICAL_COLS:
    train_df[col] = train_df[col].astype("category")

X_train = train_df[FINAL_FEATURES]
y_train = train_df[TARGET]

# =========================
# 讀 Test → 切成 Val + Hold
# =========================
test_df = pd.read_csv(TEST_PATH)

test_df["log_income"] = np.log1p(test_df["person_income"])
test_df["interest_pressure"] = test_df["loan_int_rate"] * test_df["loan_percent_income"]

for col in CATEGORICAL_COLS:
    test_df[col] = test_df[col].astype("category")

X_test_full = test_df[FINAL_FEATURES]
y_test_full = test_df[TARGET]

X_val, X_hold, y_val, y_hold = train_test_split(
    X_test_full,
    y_test_full,
    test_size=0.75,
    stratify=y_test_full,
    random_state=42
)

# =========================
# 超參數
# =========================
best_params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "average_precision",
    "learning_rate": 0.05,

    "num_leaves": 31,
    "max_depth": 15,
    "min_child_samples": 40,

    "subsample": 0.6,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,

    "n_estimators": 20000,

    "class_weight": {0:1, 1:1.4},

    "random_state": 42,
    "n_jobs": -1,

    "lambda_l1": 12,
    "lambda_l2": 30,
}

# =========================
# 訓練
# =========================
print("Training model with early stopping...")

model = lgb.LGBMClassifier(**best_params)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="average_precision",
    categorical_feature=CATEGORICAL_COLS,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(100)
    ]
)

print("Best iteration:", model.best_iteration_)

# =========================
# 預測
# =========================
train_prob = model.predict_proba(X_train)[:, 1]
val_prob   = model.predict_proba(X_val)[:, 1]
hold_prob  = model.predict_proba(X_hold)[:, 1]

# 用 0.5 threshold
train_pred = (train_prob >= 0.5).astype(int)
val_pred   = (val_prob >= 0.5).astype(int)
hold_pred  = (hold_prob >= 0.5).astype(int)

# =========================
# 評估
# =========================
train_metrics = {
    "AUC": roc_auc_score(y_train, train_prob),
    "Recall": recall_score(y_train, train_pred),
    "Precision": precision_score(y_train, train_pred),
    "F1": f1_score(y_train, train_pred),
    "Accuracy": accuracy_score(y_train, train_pred),
}

val_metrics = {
    "AUC": roc_auc_score(y_val, val_prob),
    "Recall": recall_score(y_val, val_pred),
    "Precision": precision_score(y_val, val_pred),
    "F1": f1_score(y_val, val_pred),
    "Accuracy": accuracy_score(y_val, val_pred),
}

hold_metrics = {
    "AUC": roc_auc_score(y_hold, hold_prob),
    "Recall": recall_score(y_hold, hold_pred),
    "Precision": precision_score(y_hold, hold_pred),
    "F1": f1_score(y_hold, hold_pred),
    "Accuracy": accuracy_score(y_hold, hold_pred),
}

results_table = pd.DataFrame(
    [train_metrics, val_metrics, hold_metrics],
    index=["Train", "Val", "Hold"]
)

print("\n========== 模型完整評估 ==========")
print(results_table.round(6))
print("===================================")

# # =========================
# # 存模型
# # =========================
# joblib.dump(model, MODEL_PATH, compress=3)

# with open(FEATURE_PATH, "w") as f:
#     json.dump(FINAL_FEATURES, f)

# print("Model saved to:", MODEL_PATH)
# print("Feature list saved to:", FEATURE_PATH)
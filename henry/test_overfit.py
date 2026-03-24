import pandas as pd
import numpy as np
import lightgbm as lgb

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
DATA_PATH = "loan_final_label_encoded.csv"
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
# 讀資料 + 建特徵
# =========================
df = pd.read_csv(DATA_PATH)

df["log_income"] = np.log1p(df["person_income"])
df["interest_pressure"] = df["loan_int_rate"] * df["loan_percent_income"]

for col in CATEGORICAL_COLS:
    df[col] = df[col].astype("category")

X_all = df[FINAL_FEATURES]
y_all = df[TARGET]

# =========================
# 10 次實驗
# =========================
results = []

for i in range(10):

    print(f"\n========== Round {i+1} ==========")

    # Step 1: 45000 → 36000 Train + 9000 Hold
    X_train, X_hold, y_train, y_hold = train_test_split(
        X_all,
        y_all,
        test_size=9000,
        stratify=y_all,
        random_state=i
    )

    # Step 2: Hold → 4500 Val + 4500 Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold,
        y_hold,
        test_size=0.75,
        stratify=y_hold,
        random_state=i
    )

    model = lgb.LGBMClassifier(**best_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        categorical_feature=CATEGORICAL_COLS,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(100)  # 每100輪印一次
        ]
    )

    # =========================
    # 機率
    # =========================
    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob   = model.predict_proba(X_val)[:, 1]
    test_prob  = model.predict_proba(X_test)[:, 1]

    # =========================
    # 在 Val 找 threshold：Precision 約 0.78，且 Recall 最高
    # =========================
    thresholds = np.arange(0.05, 0.95, 0.001)

    target_precision = 0.78
    precision_tol = 0.01   # 先用 ±0.01（你可改 ±0.02 比較容易找到）
    min_pred_pos = 30      # 避免 threshold 太高導致預測正類太少、precision 虛高

    candidates = []

    for t in thresholds:
        val_pred_temp = (val_prob >= t).astype(int)

        pred_pos = val_pred_temp.sum()
        if pred_pos < min_pred_pos:
            continue

        p = precision_score(y_val, val_pred_temp, zero_division=0)
        r = recall_score(y_val, val_pred_temp, zero_division=0)

        # Precision 落在目標區間內
        if abs(p - target_precision) <= precision_tol:
            candidates.append((t, p, r, pred_pos))

    # 如果找不到落在 ±tol 的，就放寬 tol（避免完全沒有解）
    if not candidates:
        precision_tol2 = 0.03
        for t in thresholds:
            val_pred_temp = (val_prob >= t).astype(int)

            pred_pos = val_pred_temp.sum()
            if pred_pos < min_pred_pos:
                continue

            p = precision_score(y_val, val_pred_temp, zero_division=0)
            r = recall_score(y_val, val_pred_temp, zero_division=0)

            if abs(p - target_precision) <= precision_tol2:
                candidates.append((t, p, r, pred_pos))

        # 再找不到，就改成：Precision >= 0.78 的前提下 Recall 最大
        if not candidates:
            for t in thresholds:
                val_pred_temp = (val_prob >= t).astype(int)

                pred_pos = val_pred_temp.sum()
                if pred_pos < min_pred_pos:
                    continue

                p = precision_score(y_val, val_pred_temp, zero_division=0)
                r = recall_score(y_val, val_pred_temp, zero_division=0)

                if p >= target_precision:
                    candidates.append((t, p, r, pred_pos))

    if not candidates:
        # 最後保底：用 0.5
        best_threshold = 0.5
        best_p = precision_score(y_val, (val_prob >= best_threshold).astype(int), zero_division=0)
        best_r = recall_score(y_val, (val_prob >= best_threshold).astype(int), zero_division=0)
        print(f"[WARN] 找不到符合 Precision 條件的 threshold，使用 0.50 (P={best_p:.3f}, R={best_r:.3f})")
    else:
        # 規則：先最大化 Recall，其次讓 Precision 更接近 0.78
        candidates.sort(key=lambda x: (x[2], -abs(x[1] - target_precision)), reverse=True)
        best_threshold, best_p, best_r, best_pred_pos = candidates[0]
        print(f"Best threshold (Val): {best_threshold:.3f} | Precision={best_p:.3f} | Recall={best_r:.3f} | PredPos={best_pred_pos}")

    # =========================
    # 用最佳 threshold 產生最終預測
    # =========================
    train_pred = (train_prob >= best_threshold).astype(int)
    val_pred   = (val_prob   >= best_threshold).astype(int)
    test_pred  = (test_prob  >= best_threshold).astype(int)

    # =========================
    # 計算三組指標
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

    test_metrics = {
        "AUC": roc_auc_score(y_test, test_prob),
        "Recall": recall_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred),
        "F1": f1_score(y_test, test_pred),
        "Accuracy": accuracy_score(y_test, test_pred),
    }

    results_table = pd.DataFrame(
        [train_metrics, val_metrics, test_metrics],
        index=["Train", "Val", "Hold"]
    )

print("\n========== 模型完整評估 ==========")
print(results_table.round(6))
print("===================================")
import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

TARGET = "loan_status"
DATA_PATH = "loan_train_36000.csv"
LOG_PATH = "lgbm_ablation_log.csv"


# =============================
# 1️⃣ 加入 6 個 engineered features
# =============================
def add_engineered_features(df):
    df = df.copy()

    df["log_income"] = np.log1p(df["person_income"])
    df["log_loan_amnt"] = np.log1p(df["loan_amnt"])
    df["debt_pressure"] = df["loan_amnt"] / (df["person_income"] + 1)
    df["interest_pressure"] = df["loan_int_rate"] * df["loan_percent_income"]

    df["credit_score_bucket"] = pd.cut(
        df["credit_score"],
        bins=[0, 600, 700, 850],
        labels=["low", "mid", "high"]
    )
    df["credit_score_bucket"] = df["credit_score_bucket"].cat.codes

    df["age_bucket"] = pd.cut(
        df["person_age"],
        bins=[0, 30, 50, 100],
        labels=["young", "mid", "senior"]
    )
    df["age_bucket"] = df["age_bucket"].cat.codes

    return df


# =============================
# 2️⃣ 單次訓練
# =============================
def train_once(df, drop_col=None):

    start_time = datetime.now()

    work_df = df.copy()

    if drop_col is not None:
        work_df = work_df.drop(columns=[drop_col])

    X = work_df.drop(columns=[TARGET])
    y = work_df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100)],
    )

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    end_time = datetime.now()

    return {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": (end_time - start_time).total_seconds(),
        "dropped_feature": drop_col if drop_col else "None(baseline)",
        "num_features": X.shape[1],
        "accuracy": accuracy_score(y_val, y_pred),
        "auc": roc_auc_score(y_val, y_prob),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1_score": f1_score(y_val, y_pred, zero_division=0),
        "best_iteration": model.best_iteration_,
    }


# =============================
# 3️⃣ 主程式：跑 baseline + 18 次 drop
# =============================
def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # 加入 engineered features
    df = add_engineered_features(df)

    # 取得所有可訓練特徵
    all_features = [col for col in df.columns if col != TARGET]

    print(f"\nTotal features after engineering: {len(all_features)}")
    print("Feature list:")
    for f in all_features:
        print(" -", f)

    results = []

    # 1️⃣ Baseline
    print("\n=== Running Baseline ===")
    baseline_result = train_once(df, drop_col=None)
    results.append(baseline_result)
    print(f"Baseline AUC: {baseline_result['auc']:.4f}")

    # 2️⃣ Drop each feature
    print("\n=== Running Feature Ablation ===")
    for feature in all_features:
        print(f"\nDropping: {feature}")
        result = train_once(df, drop_col=feature)
        results.append(result)
        print(f"AUC after dropping {feature}: {result['auc']:.4f}")

    # 存成 CSV
    result_df = pd.DataFrame(results)

    if os.path.exists(LOG_PATH):
        result_df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        result_df.to_csv(LOG_PATH, index=False)

    print(f"\nAll experiments logged to {LOG_PATH}")


if __name__ == "__main__":
    main()
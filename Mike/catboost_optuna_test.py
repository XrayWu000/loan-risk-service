import pandas as pd
import numpy as np
import optuna
import os
import torch  # 用來確認 GPU
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ================= 1. 環境檢查 =================
print(f"是否偵測到 GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"使用顯示卡: {torch.cuda.get_device_name(0)}")

# ================= 2. 資料讀取 =================
DATA_PATH = r"D:\團隊專題\loan_catboost_raw_36000.csv"
df = pd.read_csv(DATA_PATH)
TARGET = "loan_status"

# 填補缺失值 (維持你原本的邏輯)
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())


# ================= 3. 定義實驗邏輯 =================
def objective(trial):
    # --- A. 特徵實驗：隨機決定要丟掉哪些欄位 (模仿你朋友的邏輯) ---
    features_to_drop = []
    if trial.suggest_categorical("drop_interest", [True, False]):
        features_to_drop.append("loan_int_rate")
    if trial.suggest_categorical("drop_income", [True, False]):
        features_to_drop.append("person_income")

    current_df = df.drop(columns=features_to_drop)

    # 重新辨識剩餘的類別欄位
    current_cat_features = current_df.drop(columns=[TARGET]).select_dtypes(include=['object']).columns.tolist()

    # 資料切分
    X = current_df.drop(columns=[TARGET])
    y = current_df[TARGET]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=37, stratify=y
    )

    # --- B. 參數搜尋範圍 ---
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 5.0),
        "task_type": "GPU",
        "devices": "0",
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        "verbose": False,
        "allow_writing_files": False
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, cat_features=current_cat_features, eval_set=(X_valid, y_valid))

    y_proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_proba)

    return auc


# ================= 4. 執行試跑 =================
print("開始 10 分鐘試跑 (預計 50 次實驗)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # 先跑 50 次確認情況

# ================= 5. 儲存結果 =================
print("\n[試跑完成]")
print(f"最佳 AUC: {study.best_value:.4f}")
print(f"最佳參數: {study.best_params}")

# 存成 CSV 方便檢查
df_results = study.trials_dataframe()
df_results.to_csv("optuna_test_run.csv", index=False)
print("結果已存至 optuna_test_run.csv")
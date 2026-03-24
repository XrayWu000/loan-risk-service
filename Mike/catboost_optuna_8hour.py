import pandas as pd
import optuna
import torch
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ================= 1. 資料讀取 =================
DATA_PATH = r"D:\團隊專題\loan_catboost_raw_36000.csv"
df = pd.read_csv(DATA_PATH)
TARGET = "loan_status"

# 移除填補缺失值的代碼，直接進入切分
X = df.drop(columns=[TARGET])
y = df[TARGET]
cat_features = X.select_dtypes(include=['object']).columns.tolist()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=37, stratify=y)


# ================= 2. 定義 Optuna 目標函數 =================
def objective(trial):
    params = {
        "iterations": 1200,  # 縮減上限，增加實驗次數
        "early_stopping_rounds": 100,  # 稍微放寬，讓模型學得更穩
        "depth": trial.suggest_int("depth", 7, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.8, 3.5),
        "task_type": "GPU",
        "devices": "0",
        "eval_metric": "Logloss",
        "verbose": False,
        "allow_writing_files": False
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_valid, y_valid))

    y_proba = model.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, y_proba)


# ================= 3. 啟動過夜搜尋 =================
if __name__ == "__main__":
    # 先確認 GPU 狀態
    print(f"CUDA 偵測狀態: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"正在使用: {torch.cuda.get_device_name(0)}")

    study = optuna.create_study(direction="maximize")

    print("\n🚀 開始 8 小時過夜搜尋 (預計明早完成)...")
    try:
        # timeout=28800 秒 (8小時)
        study.optimize(objective, timeout=28800)
    except KeyboardInterrupt:
        print("\n偵測到手動停止，存檔中...")

    # 4. 輸出與存檔
    print(f"\n[搜尋結束] 最佳 AUC: {study.best_value:.4f}")
    study.trials_dataframe().to_csv("optuna_overnight_results.csv", index=False)

    # 使用最佳參數訓練最終模型 (用 100% 的資料)
    print("正在訓練最終模型...")
    final_params = {**study.best_params, "iterations": 3000, "task_type": "GPU", "verbose": 100}
    final_model = CatBoostClassifier(**final_params)
    final_model.fit(X, y, cat_features=cat_features)
    final_model.save_model("best_loan_model_final.cbm")
    print("✨ 任務完成！結果已儲存。")
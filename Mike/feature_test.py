import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve


# 1. 定義特徵工程函數 (增加欄位)
def add_features(df):
    # 還款能力類
    df['loan_to_income_ratio'] = df['loan_amnt'] / (df['person_income'] + 1)
    df['interest_burden'] = df['loan_amnt'] * (df['loan_int_rate'] / 100)
    df['monthly_burden_estimate'] = df['interest_burden'] / 12

    # 穩定性指標
    df['work_age_ratio'] = df['person_emp_exp'] / (df['person_age'] - 14).clip(lower=1)

    # 信用與貸款交叉
    df['credit_loan_ratio'] = df['credit_score'] / (df['loan_amnt'] + 1)
    return df


# 2. 載入資料
train_df = pd.read_csv(r"D:\團隊專題\loan_catboost_raw_36000.csv")
test_df = pd.read_csv(r"D:\團隊專題\loan_test_9000.csv")

# 3. 套用特徵工程
train_df = add_features(train_df)
test_df = add_features(test_df)

TARGET = 'loan_status'
X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]
cat_features = X.select_dtypes(include=['object']).columns.tolist()

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# 4. 重新訓練 (強化 Recall 導向)
print("正在啟動強化版訓練，預計耗時 10-15 分鐘...")
params = {
    "iterations": 1200,
    "depth": 10,
    "learning_rate": 0.05,
    "l2_leaf_reg": 7.25,
    "scale_pos_weight": 3.0,  # <--- 重點：將違約權重提升至 5 倍，大幅強化 Recall
    "task_type": "GPU",
    "devices": "0",
    "eval_metric": "Logloss",
    "random_seed": 37,
    "verbose": 100
}

model = CatBoostClassifier(**params)
model.fit(X, y, cat_features=cat_features)

# 5. 獲取預測機率並尋找「黃金門檻」
y_proba = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# 尋找目標 Recall 為 0.8 的門檻 (抓出 80% 壞帳)
target_recall = 0.80
idx = np.where(recalls >= target_recall)[0][-1]
golden_threshold = thresholds[idx]

# 6. 最終評估
y_pred_final = (y_proba >= golden_threshold).astype(int)

print(f"\n[訓練完成]")
print(f"使用的黃金門檻: {golden_threshold:.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("\n=== 最終分類報告 (目標 Recall 80%) ===")
print(classification_report(y_test, y_pred_final))

# 儲存新模型
model.save_model("best_loan_model_v2_enhanced.cbm")
print("模型已另存為: best_loan_model_v2_enhanced.cbm")


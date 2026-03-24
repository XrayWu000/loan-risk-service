import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report

# 1. 讀取原始資料
df = pd.read_csv(r"D:\團隊專題\loan_catboost_raw_36000.csv")

# 2. 自動辨識類別型欄位
cat_features = df.select_dtypes(include=['object']).columns.tolist()

# 3. 處理缺失值
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

# 4. 資料切分
TARGET = "loan_status"
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=37, stratify=y
)

# ========== 優化重點：計算類別權重 ==========
# 計算方式：負樣本數量 / 正樣本數量 (7000 / 2000 = 3.5)
num_negative = len(y_train[y_train == 0])
num_positive = len(y_train[y_train == 1])
calc_weight = num_negative / num_positive
print(f"自動計算的類別權重 (scale_pos_weight): {calc_weight:.2f}")

# 5. 模型建立 (加入優化參數)
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,        # 稍微調低學習率，讓模型學得更穩
    depth=4,
    eval_metric='AUC',
    random_seed=42,
    scale_pos_weight=calc_weight, # <--- 這是優化核心
    verbose=100,
    early_stopping_rounds=50,
    use_best_model=True
)

# 6. 開始訓練
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_valid, y_valid)
)

# 7. 驗證與存檔
y_proba = model.predict_proba(X_valid)[:, 1]
print(f"\n優化後 Validation ROC-AUC: {roc_auc_score(y_valid, y_proba):.4f}")

model.save_model("catboost_loan_model_optimized_final.cbm")
print("優化版模型已儲存。")
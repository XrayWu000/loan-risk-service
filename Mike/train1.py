import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, classification_report

# 1. 讀取原始資料
df = pd.read_csv(r"D:\團隊專題\loan_catboost_raw_36000.csv")

# 2. 自動辨識類別型欄位 (Object 型態)
cat_features = df.select_dtypes(include=['object']).columns.tolist()
print(f"偵測到的類別型欄位: {cat_features}")

# 3. 處理缺失值
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

# 資料切分
TARGET = "loan_status"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 切分比例 8:2
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=37, stratify=y
)

# 模型建立與訓練
# 建立模型
model = CatBoostClassifier(
    iterations=1000,           # 最大迭代次數
    learning_rate=0.05,        # 學習率
    depth=6,                   # 樹的深度
    eval_metric='AUC',         # 監控指標
    random_seed=42,
    verbose=100,               # 每 100 輪印一次 log
    early_stopping_rounds=50,  # 如果 50 輪內沒進步就停止
    use_best_model=True        # 最終保留效果最好的那一輪
)

# 開始訓練
model.fit(
    X_train, y_train,
    cat_features=cat_features, # 這是 CatBoost 的靈魂：直接餵文字欄位
    eval_set=(X_valid, y_valid),
    plot=False
)

# 模型評估與可解釋性分析
# 1. 預測與計算 AUC
y_proba = model.predict_proba(X_valid)[:, 1]
auc_score = roc_auc_score(y_valid, y_proba)
print(f"\nCatBoost Validation ROC-AUC: {auc_score:.4f}")

# 2. 特徵重要性 (Feature Importance)
importance = model.get_feature_importance()
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
print("\n前五大關鍵影響因素:")
print(feature_importance_df.sort_values(by='importance', ascending=False).head(5))

model.save_model("catboost_loan_model.cbm")
print("CatBoost 模型已儲存。")
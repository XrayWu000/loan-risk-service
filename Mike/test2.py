import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report

# 1. 載入模型
model = CatBoostClassifier()
model.load_model(r"C:\Users\蔡秉翰\Desktop\loan-approval-ml\Mike\catboost_loan_model_optimized.cbm")

# 2. 讀取測試集 (假設檔名是 loan_test_9000.csv)
# 注意：這份資料必須跟原始資料格式一模一樣，且包含標籤(loan_status)
test_df = pd.read_csv(r"D:\團隊專題\loan_test_9000.csv")

X_test = test_df.drop(columns=["loan_status"])
y_test = test_df["loan_status"]

# 3. 進行預測
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# 4. 計算測試集 AUC
test_auc = roc_auc_score(y_test, y_proba)
print(f"測試集 (9000筆) ROC-AUC: {test_auc:.4f}")

# 5. 輸出詳細報告 (精確度、召回率)
print("\n詳細分類報告：")
print(classification_report(y_test, y_pred))
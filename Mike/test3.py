import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

# 1. 載入模型
model = CatBoostClassifier()
model.load_model(r"C:\Users\蔡秉翰\Desktop\loan-approval-ml\Mike\catboost_loan_model_optimized_final.cbm")

# 2. 讀取測試集
test_df = pd.read_csv(r"D:\團隊專題\loan_test_9000.csv")
X_test = test_df.drop(columns=["loan_status"])
y_test = test_df["loan_status"]

# 3. 進行預測
# 取得機率用於計算 AUC
y_proba = model.predict_proba(X_test)[:, 1]

# 根據你的圖片顯示，該結果似乎是使用「預設門檻 0.5」跑出來的
# (如果你要對應自定義門檻，請將 0.5 改為 custom_threshold)
y_pred = (y_proba > 0.3).astype(int)

# 4. 計算各項指標
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

# 5. 按照圖片格式輸出結果
print("=== Test Set Metrics ===")
print(f"Accuracy  : {accuracy}")
print(f"F1-score  : {f1}")
print(f"ROC-AUC   : {roc_auc}")
print()

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))
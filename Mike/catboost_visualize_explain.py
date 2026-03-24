import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# ========== 1. 準備資料 (為了拿驗證集來評估) ==========
DATA_PATH = r"D:\團隊專題\loan_catboost_raw_36000.csv"
TARGET = "loan_status"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 保持跟當初訓練時一樣的切分方式
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=37, stratify=y
)

# ========== 2. 讀取原本存好的模型 ==========
model = CatBoostClassifier()
model.load_model("catboost_loan_model.cbm") # 確保檔名正確
print("模型載入成功！")

# ========== 3. 分析與呈現階段 ==========

# --- (A) 效能評估 (AUC) ---
y_proba = model.predict_proba(X_valid)[:, 1]
auc_score = roc_auc_score(y_valid, y_proba)
print(f"\n模型的驗證集 AUC 為: {auc_score:.4f}")

# --- (B) 視覺化：特徵重要性 (Feature Importance) ---
# 這是報告最重要的部分：解釋模型在看什麼
feature_importance = model.get_feature_importance()
fi_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette='magma')
plt.title('Why does the model flag a loan? (Feature Importance)', fontsize=15)
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- (C) 視覺化：混淆矩陣 (Confusion Matrix) ---
# 看看模型誤判的情況
y_pred = model.predict(X_valid)
cm = confusion_matrix(y_valid, y_pred)

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe', 'Default'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix: Prediction vs Reality')
plt.show()

# --- (D) 實際預測 Demo ---
# 模擬一個新客戶，看看模型的反應
print("\n[測試] 模擬新客戶申請預測：")
sample_customer = X_valid.iloc[:1] # 拿驗證集第一筆當範例
pred_prob = model.predict_proba(sample_customer)[0][1]
actual_status = "違約" if y_valid.iloc[0] == 1 else "正常"

print(f"客戶特徵摘要: 收入 {sample_customer['person_income'].values[0]}, 貸款金額 {sample_customer['loan_amnt'].values[0]}")
print(f"模型預測違約機率: {pred_prob:.2%}")
print(f"實際該客戶狀態: {actual_status}")
import pandas as pd
import numpy as np
data = pd.read_csv('/content/drive/MyDrive/XGBoost/loan_test_9000.csv')   # <- 這邊要改成實際的檔案
data.head()

# 2. 定義需要進行 Log 轉換的數值欄位
# 收入與貸款金額通常有很長的尾巴（少數人極高），最適合轉換
log_cols = ['person_income', 'loan_amnt']

# 3. 執行 Log 轉換
# 使用 np.log1p 可以確保數值穩定性
for col in log_cols:
    data[col + '_log'] = np.log1p(data[col])

# 4. 查看轉換後的結果
print(data[['person_income', 'person_income_log', 'loan_amnt', 'loan_amnt_log']].head())

#列出欄位名稱
data.columns

X = data[['person_age', 'person_gender', 'person_education',
      'person_income_log', 'person_emp_exp',
       'person_home_ownership', 'loan_amnt_log',
       'loan_intent', 'loan_int_rate',
       'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']]
# ,'credit_value_ratio','experience_burden_ratio'

y = data['loan_status']


##
import pickle
import zipfile
import os

zip_filename = ['替換成模型檔']
# 修改為你雲端硬碟中的實際完整路徑
file_path = '/content/drive/MyDrive/XGBoost'
target_full_path = os.path.join(file_path, f"{zip_filename}")


if os.path.exists(target_full_path):
    with zipfile.ZipFile(target_full_path, "r") as zf:
        with zf.open("model.bin") as f:
            loaded_model = pickle.load(f)
    print("✅ 已從雲端硬碟成功載入模型！")
else:
    print("❌ 找不到檔案，請確認路徑是否正確。")

import plotly.express as px
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,average_precision_score
)

# 取得分類預測
y_pred = model.predict(X)



print("測試組 Accuracy:", accuracy_score(y, y_pred))
print("測試組 Balanced Accuracy:", balanced_accuracy_score(y, y_pred))
print("測試組 F1:", f1_score(y, y_pred, average='macro'))
print("測試組 Precision:", precision_score(y, y_pred, average='macro'))
print("測試組 Recall:", recall_score(y, y_pred, average='macro'))

# roc_auc_score 必須用機率值
try:
    # 加上 [:, 1] 取出正類別 (類別 1) 的預測機率，變成 1D 陣列
    y_prob = model.predict_proba(X)[:, 1]

    # 二元分類直接計算即可，不需要 multi_class 與 average 參數
    print("測試組 ROC AUC:", roc_auc_score(y, y_prob))

except AttributeError:
    print("模型沒有 predict_proba，無法計算 ROC AUC")

auc_pr = average_precision_score(y, y_prob)
print(f"測試組 AUC-PR Score: {auc_pr:.4f}")


# 計算混淆矩陣
cm = confusion_matrix(y, y_pred)
labels = model.classes_  # pipeline 最後分類器的類別

# 轉成 DataFrame
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# 用 Plotly 畫熱力圖
fig = px.imshow(
    cm_df,
    text_auto=True,       # 顯示數字
    color_continuous_scale="Blues",
    labels=dict(x="Predicted Label", y="True Label", color="Count"),
)

fig.update_layout(
    title="Confusion Matrix",
    xaxis_title="Predicted Label",
    yaxis_title="True Label"
)

fig.show()

from sklearn.metrics import classification_report, confusion_matrix

print("\n=============================================")
print("\nFinal Test Report (9,000 Samples)")
print("\n=============================================")
print("\n=== Test Set Metrics ===")
print("Accuracy:", accuracy_score(y, y_pred))
print("F1:", f1_score(y, y_pred, average='macro'))
print("ROC-AUC:", roc_auc_score(y_true, y_prob_positive))

y_pred = model.predict(X)
print("\n=== classification_report ===")
print(classification_report(y, y_pred))
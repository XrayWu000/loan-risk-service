# 1. 確保掛載 (如果已掛載，這行執行很快)
from google.colab import drive
drive.mount('/content/drive')

# 2. 定義雲端硬碟路徑 (確保路徑對應到你的雲端硬碟)
import os
save_path = '/content/drive/MyDrive/XGBoost' # 建議建個資料夾
file_path = os.path.join(save_path, 'all_qualified_models(LGBM).csv')

# 3. 確保資料夾存在 (避免路徑報錯)
if not os.path.exists(save_path):
    os.makedirs(save_path)

import matplotlib

# matplotlib.use('Agg')
# %matplotlib inline
# <-- 加上這行 (如果是用 Colab/Jupyter 的話)

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib



# 1. 載入資料
file_path = '/content/drive/MyDrive/XGBoost/loan_train_36000.csv'
print(f"正在載入資料: {file_path} ...")

try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"錯誤: 找不到檔案 {file_path}。請確認檔案是否在同一資料夾內。")


# 2. 準備特徵與目標

# 2. 定義需要進行 Log 轉換的數值欄位
# 收入與貸款金額通常有很長的尾巴（少數人極高），最適合轉換
log_cols = ['person_income', 'loan_amnt']

# 3. 執行 Log 轉換
# 使用 np.log1p 可以確保數值穩定性
for col in log_cols:
    data[col + '_log'] = np.log1p(data[col])

# 4. 查看轉換後的結果
print(data[['person_income', 'person_income_log', 'loan_amnt', 'loan_amnt_log']].head())
#「信用價值比」 (Credit Value Ratio)
data['credit_value_ratio'] = data['loan_int_rate'] / data['credit_score']
print(data[['credit_value_ratio']].head())

#「年資負擔比」 (Experience Burden Ratio)
data['experience_burden_ratio'] = data['loan_percent_income'] / (data['person_emp_exp']+1)
print(data[['experience_burden_ratio']].head())


X = data.drop(columns=['loan_status'])
y = data['loan_status']

# 3. 切分資料
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 建立模型
print("開始訓練 LightGBM 模型... (請稍候)")
model = lgb.LGBMClassifier(

    n_estimators =2000,              # 配合早停(Early Stopping)通常設大
    learning_rate = 0.05,       # 較小的學習率能學得更細
    num_leaves = 15,              # Leaf-wise 的核心，建議小於 2^(max_depth)
    max_depth = 4,             # 限制深度防止過擬合，-1 為不限制

    # 處理不平衡 (妳特別要求的)
    is_unbalance = True,              # 自動權重平衡，適合 22% vs 78%
    # 或者可以使用 'model__scale_pos_weight': [3.5], (兩者選其一)

    # 針對過擬合與邊界值的優化：
    min_child_samples = 300,  # 相當於 min_data_in_leaf，越大模型越保守
    reg_alpha = 1,            # L1 正則化 (reg_alpha)
    reg_lambda = 1,            # L2 正則化 (reg_lambda)
    # 隨機採樣，增加模型穩定性
    # feature_fraction = 0.7,      # 每次訓練隨機選取 70%-80% 特徵
    # subsample = 0.8,
    # colsample_bytree = 0.8,
    #'bagging_fraction': [0.7, 0.8],      # 每次訓練隨機選取 70%-80% 資料
    n_jobs = -1
)

# 找出妳要標註為類別的欄位在 X 中的索引
cat_cols = ['person_gender', 'person_home_ownership', 'loan_intent']
cat_indices = [X.columns.get_loc(col) for col in cat_cols]

# 看看 cat_indices 是不是一串數字，例如 [0, 2, 5, 6, 8]
print(f"類別欄位的索引位置: {cat_indices}")

# 在 RandomizedSearchCV 的 fit 中傳入參數
# 注意：因為有 Pipeline，參數名稱要加上 "model__" 前綴
fit_params = {
    'categorical_feature': cat_indices,  # 告訴模型哪些是類別欄位
    'eval_set':[(X_train, y_train), (X_val, y_val)], # 告訴模型：看這組資料來決定要不要停止
    'eval_names':['train', 'valid'],
    'callbacks': [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ],
    'eval_metric': ['binary_logloss', 'average_precision']
}

# 5. 訓練
model.fit(X_train, y_train, **fit_params)


# 6. 評估與繪圖
print("\n--- 模型評估結果 ---")
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

# # 基本指標
print(f"準確率 (Accuracy): {accuracy_score(y_val, y_pred):.4f}")
print(f"AUC Score       : {roc_auc_score(y_val, y_prob):.4f}")
from sklearn.metrics import f1_score
print(f"F1-score (類別1) : {f1_score(y_val, y_pred):.4f}")
# 確保你有匯入這個函數
from sklearn.metrics import average_precision_score
print(f"AUC-PR Score    : {average_precision_score(y_val, y_prob):.4f}")

# ==========================================
# 7. 提取數據並繪製學習曲線 (含參數顯示)
# ==========================================
import matplotlib.pyplot as plt

# 1. 提取評估結果
evals_result = model.evals_result_
train_auc_pr = evals_result['train']['average_precision']
valid_auc_pr = evals_result['valid']['average_precision']

# 2. 繪製圖表 (把寬度稍微調寬一點，為了放右邊的文字方塊)
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(train_auc_pr, label='Train AUC-PR', color='#1f77b4', linewidth=2)
ax.plot(valid_auc_pr, label='Valid AUC-PR', color='#ff7f0e', linewidth=2)

# 標註 Early Stopping 停止的位置
best_iter = model.best_iteration_
ax.axvline(x=best_iter, color='green', linestyle='--', alpha=0.6, label=f'Best Iteration ({best_iter})')

ax.set_title('Learning Curve - LightGBM AUC-PR', fontsize=14)
ax.set_xlabel('Number of Iterations (Trees)', fontsize=12)
ax.set_ylabel('AUC-PR Score', fontsize=12)
ax.legend(loc='lower right') # 把圖例放在右下角比較不會擋到線
ax.grid(True, linestyle=':', alpha=0.7)

# ==========================================
# 新增：在圖片右側加上參數文字方塊
# ==========================================
# 自動獲取模型參數
params = model.get_params()

# 挑選你想顯示的核心參數 (如果全部印出來會太長)
key_params = [
    'n_estimators', 'learning_rate', 'num_leaves', 'max_depth',
    'is_unbalance', 'min_child_samples', 'reg_alpha', 'reg_lambda'
]

# 將參數組裝成多行字串
param_text = "Model Parameters:\n" + "-"*20 + "\n"
for k in key_params:
    param_text += f"{k}: {params.get(k)}\n"

# 在圖上加上文字方塊
# transform=ax.transAxes 讓座標以圖表比例 (0~1) 計算，x=1.02 代表在圖表右邊緣再往外一點
ax.text(1.02, 0.5, param_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.9))

# ==========================================

# 調整排版，確保右側的文字方塊不會被切掉
plt.tight_layout()
import datetime

# 1. 取得現在的時間，並格式化成 YYYYMMDD_HHMMSS 的字串 (例如：20260220_143000)
current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")

# 2. 將時間加進檔名中
file_path = f'/content/drive/MyDrive/XGBoost/learning_curve_{current_time}.png'

# 3. 存檔 (加上 bbox_inches='tight' 可以避免右側文字方塊被切掉)
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"圖表已存檔至：{file_path}")
plt.show()

# 輸出結果
best_score = model.best_score_['valid']['average_precision']
print(f"\n訓練完成！最佳 AUC-PR (驗證集): {best_score:.4f}")
print(f"模型實際在第 {best_iter} 棵樹停止。")
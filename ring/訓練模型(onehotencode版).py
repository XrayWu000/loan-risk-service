# 1. 確保掛載 (如果已掛載，這行執行很快)
from google.colab import drive
drive.mount('/content/drive')

# 2. 定義雲端硬碟路徑 (確保路徑對應到你的雲端硬碟)
import os
save_path = '/content/drive/MyDrive/XGBoost' # 建議建個資料夾
file_path = os.path.join(save_path, 'all_qualified_models.csv')

# 3. 確保資料夾存在 (避免路徑報錯)
if not os.path.exists(save_path):
    os.makedirs(save_path)

#載入訓練用資料
import pandas as pd
import numpy as np
data = pd.read_csv('/content/drive/MyDrive/XGBoost/loan_train_36000.csv')
data.head()

#列出欄位名稱
data.columns

import sys
import os

# 定義你的 .py 檔所在的資料夾路徑
folder_path = '/content/drive/MyDrive/XGBoost'

# 如果這個路徑不在 Python 的搜尋清單中，就把它加進去
if folder_path not in sys.path:
    sys.path.append(folder_path)

import AutoPreprocessV1
# 在這個檔案有更動內容時使用
# import importlib
# importlib.reload(AutoPreprocess) # 強制更新最新的程式碼

ap = AutoPreprocessV1.AutoPreprocess()

ap.fit(data, field_names=['person_age', 'person_gender', 'person_education', 'person_income',
       'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
       'credit_score','loan_status'])

# # 轉換 panddas dataframe
data = ap.transform(data)
data.columns

#選取使用欄位
X = data[['person_age', 'person_gender', 'person_education_2',
       'person_education_0', 'person_education_1', 'person_education_3',
       'person_education_4', 'person_income', 'person_emp_exp',
       'person_home_ownership_0', 'person_home_ownership_1',
       'person_home_ownership_2', 'person_home_ownership_3', 'loan_amnt',
       'loan_intent_3', 'loan_intent_1', 'loan_intent_2', 'loan_intent_0',
       'loan_intent_4', 'loan_intent_5', 'loan_int_rate',
       'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']]
y = data['loan_status']

# RandomizedSearchCV版訓練參數設定
from xgboost import XGBClassifier
param_grid = [
    {
        'n_estimators': [600],  # 固定大一點，配合早停
        'learning_rate': [0.04],        # 降低學習率
        'max_depth': [4],         # 維持淺度
        'scale_pos_weight': [3.5],    # 處理 22% vs 78% 的不平衡
        # 針對過擬合與邊界值的優化：
        'min_child_weight': [5],       # 之前選 3 (邊界)，這次往上測(加這個可以有效控制過擬合)
        'reg_alpha': [0],            # 加入 L1 正則化，過濾雜訊
        'reg_lambda': [1]            # 加入 L2 正則化，讓權重更平滑

        # 加入隨機採樣提升泛化能力：
        #'subsample': [0.8, 0.9],             # 每次只用 80%~90% 的資料訓練
        #'colsample_bytree': [0.8, 0.9],      # 每次只用 80%~90% 的特徵
        #'subsample': [0.8],         # 增加隨機性
        #'colsample_bytree': [0.8],  # 隨機選取特徵
        #'reg_alpha': [1,5],        # L1 正規化
        #'reg_lambda': [1]            # L2 正規化
    }
]
model = XGBClassifier(random_state=42)

# RandomizedSearchCV版訓練  設定Pipeline
from sklearn.pipeline import Pipeline
import AutoPreprocessV1
new_param_grid = []
for pg in param_grid:
    new_param_grid.append({f"model__{k}": v for k, v in pg.items()})

pipe = Pipeline([('AutoPreprocess', AutoPreprocessV1.AutoPreprocess()), ('model', model)])
pipe

# RandomizedSearchCV版訓練  #跑模型
from sklearn.model_selection import RandomizedSearchCV

# 定義適合分類任務的評估指標
# average_precision 對應的就是 AUC-PR
scoring = {
    'recall': 'recall', 'precision_macro':'precision_macro',
    'f1': 'f1',
    'auc_pr': 'average_precision','accuracy':'accuracy'
}
rcv = RandomizedSearchCV(
    estimator=pipe,                  # 你要調參的模型
    param_distributions=new_param_grid,      # 超參數搜尋空間（隨機選取參數組合的候選範圍）
    return_train_score=True,              # 是否返回訓練集的分數，方便比較過擬合情況
    scoring=scoring,                      # 評估指標，支援列表或字串（例：['r2', 'neg_mean_absolute_error']）
    refit='auc_pr',                         # 用哪個評估指標做最終模型擬合（最佳參數選擇依據）
    n_iter=1,                          # 隨機挑選的超參數組合數（總共嘗試多少組參數）
    cv=5,                              # 交叉驗證摺數，預設為5（可以自訂）
    verbose=2,
    n_jobs=-1
)
rcv.fit(X, y)

# RandomizedSearchCV版訓練  計算平均分數
print("平均分數 (train vs test)：")
for score_name in scoring:
    train_key = f'mean_train_{score_name}'
    test_key = f'mean_test_{score_name}'
    train_score = rcv.cv_results_[train_key].mean()
    test_score = rcv.cv_results_[test_key].mean()
    print(f"{score_name}: Train={train_score:.4f}, Test={test_score:.4f}")

# RandomizedSearchCV版訓練
#詳細數據
rcv.cv_results_

print('train', rcv.cv_results_['mean_train_auc_pr'])
print('test', rcv.cv_results_['mean_test_auc_pr'])
print('train', rcv.cv_results_['mean_train_f1'])
print('test', rcv.cv_results_['mean_test_f1'])

# RandomizedSearchCV版訓練
# 最好的那一組模型
rcv.best_estimator_
# n1 = rcv.cv_results_['params' ] [0]
# n2 = rcv.cv_results_['params' ] [1]
rcv.best_params_
# print(n1)
# print(n2)
rcv.best_index_

#可以看指定的那組的參數
best_num = 7
print("params:",rcv.cv_results_['params' ] [best_num])
print("mean_train_auc_pr:",rcv.cv_results_['mean_train_auc_pr'] [best_num])
print("mean_test_auc_pr:",rcv.cv_results_['mean_test_auc_pr'] [best_num])
print("mean_train_f1:",rcv.cv_results_['mean_train_f1'] [best_num])
print("mean_test_f1:",rcv.cv_results_['mean_test_f1'] [best_num])

# 轉成 DataFrame 方便看第 index 行
import pandas as pd
results_df = pd.DataFrame(rcv.cv_results_)
cols = ['params', 'mean_train_auc_pr', 'mean_test_auc_pr', 'mean_train_recall', 'mean_test_recall', 'mean_train_f1', 'mean_test_f1']
results_df[cols]


#篩選出 auc valid>0.859(分數比較好) 且 auc train-valid < 0.05(避免過擬合)
import pandas as pd
import os
from datetime import datetime

## --- 1. 定義固定參數欄位 ---
master_param_cols = ['model__n_estimators', 'model__learning_rate', 'model__max_depth',
                     'model__scale_pos_weight', 'model__min_child_weight', 'model__reg_alpha', 'model__reg_lambda']

results_df = pd.DataFrame(rcv.cv_results_)
## 第一道篩選：基礎分數門檻
mask = (results_df['mean_train_auc_pr'] > 0.86) & (results_df['mean_test_auc_pr'] > 0.859)
good_models = results_df[mask].copy()

if not good_models.empty:
    # A. 展開參數並補齊 Master 欄位
    params_expanded = pd.json_normalize(good_models['params'])
    params_expanded = params_expanded.reindex(columns=master_param_cols + list(set(params_expanded.columns) - set(master_param_cols)))

    # B. 選擇分數欄位
    metrics_cols = metrics_cols = [
        'mean_train_auc_pr', 'mean_test_auc_pr',
        'mean_train_f1', 'mean_test_f1',
        'mean_train_precision_macro', 'mean_test_precision_macro',
        'mean_train_recall', 'mean_test_recall'
    ]

    # C. 建立本次初步 DataFrame
    new_data = pd.concat([params_expanded.reset_index(drop=True),
                          good_models[metrics_cols].reset_index(drop=True)], axis=1)

    # D. 計算過擬合差距 (Gap)
    new_data['auc_pr_gap'] = new_data['mean_train_auc_pr'] - new_data['mean_test_auc_pr']

    # ✨【新增重點】：第二道篩選 - 嚴格過濾掉 Gap > 0.05 的組合
    # 只有符合 auc_pr_gap <= 0.05 的才會留下來
    new_data = new_data[new_data['auc_pr_gap'] <= 0.05].copy()

    if not new_data.empty:
        new_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # --- 2. 存檔邏輯 ---
        file_path = '/content/drive/MyDrive/XGBoost/all_qualified_models.csv'

        if os.path.isfile(file_path):
            old_data = pd.read_csv(file_path)
            combined_df = pd.concat([old_data, new_data], ignore_index=True, sort=False)
        else:
            combined_df = new_data

        # 排序：優先看測試集表現
        # combined_df = combined_df.sort_values(by='mean_test_auc_pr', ascending=False)
        # combined_df.to_csv(file_path, index=False)
        # 按時間排序，讓最新的實驗結果在最上面
        combined_df = combined_df.sort_values(by='timestamp', ascending=False)
        combined_df.to_csv(file_path, index=False)

        print(f"✅ 品管通過！本次共有 {len(new_data)} 組組合符合 Gap <= 0.05 並已存檔。")
    else:
        print("⚠️ 雖然分數達標，但所有組合的 Gap 都大於 0.05（過擬合嚴重），本次不予存檔。")
else:
    print("❌ 分數未達標，不予存檔。")

#儲存模型
import pickle

with open("model.bin", "wb") as f:
    pickle.dump(rcv.best_estimator_, f)

#查看目前儲存的參數和分數
import pandas as pd
# 確保你已經執行過 drive.mount('/content/drive')
file_path1 = '/content/drive/MyDrive/XGBoost/all_qualified_models.csv'
try:
    history = pd.read_csv(file_path1)
    print("✅ 成功讀取雲端硬碟紀錄！")
    display(history)
except FileNotFoundError:
    print("❌ 找不到檔案！請檢查：")
    print("1. 左側資料夾是否有出現 'drive' 資料夾？(沒掛載)")
    print("2. 檔案是否真的存在於 MyDrive 根目錄下？")


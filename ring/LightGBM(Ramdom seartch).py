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

#載入訓練用資料
import pandas as pd
import numpy as np
data = pd.read_csv('/content/drive/MyDrive/XGBoost/loan_train_36000.csv')
data.head()

#列出欄位名稱
data.columns

## 定義需要進行 Log 轉換的數值欄位
# 收入與貸款金額通常有很長的尾巴（少數人極高），最適合轉換
log_cols = ['person_income', 'loan_amnt']

# 3. 執行 Log 轉換
# 使用 np.log1p 可以確保數值穩定性
for col in log_cols:
    data[col + '_log'] = np.log1p(data[col])

# 4. 查看轉換後的結果
print(data[['person_income', 'person_income_log', 'loan_amnt', 'loan_amnt_log']].head())

##新增特徵
#「信用價值比」 (Credit Value Ratio)
data['credit_value_ratio'] = data['loan_int_rate'] / data['credit_score']
print(data[['credit_value_ratio']].head())

#「年資負擔比」 (Experience Burden Ratio)
data['experience_burden_ratio'] = data['loan_percent_income'] / (data['person_emp_exp']+1)
print(data[['experience_burden_ratio']].head())

# 2026/2/20新增
# 財務壓力指數 (Financial Stress Index)
# 邏輯：如果一個人的「收入負債比」很高，同時他的「貸款利率」也很高，這代表他每個月不僅要還很多錢，而且裡面大部分都是利息！這種人的違約機率會呈現指數級上升。
# 強強聯手：將第一名和第四名的特徵相乘
data['financial_stress_index'] = data['loan_percent_income'] * (data['loan_int_rate']/100)

# 剩餘可用絕對所得 (Disposable Income)
# 邏輯：loan_percent_income 是個「比例」。但同樣是貸款佔收入的50%:月入20萬的人，繳完貸款還有10萬可以生活（很安全）。月入4萬的人，繳完貸款只剩2萬（非常危險，一有意外就違約）。
# 假設您原本的 DataFrame 裡有未取 log 的 person_income
data['disposable_income'] = data['person_income'] * (1 - data['loan_percent_income']/100)
# 如果數值很大，建議可以加上 log 處理
data['disposable_income_log'] = np.log1p(data['disposable_income'])

# 首次信用啟用的年齡 (Age at First Credit)
# 邏輯：目前的person_age和cb_person_cred_hist_length(信用歷史長度)單獨看都很弱。但如果你把它們相減：某人40歲，信用歷史才2年(38歲才辦第一張卡/貸款)➡️異常，可能之前信用破產過，或是小白。某人25歲，信用歷史3年(22歲開始累積)➡️正常。
# 搶救弱特徵：抓出異常的信用起步時間
data['age_at_first_credit'] = data['person_age'] - data['cb_person_cred_hist_length']

##選取放入模型的特徵
X = data[['person_age', 'person_gender', 'person_education',
      'person_income_log', 'person_emp_exp',
       'person_home_ownership', 'loan_amnt_log',
       'loan_intent', 'loan_int_rate',
       'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score','credit_value_ratio','experience_burden_ratio']]
# ,'credit_value_ratio','experience_burden_ratio'

y = data['loan_status']

## RandomizedSearchCV版訓練
import lightgbm as lgb

param_grid = [
    {
        'n_estimators': [1300],              # 配合早停(Early Stopping)通常設大
        'learning_rate': [0.05],       # 較小的學習率能學得更細
        'num_leaves': [15],              # Leaf-wise 的核心，建議小於 2^(max_depth)
        'max_depth': [4],             # 限制深度防止過擬合，-1 為不限制

        # 處理不平衡 (妳特別要求的)
        'is_unbalance': [True],              # 自動權重平衡，適合 22% vs 78%
        # 或者可以使用 'model__scale_pos_weight': [3.5], (兩者選其一)

        # 針對過擬合與邊界值的優化：
        'min_child_samples': [300],  # 相當於 min_data_in_leaf，越大模型越保守
        'reg_alpha': [1],            # L1 正則化 (reg_alpha)
        'reg_lambda': [1],            # L2 正則化 (reg_lambda)

        # 隨機採樣，增加模型穩定性
        #'feature_fraction': [0.7, 0.8],      # 每次訓練隨機選取 70%-80% 特徵
        #'bagging_fraction': [0.7, 0.8],      # 每次訓練隨機選取 70%-80% 資料
        #'bagging_freq': [5],                 # 每 5 輪進行一次隨機採樣
        'n_jobs': [-1]
    }
]
model = lgb.LGBMClassifier(random_state=42)


## 放入Pipeline
from sklearn.pipeline import Pipeline
#from category_encoders import TargetEncoder
new_param_grid = []
for pg in param_grid:
    new_param_grid.append({f"model__{k}": v for k, v in pg.items()})

pipe = Pipeline([('model', model)])
pipe


##自己切驗證集不用CV的RandomizedSearch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, PredefinedSplit, RandomizedSearchCV

# 1. 第一步：先切分資料
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 第二步：建立 PredefinedSplit (這是關鍵)
# 我們需要告訴 RandomizedSearchCV 哪一部分是訓練，哪一部分是驗證
# -1 代表訓練集, 0 代表驗證集
train_indices = np.full(X_train.shape[0], -1)
val_indices = np.full(X_val.shape[0], 0)
test_fold = np.append(train_indices, val_indices)

ps = PredefinedSplit(test_fold)

# 3. 第三步：合併資料 (因為 RandomizedSearchCV.fit 需要收一個整體的 X 和 y)
X_combined = pd.concat([X_train, X_val])
y_combined = pd.concat([y_train, y_val])

# 4. 第四步：設定 RandomizedSearchCV
scoring = {
    'recall': 'recall',
    'precision_macro': 'precision_macro',
    'f1': 'f1',
    'auc_pr': 'average_precision',
    'roc_auc': 'roc_auc',
    'accuracy': 'accuracy'
}

rcv = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=new_param_grid,
    return_train_score=True,
    scoring=scoring,
    refit='auc_pr',
    n_iter=1,
    cv=ps,        # <--- 這裡改用我們定義好的 PredefinedSplit，就等於沒有跑 CV 了
    verbose=2,
    n_jobs=-1
)

# 找出妳要標註為類別的欄位在 X 中的索引
cat_cols = ['person_gender','person_home_ownership', 'loan_intent']
cat_indices = [X.columns.get_loc(col) for col in cat_cols]

# 看看 cat_indices 是不是一串數字，例如 [0, 2, 5, 6, 8]
print(f"類別欄位的索引位置: {cat_indices}")

# 在 RandomizedSearchCV 的 fit 中傳入參數
# 注意：因為有 Pipeline，參數名稱要加上 "model__" 前綴
fit_params = {
    'model__categorical_feature': cat_indices,  # 告訴模型哪些是類別欄位
    'model__eval_set':[(X_train, y_train), (X_val, y_val)], # 告訴模型：看這組資料來決定要不要停止
    'model__eval_names':['train', 'valid'],
    'model__callbacks': [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ],
    'model__eval_metric': ['binary_logloss', 'average_precision']
}

# 使用合併後的資料進行訓練
rcv.fit(X_combined, y_combined, **fit_params)

# 訓練完成後，你可以直接看結果
print(f"最佳參數: {rcv.best_params_}")
print(f"驗證集最佳 AUC-PR: {rcv.best_score_}")

##訓練分數
print("平均分數 (train vs test)：")
for score_name in scoring:
    train_key = f'mean_train_{score_name}'
    test_key = f'mean_test_{score_name}'
    train_score = rcv.cv_results_[train_key].mean()
    test_score = rcv.cv_results_[test_key].mean()
    print(f"{score_name}: Train={train_score:.4f}, Test={test_score:.4f}")

##存模型
import pickle
import zipfile
import os
from datetime import datetime
from google.colab import drive

# 1. 掛載 Google Drive (如果還沒掛載的話)
drive.mount('/content/drive')

# 2. 設定儲存路徑與檔名
file_path = '/content/drive/MyDrive/XGBoost'
model_name = "LightGBM_Model"
version = "v1"
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# 確保資料夾存在，如果不存在就建立它
if not os.path.exists(file_path):
    os.makedirs(file_path)
    print(f"📁 已建立新資料夾: {file_path}")

# 組合完整的儲存路徑 (檔名包含路徑)
zip_filename = f"model_{model_name}_{version}_{timestamp}.zip"
full_zip_path = os.path.join(file_path, zip_filename)
internal_filename = "model.bin"

# 3. 執行壓縮儲存
try:
    with zipfile.ZipFile(full_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 將模型序列化
        model_bytes = pickle.dumps(rcv.best_estimator_)
        # 寫入 ZIP
        zf.writestr(internal_filename, model_bytes)

    print(f"✨ 儲存成功！")
    print(f"📍 儲存位置: {full_zip_path}")
except Exception as e:
    print(f"❌ 儲存失敗，錯誤原因: {e}")


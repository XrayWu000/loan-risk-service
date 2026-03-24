import numpy as np
import pandas as pd
import torch
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from pytorch_tabnet.tab_model import TabNetClassifier


# ========== 1. 讀資料 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "loan_train_36000.csv")   # 確保路徑正確
TARGET = "loan_status"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET].values


# ========== 2. 類別型特徵處理 ==========
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

cat_idxs = []
cat_dims = []

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    cat_idxs.append(X.columns.get_loc(col))
    cat_dims.append(len(le.classes_))


# ========== 3. Train / Valid split ==========
X_train, X_valid, y_train, y_valid = train_test_split(
    X.values,
    y,
    test_size=0.2,
    random_state=37,
    stratify=y
)


# ========== 4. 建立 TabNet ==========
model = TabNetClassifier(
    n_d=16,
    n_a=16,
    n_steps=5,
    gamma=1.5,
    lambda_sparse=1e-4,
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=1,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type="entmax",
    device_name="cuda"
)


# ========== 5. 訓練 ==========
model.fit(
    X_train=X_train,
    y_train=y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric=["auc"],
    max_epochs=200,
    patience=20,
    batch_size=8192,          # 5070 Ti 建議值
    virtual_batch_size=1024,
    num_workers=0,
    drop_last=False
)


# ========== 6. 驗證 ==========
y_proba = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, y_proba)

print(f"Validation ROC-AUC: {auc:.4f}")


# ========== 7. 存模型 ==========
model.save_model("tabnet_loan_default-1")
print("Model saved.")

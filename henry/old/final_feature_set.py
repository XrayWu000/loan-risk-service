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
DATA_PATH = os.path.join(BASE_DIR, "loan_train_36000.csv")
TARGET = "loan_status"

df = pd.read_csv(DATA_PATH)


# ========== 2. 特徵工程（只建立必要欄位） ==========
df["interest_pressure"] = df["loan_int_rate"] * df["loan_percent_income"]
df["age_bucket"] = pd.cut(
    df["person_age"],
    bins=[0, 30, 50, 100],
    labels=["young", "mid", "senior"]
)
df["log_income"] = np.log1p(df["person_income"])
df["log_loan_amnt"] = np.log1p(df["loan_amnt"])


# ========== 3. 僅保留指定 7 欄 ==========
FEATURES = [
    "loan_int_rate",
    "loan_percent_income",
    "interest_pressure",
    "debt_pressure",
    "log_income",
    "log_loan_amnt",
    "person_home_ownership",
]

X = df[FEATURES].copy()
y = df[TARGET].values


# ========== 4. 類別型特徵處理 ==========
categorical_cols = ["person_home_ownership"]

cat_idxs = []
cat_dims = []

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    cat_idxs.append(X.columns.get_loc(col))
    cat_dims.append(len(le.classes_))


# ========== 5. Train / Validation split ==========
X_train, X_valid, y_train, y_valid = train_test_split(
    X.values,
    y,
    test_size=0.2,
    random_state=37,
    stratify=y
)


# ========== 6. 建立 TabNet ==========
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


# ========== 7. 訓練 ==========
model.fit(
    X_train=X_train,
    y_train=y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric=["auc"],
    max_epochs=1000,
    patience=20,
    batch_size=32768,          # ← 原 8192
    virtual_batch_size=2048,   # ← 原 1024
    num_workers=0,
    drop_last=False
)


# ========== 8. 驗證 ==========
y_proba = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, y_proba)

print(f"Validation ROC-AUC (7 features): {auc:.4f}")


# ========== 9. 存模型 ==========
model.save_model("tabnet_loan_default_7features")
print("Model saved.")
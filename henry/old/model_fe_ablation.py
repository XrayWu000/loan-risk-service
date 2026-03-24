import numpy as np
import pandas as pd
import torch
import os
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from pytorch_tabnet.tab_model import TabNetClassifier


# ================= 1. 讀資料 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "loan_train_36000.csv")
TARGET = "loan_status"

df = pd.read_csv(DATA_PATH)


# ================= 2. 基礎特徵 =================
BASE_FEATURES = [c for c in df.columns if c != TARGET]


# ================= 3. 定義 6 個特徵工程 =================
def add_engineered_features(df):
    df = df.copy()

    df["log_income"] = np.log1p(df["person_income"])
    df["log_loan_amnt"] = np.log1p(df["loan_amnt"])
    df["debt_pressure"] = df["loan_amnt"] / (df["person_income"] + 1)
    df["interest_pressure"] = df["loan_int_rate"] * df["loan_percent_income"]

    df["credit_score_bucket"] = pd.cut(
        df["credit_score"],
        bins=[0, 600, 700, 850],
        labels=["low", "mid", "high"]
    )

    df["age_bucket"] = pd.cut(
        df["person_age"],
        bins=[0, 30, 50, 100],
        labels=["young", "mid", "senior"]
    )

    return df


ENGINEERED_FEATURES = [
    "log_income",
    "log_loan_amnt",
    "debt_pressure",
    "interest_pressure",
    "credit_score_bucket",
    "age_bucket",
]


df = add_engineered_features(df)


# ================= 4. 類別編碼 =================
X_full = df.drop(columns=[TARGET])
y = df[TARGET].values

categorical_cols = X_full.select_dtypes(include=["object", "category"]).columns.tolist()

cat_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_full[col] = le.fit_transform(X_full[col].astype(str))
    cat_encoders[col] = le


# ================= 5. Train / Valid split =================
X_train_all, X_valid_all, y_train, y_valid = train_test_split(
    X_full,
    y,
    test_size=0.2,
    random_state=37,
    stratify=y
)


# ================= 6. Feature ablation =================
RESULTS = []
MODEL_DIR = os.path.join(BASE_DIR, "fe_models")
os.makedirs(MODEL_DIR, exist_ok=True)

for k in range(1, len(ENGINEERED_FEATURES) + 1):
    for subset in itertools.combinations(ENGINEERED_FEATURES, k):

        FEATURES = BASE_FEATURES + list(subset)

        X_tr = X_train_all[FEATURES].values
        X_va = X_valid_all[FEATURES].values

        cat_idxs = [FEATURES.index(c) for c in categorical_cols if c in FEATURES]
        cat_dims = [X_full[c].nunique() for c in categorical_cols if c in FEATURES]

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
            device_name="cuda",
            verbose=0
        )

        print(f"Training with engineered features: {subset}")

        model.fit(
            X_tr, y_train,
            eval_set=[(X_va, y_valid)],
            eval_metric=["auc"],
            max_epochs=200,
            patience=20,
            batch_size=8192,
            virtual_batch_size=1024,
            drop_last=False
        )

        y_proba = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_valid, y_proba)

        model_name = f"fe_{'_'.join(subset)}"
        model.save_model(os.path.join(MODEL_DIR, model_name))

        RESULTS.append({
            "engineered_features": subset,
            "num_engineered": len(subset),
            "roc_auc": auc
        })

        pd.DataFrame(RESULTS).to_csv(
            os.path.join(MODEL_DIR, "fe_ablation_results.csv"),
            index=False
        )


print("=== Feature Engineering Ablation Finished ===")
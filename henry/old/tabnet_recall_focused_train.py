import os
import json
import itertools
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)

from pytorch_tabnet.tab_model import TabNetClassifier


# ================= 基本設定 =================
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "loan_train_36000.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "overnight_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "loan_status"


# ================= 讀資料 =================
df = pd.read_csv(DATA_PATH)

ALL_FEATURES = [c for c in df.columns if c != TARGET]


# ================= Feature Sets（≤3 drop） =================
feature_sets = {
    "all": ALL_FEATURES,
    "drop_interest": [c for c in ALL_FEATURES if c != "loan_int_rate"],
    "drop_income": [c for c in ALL_FEATURES if c != "person_income"],
    "drop_percent": [c for c in ALL_FEATURES if c != "loan_percent_income"],
    "drop_interest_income": [
        c for c in ALL_FEATURES if c not in ["loan_int_rate", "person_income"]
    ],
    "drop_interest_percent": [
        c for c in ALL_FEATURES if c not in ["loan_int_rate", "loan_percent_income"]
    ],
    "drop_interest_income_percent": [
        c for c in ALL_FEATURES
        if c not in ["loan_int_rate", "person_income", "loan_percent_income"]
    ],
}


# ================= 模型結構 grid =================
param_grid = {
    "n_d": [32, 48],
    "n_a": [32, 48],
    "n_steps": [5, 6, 7],
    "gamma": [1.5, 1.8],
    "lr": [0.02, 0.01],
}

model_params = list(itertools.product(
    param_grid["n_d"],
    param_grid["n_a"],
    param_grid["n_steps"],
    param_grid["gamma"],
    param_grid["lr"],
))


# ================= 訓練策略 =================
train_strategies = [
    {"name": "auc_w1.0", "eval": ["auc"], "w_pos": 1.0},
    {"name": "auc_w1.1", "eval": ["auc"], "w_pos": 1.1},
    {"name": "auc_w1.2", "eval": ["auc"], "w_pos": 1.2},
    {"name": "auc_w1.3", "eval": ["auc"], "w_pos": 1.3},
    {"name": "auc_w1.5", "eval": ["auc"], "w_pos": 1.5},
    {"name": "auc_bal_w1.0", "eval": ["auc", "balanced_accuracy"], "w_pos": 1.0},
    {"name": "auc_bal_w1.1", "eval": ["auc", "balanced_accuracy"], "w_pos": 1.1},
    {"name": "auc_bal_w1.2", "eval": ["auc", "balanced_accuracy"], "w_pos": 1.2},
]


# ================= 開跑 =================
results = []

for feat_name, FEATURES in feature_sets.items():

    df_f = df[FEATURES + [TARGET]].copy()

    # 類別處理
    cat_cols = df_f[FEATURES].select_dtypes(include="object").columns.tolist()
    cat_idxs, cat_dims = [], []

    for c in cat_cols:
        le = LabelEncoder()
        df_f[c] = le.fit_transform(df_f[c].astype(str))
        cat_idxs.append(FEATURES.index(c))
        cat_dims.append(len(le.classes_))

    X = df_f[FEATURES].values
    y = df_f[TARGET].values

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    for n_d, n_a, n_steps, gamma, lr in model_params:
        for strat in train_strategies:

            weights = np.where(y_tr == 1, strat["w_pos"], 1.0).astype(np.float32)

            model_name = (
                f"{feat_name}_nd{n_d}_na{n_a}_ns{n_steps}"
                f"_g{gamma}_lr{lr}_{strat['name']}"
            )

            print("Training:", model_name)

            model = TabNetClassifier(
                n_d=n_d,
                n_a=n_a,
                n_steps=n_steps,
                gamma=gamma,
                lambda_sparse=1e-4,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=lr),
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                cat_emb_dim=1,
                device_name="cuda",
                verbose=0
            )

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric=strat["eval"],
                max_epochs=120,
                patience=20,
                batch_size=32768,
                virtual_batch_size=2048,
                weights=weights,
                drop_last=False
            )

            y_proba = model.predict_proba(X_va)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            res = {
                "model": model_name,
                "feature_set": feat_name,
                "dropped_features": list(set(ALL_FEATURES) - set(FEATURES)),
                "n_d": n_d,
                "n_a": n_a,
                "n_steps": n_steps,
                "gamma": gamma,
                "lr": lr,
                "strategy": strat["name"],
                "roc_auc": roc_auc_score(y_va, y_proba),
                "accuracy": accuracy_score(y_va, y_pred),
                "precision": precision_score(y_va, y_pred, zero_division=0),
                "recall": recall_score(y_va, y_pred, zero_division=0),
                "f1": f1_score(y_va, y_pred, zero_division=0),
            }

            model_path = os.path.join(OUTPUT_DIR, f"{model_name}.zip")
            model.save_model(model_path)

            with open(model_path.replace(".zip", ".features.json"), "w") as f:
                json.dump(
                    {"features": FEATURES, "dropped": res["dropped_features"]},
                    f, indent=2
                )

            results.append(res)
            pd.DataFrame(results).to_csv(
                os.path.join(OUTPUT_DIR, "overnight_results.csv"),
                index=False
            )


print("=== Overnight search finished ===")

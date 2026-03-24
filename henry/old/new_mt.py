import numpy as np
import pandas as pd
import torch
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from pytorch_tabnet.tab_model import TabNetClassifier


def add_engineered_features(df):
    df = df.copy()

    df["log_income"] = np.log1p(df["person_income"])
    df["log_loan_amnt"] = np.log1p(df["loan_amnt"])
    df["interest_pressure"] = df["loan_int_rate"] * df["loan_percent_income"]
    df["age_bucket"] = pd.cut(
        df["person_age"],
        bins=[0, 30, 50, 100],
        labels=["young", "mid", "senior"]
    )

    # 🔒 強制指定完整類別全集
    df["age_bucket"] = pd.Categorical(
        df["age_bucket"],
        categories=["young", "mid", "senior"]
    )

    return df


def main():
    # ================= 1. ???? =================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "loan_train_36000.csv")
    TARGET = "loan_status"

    MODEL_DIR = os.path.join(BASE_DIR, "fe_models_4f")
    os.makedirs(MODEL_DIR, exist_ok=True)

    NUM_WORKERS = 8
    torch.set_num_threads(NUM_WORKERS)

    # ================= 2. ??? =================
    df = pd.read_csv(DATA_PATH)

    # ================= 3. engineered features?4?? =================
    ENGINEERED_FEATURES = [
        "log_income",
        "log_loan_amnt",
        "interest_pressure",
        "age_bucket",
    ]

    df = add_engineered_features(df)

    # ================= 4. ???? =================
    BASE_FEATURES = [c for c in df.columns if c != TARGET]
    FEATURES = BASE_FEATURES  # ??? engineered features

    X_full = df[FEATURES]
    y = df[TARGET].values

    # ================= 5. ?????? =================
    categorical_cols = X_full.select_dtypes(include=["object", "category"]).columns.tolist()

    cat_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_full[col] = le.fit_transform(X_full[col].astype(str))
        cat_encoders[col] = le

    # ================= 6. Train / Valid split =================
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_full,
        y,
        test_size=0.2,
        random_state=37,
        stratify=y
    )

    # ================= 7. TabNet ?? =================
    cat_idxs = [FEATURES.index(c) for c in categorical_cols]
    cat_dims = [X_full[c].nunique() for c in categorical_cols]

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
        verbose=1
    )

    model.fit(
        X_train.values, y_train,
        eval_set=[(X_valid.values, y_valid)],
        eval_metric=["auc"],
        max_epochs=1000,
        patience=50,
        batch_size=8192,
        virtual_batch_size=128,
        num_workers=NUM_WORKERS,
        drop_last=False
    )

    y_proba = model.predict_proba(X_valid.values)[:, 1]
    auc = roc_auc_score(y_valid, y_proba)

    print(f"Validation ROC-AUC = {auc:.4f}")

    # ================= 8. ??? =================
    MODEL_PATH = os.path.join(MODEL_DIR, "fe_4f_model.zip")
    model.save_model(MODEL_PATH)

    # ================= 9. ? metadata =================
    meta = {
        "features": FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "categorical_cols": categorical_cols,
        "target": TARGET
    }

    with open(os.path.join(MODEL_DIR, "features.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    joblib.dump(cat_encoders, os.path.join(MODEL_DIR, "cat_encoders.pkl"))

    print("=== Saved Artifacts ===")
    print("Model      :", MODEL_PATH)
    print("features.json")
    print("cat_encoders.pkl")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()

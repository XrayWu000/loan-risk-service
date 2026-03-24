# report_pack.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

from sklearn.inspection import permutation_importance


DATA_PATH = Path("henry/loan_train_36000.csv")  # 你若檔名不同就改這行
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )
    return pre


def eval_model(name: str, pipe: Pipeline, X_tr, y_tr, X_va, y_va) -> dict:
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_va)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_va, pred)),
        "precision": float(precision_score(y_va, pred, zero_division=0)),
        "recall": float(recall_score(y_va, pred, zero_division=0)),
        "f1": float(f1_score(y_va, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_va, proba)),
        "cm": confusion_matrix(y_va, pred).tolist(),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到資料檔：{DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)

    # 1) 基本清理：移除 ID（若存在）
    for id_col in ["ID", "Id", "id"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    if "loan_status" not in df.columns:
        raise ValueError("找不到目標欄位 loan_status")

    y = df["loan_status"].astype(int)
    X = df.drop(columns=["loan_status"])

    # 2) 分層切分（Stratified Split）
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pre = build_preprocess(X)

    # 3) 三個可解釋/傳統模型
    lr = Pipeline(steps=[
        ("preprocess", pre),
        ("model", LogisticRegression(max_iter=2000, n_jobs=None, class_weight="balanced")),
    ])

    dt = Pipeline(steps=[
        ("preprocess", pre),
        ("model", DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")),
    ])

    rf = Pipeline(steps=[
        ("preprocess", pre),
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE, n_estimators=400, n_jobs=-1, class_weight="balanced"
        )),
    ])

    results = []
    cms = {}

    for name, pipe in [("LogisticRegression", lr), ("DecisionTree", dt), ("RandomForest", rf)]:
        r = eval_model(name, pipe, X_tr, y_tr, X_va, y_va)
        results.append({k: r[k] for k in ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]})
        cms[name] = r["cm"]

    # 4) 參數優化（Hyperparameter Tuning）示範：RandomForest 小範圍 GridSearch
    # 目的：報告能說「我們有做參數調整」，不是拼極限分數
    rf_tune = Pipeline(steps=[
        ("preprocess", pre),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")),
    ])
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 8, 14],
        "model__min_samples_leaf": [1, 3, 8],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(rf_tune, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)

    best_rf = gs.best_estimator_
    r_best = eval_model("RandomForest_Tuned", best_rf, X_tr, y_tr, X_va, y_va)
    results.append({k: r_best[k] for k in ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]})
    cms["RandomForest_Tuned"] = r_best["cm"]

    # 5) 特徵篩選（Feature Selection）證據：Permutation Importance（用 tuned RF）
    perm = permutation_importance(
        best_rf, X_va, y_va, scoring="f1", n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    imp.to_csv(OUT_DIR / "permutation_importance_all.csv", index=False)
    imp.head(20).to_csv(OUT_DIR / "permutation_importance_top20.csv", index=False)

    # 6) 特徵刪減實驗：移除 bottom 20%（或 importance <= 0）
    cutoff = np.quantile(imp["importance_mean"], 0.2)
    drop_features = imp.loc[imp["importance_mean"] <= max(cutoff, 0.0), "feature"].tolist()
    keep_features = [c for c in X.columns if c not in drop_features]

    # 原始 tuned RF
    base = eval_model("RF_Tuned_Full", best_rf, X_tr, y_tr, X_va, y_va)

    # 刪減後 tuned RF（只用 keep_features）
    X_tr_r = X_tr[keep_features].copy()
    X_va_r = X_va[keep_features].copy()
    pre_r = build_preprocess(X_tr_r)

    best_params = gs.best_params_
    rf_reduced = Pipeline(steps=[
        ("preprocess", pre_r),
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            n_estimators=best_params["model__n_estimators"],
            max_depth=best_params["model__max_depth"],
            min_samples_leaf=best_params["model__min_samples_leaf"],
        )),
    ])
    reduced = eval_model("RF_Tuned_Reduced", rf_reduced, X_tr_r, y_tr, X_va_r, y_va)

    comp = pd.DataFrame([
        {"setting": "full_features", **{k: base[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
         "n_features": int(len(X.columns))},
        {"setting": "reduced_features", **{k: reduced[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
         "n_features": int(len(keep_features))},
    ])
    comp.to_csv(OUT_DIR / "feature_reduction_compare.csv", index=False)

    # 7) 輸出總表
    pd.DataFrame(results).to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    (OUT_DIR / "confusion_matrices.json").write_text(json.dumps(cms, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Done. 產出檔案：")
    for p in sorted(OUT_DIR.glob("*")):
        print(" -", p.as_posix())


if __name__ == "__main__":
    main()

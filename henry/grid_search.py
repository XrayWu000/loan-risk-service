import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, ParameterGrid

# =========================
# 設定
# =========================
DATA_PATH = "loan_train_36000.csv"
RESULTS_PATH = "lgbm_grid_results.csv"
TARGET = "loan_status"

FINAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_int_rate",
    "cb_person_cred_hist_length",
    "interest_pressure",
    "person_emp_exp",
    "person_age",
    "person_gender",
    "loan_amnt",
    "log_income",
]

CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "person_gender",
]


# =========================
# 建資料
# =========================
def build_dataset():
    df = pd.read_csv(DATA_PATH)

    df["log_income"] = np.log1p(df["person_income"])
    df["interest_pressure"] = df["loan_int_rate"] * df["loan_percent_income"]

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    X = df[FINAL_FEATURES]
    y = df[TARGET].astype(int)

    return X, y


# =========================
# 續跑機制
# =========================
def load_done_keys():
    if not os.path.exists(RESULTS_PATH):
        return set()

    df = pd.read_csv(RESULTS_PATH)
    if "param_key" not in df.columns:
        return set()

    return set(df["param_key"].astype(str).tolist())


def param_to_key(param):
    return json.dumps(param, sort_keys=True)


def append_result(row):
    df_row = pd.DataFrame([row])
    if os.path.exists(RESULTS_PATH):
        df_row.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(RESULTS_PATH, index=False)


# =========================
# 主程式
# =========================
def main():

    X, y = build_dataset()

    dtrain = lgb.Dataset(
        X,
        label=y,
        categorical_feature=CATEGORICAL_COLS,
        free_raw_data=False,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ===== 972 組參數 =====
    param_grid = {
        "n_estimators": [800, 1200, 1600],
        "learning_rate": [0.01, 0.03, 0.05],
        "num_leaves": [15, 31, 63],
        "max_depth": [-1, 10, 20],
        "min_child_samples": [10, 20, 50],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    grid = list(ParameterGrid(param_grid))
    total = len(grid)

    print(f"Total param sets: {total}")
    print(f"Total trainings (5-fold): {total * 5}")

    done_keys = load_done_keys()
    print(f"Already done: {len(done_keys)}")

    best_auc = -1.0
    best_param = None

    base_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "class_weight": "balanced",
        "seed": 42,
        "feature_pre_filter": False,
    }

    for idx, p in enumerate(grid, 1):

        key = param_to_key(p)
        if key in done_keys:
            continue

        num_boost_round = int(p["n_estimators"])

        params = dict(base_params)
        params.update({
            "learning_rate": p["learning_rate"],
            "num_leaves": p["num_leaves"],
            "max_depth": p["max_depth"],
            "min_child_samples": p["min_child_samples"],
            "subsample": p["subsample"],
            "colsample_bytree": p["colsample_bytree"],
        })

        cv_result = lgb.cv(
            params=params,
            train_set=dtrain,
            folds=skf,
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(100, verbose=False),
            ],
            seed=42,
        )

        # === 安全抓 AUC key ===
        auc_key = None
        for k in cv_result.keys():
            if "auc" in k and "mean" in k:
                auc_key = k
                break

        if auc_key is None:
            raise ValueError(f"AUC key not found. Keys: {cv_result.keys()}")

        auc_list = cv_result[auc_key]
        best_iter = len(auc_list)
        auc_best = float(np.max(auc_list))

        row = {
            "param_key": key,
            "cv_auc_best": auc_best,
            "best_iteration": best_iter,
            "n_estimators": p["n_estimators"],
            "learning_rate": p["learning_rate"],
            "num_leaves": p["num_leaves"],
            "max_depth": p["max_depth"],
            "min_child_samples": p["min_child_samples"],
            "subsample": p["subsample"],
            "colsample_bytree": p["colsample_bytree"],
        }

        append_result(row)
        done_keys.add(key)

        if auc_best > best_auc:
            best_auc = auc_best
            best_param = p

        if idx % 10 == 0:
            print(f"[{idx}/{total}] current_auc={auc_best:.6f} | best_auc={best_auc:.6f}")

    print("\n=== GRID SEARCH FINISHED ===")
    print("Best CV AUC:", best_auc)
    print("Best Params:", best_param)

    # === 顯示前 10 名 ===
    df = pd.read_csv(RESULTS_PATH)
    df = df.sort_values("cv_auc_best", ascending=False)

    print("\nTop 10 Results:")
    print(df.head(10)[[
        "cv_auc_best",
        "best_iteration",
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree"
    ]])


if __name__ == "__main__":
    main()
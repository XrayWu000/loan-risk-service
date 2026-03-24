import os
import itertools
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score
)

from pytorch_tabnet.tab_model import TabNetClassifier


# =========================================================
# 1) Paths & config
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../loan-approval-ml/henry
PROJECT_ROOT = os.path.dirname(BASE_DIR)                      # .../loan-approval-ml

DATA_PATH = os.path.join(BASE_DIR, "loan_train_36000.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_models")
RESULT_CSV = os.path.join(PROJECT_ROOT, "gridsearch_results_tabnet.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "loan_status"
RANDOM_STATE = 42


# =========================================================
# 2) Load data (use ALL feature columns)
# =========================================================
df = pd.read_csv(DATA_PATH)

# 檢查必要欄位是否存在（避免欄名拼錯浪費時間）
required_cols = [
    "person_age","person_gender","person_education","person_income","person_emp_exp",
    "person_home_ownership","loan_amnt","loan_intent","loan_int_rate","loan_percent_income",
    "cb_person_cred_hist_length","credit_score", TARGET
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].values


# =========================================================
# 3) Categorical columns (IMPORTANT: they are already numeric-coded)
#    We still need to remap them to 0..K-1 for TabNet embeddings
# =========================================================
categorical_cols = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent"
]

# 將類別欄位統一做 factorize -> 0..K-1（不管原本是 int 或 str 都 OK）
cat_dims = []
for col in categorical_cols:
    codes, uniques = pd.factorize(X[col], sort=True)
    X[col] = codes.astype(np.int64)
    cat_dims.append(len(uniques))

cat_idxs = [X.columns.get_loc(col) for col in categorical_cols]


# =========================================================
# 4) Train/Validation split
# =========================================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X.values,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)


# =========================================================
# 5) Hyperparameter grid (long-run)
#    3*3*3*3*2*2 = 108 models
# =========================================================
param_grid = {
    "n_d": [16, 24, 32],
    "n_a": [16, 24, 32],
    "n_steps": [4, 5, 6],
    "gamma": [1.3, 1.5, 1.7],
    "lambda_sparse": [1e-4, 1e-3],
    "lr": [0.01, 0.02],
}

param_combinations = list(itertools.product(
    param_grid["n_d"],
    param_grid["n_a"],
    param_grid["n_steps"],
    param_grid["gamma"],
    param_grid["lambda_sparse"],
    param_grid["lr"],
))


# =========================================================
# 6) Grid search loop + save every model + record many metrics
# =========================================================
results = []
start_time = datetime.now()

for idx, (n_d, n_a, n_steps, gamma, lambda_sparse, lr) in enumerate(param_combinations, 1):

    print(f"\n=== Model {idx}/{len(param_combinations)} ===")
    print(f"n_d={n_d}, n_a={n_a}, n_steps={n_steps}, gamma={gamma}, lambda_sparse={lambda_sparse}, lr={lr}")

    model = TabNetClassifier(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=lr),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 20, "gamma": 0.9},
        verbose=0,
        device_name="cuda"
    )

    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=["auc"],
        max_epochs=100,
        patience=15,
        batch_size=32768,          # ← 原 8192
        virtual_batch_size=2048,   # ← 原 1024
        drop_last=False
    )

    # ---- predictions
    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]

    # ---- metrics (ranking / probability quality)
    roc_auc = roc_auc_score(y_valid, y_proba)
    pr_auc = average_precision_score(y_valid, y_proba)          # PR-AUC (平均精確率)
    ll = log_loss(y_valid, y_proba)

    # ---- metrics (threshold = 0.5)
    acc = accuracy_score(y_valid, y_pred)
    bal_acc = balanced_accuracy_score(y_valid, y_pred)          # balanced accuracy (平衡準確率)
    prec_1 = precision_score(y_valid, y_pred, pos_label=1)
    rec_1 = recall_score(y_valid, y_pred, pos_label=1)
    f1_1 = f1_score(y_valid, y_pred, pos_label=1)

    # ---- confusion-derived (risk style)
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0              # false positive rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0              # false negative rate
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0              # true positive rate

    # ---- save model (NOTE: save_model auto-adds .zip)
    model_name = f"tabnet_nd{n_d}_na{n_a}_ns{n_steps}_g{gamma}_ls{lambda_sparse}_lr{lr}"
    model_path_no_ext = os.path.join(MODEL_DIR, model_name)
    model.save_model(model_path_no_ext)

    # ---- record
    results.append({
        "model_name": model_name,
        "saved_path": model_path_no_ext + ".zip",

        # hyperparameters
        "n_d": n_d,
        "n_a": n_a,
        "n_steps": n_steps,
        "gamma": gamma,
        "lambda_sparse": lambda_sparse,
        "learning_rate": lr,

        # ranking / probability
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": ll,

        # classification
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_1": prec_1,
        "recall_1": rec_1,
        "f1_1": f1_1,

        # confusion counts
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,

        # risk rates
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
    })

    # 每跑完一個模型就「即時落盤」：避免跑到一半當機全沒了
    pd.DataFrame(results).to_csv(RESULT_CSV, index=False)

    print(f"Validation ROC-AUC: {roc_auc:.5f} | Recall_1: {rec_1:.5f} | FNR: {fnr:.5f}")


# =========================================================
# 7) Final sort & report
# =========================================================
results_df = pd.DataFrame(results).sort_values(
    by=["roc_auc", "recall_1"],
    ascending=[False, False]
)

results_df.to_csv(RESULT_CSV, index=False)

print("\n=== Grid Search Finished ===")
print(f"Total time: {datetime.now() - start_time}")
print(f"Results saved to: {RESULT_CSV}")
print("\nTop 5 models:")
print(results_df.head(5)[["model_name","roc_auc","pr_auc","accuracy","balanced_accuracy","precision_1","recall_1","f1_1","fnr"]])
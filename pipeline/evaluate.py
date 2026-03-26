import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


# =========================
# 基本評估
# =========================
def evaluate_model(model, X, y, threshold=0.3, name="Dataset"):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    auc = roc_auc_score(y, prob)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)
    acc = accuracy_score(y, pred)

    print(f"\n===== {name} =====")
    print(f"AUC:       {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {acc:.4f}")

    return {
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
    }


# =========================
# Threshold tuning
# =========================
def find_best_threshold(model, X, y, target_precision=0.78):
    prob = model.predict_proba(X)[:, 1]

    best_threshold = 0.5
    best_recall = 0

    print("\n===== Threshold Tuning =====")

    for t in np.linspace(0.1, 0.9, 50):
        pred = (prob >= t).astype(int)

        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)

        if precision >= target_precision and recall > best_recall:
            best_recall = recall
            best_threshold = t

        print(f"t={t:.3f} | precision={precision:.3f} | recall={recall:.3f}")

    print("\n===== Best Threshold =====")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Recall:    {best_recall:.3f}")

    return best_threshold


# =========================
# 一次完整評估流程
# =========================
def full_evaluation(model, X_val, y_val, X_hold, y_hold):
    # 1️⃣ 找 threshold（用 validation）
    best_t = find_best_threshold(model, X_val, y_val)

    # 2️⃣ validation 評估
    evaluate_model(model, X_val, y_val, threshold=best_t, name="Validation")

    # 3️⃣ hold 評估（真正測試）
    evaluate_model(model, X_hold, y_hold, threshold=best_t, name="Holdout")

    return best_t
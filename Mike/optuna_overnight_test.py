import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve, classification_report
import matplotlib.pyplot as plt

# 1. 載入模型與測試集
model = CatBoostClassifier()
model.load_model("best_loan_model_final.cbm")
DATA_PATH = r"D:\團隊專題\loan_test_9000.csv"
test_df = pd.read_csv(DATA_PATH)

X_test = test_df.drop(columns=["loan_status"])
y_test = test_df["loan_status"]

# 2. 獲取預測機率 (Probability)
y_proba = model.predict_proba(X_test)[:, 1]

# 3. 計算 PR 曲線
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# 4. 定義你的目標 Recall (例如 0.75，代表抓出 75% 的違約者)
target_recall = 0.75
# 找到最接近目標 Recall 的索引
# 因為 recalls 是遞減的，我們找最後一個大於等於目標值的索引
idx = np.where(recalls >= target_recall)[0][-1]
best_threshold = thresholds[idx]

print(f"--- 門檻搜尋結果 ---")
print(f"目標 Recall: {target_recall}")
print(f"對應的最佳門檻 (Threshold): {best_threshold:.4f}")
print(f"此門檻下的 Precision: {precisions[idx]:.4f}")

# 5. 使用這個新門檻產出最終報告
y_pred_final = (y_proba >= best_threshold).astype(int)

print("\n=== 套用最佳門檻後的分類報告 ===")
print(classification_report(y_test, y_pred_final))

# 6. (選做) 畫圖視覺化，看看 Precision 和 Recall 的交叉點
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], label="Precision", color="blue")
plt.plot(thresholds, recalls[:-1], label="Recall", color="red")
plt.axvline(best_threshold, color="black", linestyle="--", label=f"Threshold {best_threshold:.2f}")
plt.title("Precision-Recall vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid()
plt.show()
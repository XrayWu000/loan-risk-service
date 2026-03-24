import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

# =========================
# 1. 載入模型
# =========================
MODEL_PATH = "tabnet_loan_default.zip"

print("Loading TabNet model...")
model = TabNetClassifier()
model.load_model(MODEL_PATH)
print("Model loaded successfully.\n")

# =========================
# 2. 定義特徵欄位（⚠️順序必須和訓練時一模一樣）
# =========================
FEATURES = [
    ("person_age", "申請人年齡（years）"),
    ("person_gender", "性別（encoding 後數值）"),
    ("person_education", "教育程度（encoding 後數值）"),
    ("person_income", "年收入（annual income）"),
    ("person_emp_exp", "就業年資（years）"),
    ("person_home_ownership", "房屋持有狀況（encoding 後數值）"),
    ("loan_amnt", "貸款金額（loan amount）"),
    ("loan_intent", "貸款用途（encoding 後數值）"),
    ("loan_int_rate", "貸款利率（interest rate）"),
    ("loan_percent_income", "貸款金額佔收入比例"),
    ("cb_person_cred_hist_length", "信用歷史長度（years）"),
    ("credit_score", "信用評分（credit score）"),
]

# =========================
# 3. 互動式輸入
# =========================
inputs = []

print("請依序輸入貸款申請資料：\n")

for idx, (col, desc) in enumerate(FEATURES, start=1):
    while True:
        try:
            value = float(input(f"[{idx}/{len(FEATURES)}] {col} - {desc}: "))
            inputs.append(value)
            print(f"✔ 已輸入 {col} = {value}\n")
            break
        except ValueError:
            print("❌ 請輸入數值（numeric value）\n")

# =========================
# 4. 組成模型輸入格式
# =========================
X = np.array(inputs, dtype=np.float32).reshape(1, -1)

# =========================
# 5. 預測
# =========================
proba = model.predict_proba(X)[0, 1]

print("===================================")
print("📊 預測結果 Prediction Result")
print("===================================")
print(f"違約機率（Default Probability）: {proba:.4f}")
print(f"未違約機率（Non-default）      : {1 - proba:.4f}")

# =========================
# 6. 風控門檻示範（可自行調整）
# =========================
THRESHOLD = 0.6
decision = "拒貸（High Risk）" if proba >= THRESHOLD else "核准（Low Risk）"

print("-----------------------------------")
print(f"風控門檻 Threshold : {THRESHOLD}")
print(f"系統建議 Decision : {decision}")
print("===================================")
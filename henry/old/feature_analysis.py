import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

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
    df["credit_score_bucket"] = df["credit_score_bucket"].cat.codes

    df["age_bucket"] = pd.cut(
        df["person_age"],
        bins=[0, 30, 50, 100],
        labels=["young", "mid", "senior"]
    )
    df["age_bucket"] = df["age_bucket"].cat.codes

    return df

# =============================
# 1. 讀取資料
# =============================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "loan_train_36000.csv"
TARGET = "loan_status"

df = pd.read_csv(DATA_PATH)
df = add_engineered_features(df)

# =============================
# 2. 定義「原始特徵」與「工程特徵」
# =============================
original_features = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

engineered_features = [c for c in df.columns
                        if c not in original_features + [TARGET]]

print("原始特徵:", original_features)
print("工程特徵（你新增的）:", engineered_features)

X = df[original_features + engineered_features]
y = df[TARGET]

# =============================
# 3. 單變數分析
# =============================
print("\n===== 單變數分析（Original Features） =====")
for col in original_features:
    mean_0 = df[df[TARGET] == 0][col].mean()
    mean_1 = df[df[TARGET] == 1][col].mean()
    print(f"{col}: non-default={mean_0:.2f}, default={mean_1:.2f}")

print("\n===== 單變數分析（Engineered Features） =====")
for col in engineered_features:
    mean_0 = df[df[TARGET] == 0][col].mean()
    mean_1 = df[df[TARGET] == 1][col].mean()
    print(f"{col}: non-default={mean_0:.4f}, default={mean_1:.4f}")

# =============================
# 4. Correlation
# =============================
print("\n===== Correlation with loan_status =====")
corr = df[X.columns.tolist() + [TARGET]].corr()[TARGET].sort_values(ascending=False)
print(corr)

# =============================
# 5. Logistic Regression（可解釋）
# =============================
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )),
])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_val)

print("\nValidation F1-score:", f1_score(y_val, y_pred))

coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": pipe.named_steps["model"].coef_[0]
}).sort_values(by="coefficient", ascending=False)

print("\n===== Logistic Regression Coefficients =====")
print(coef_df)

# =============================
# 6. Permutation Importance
# =============================
perm = permutation_importance(
    pipe,
    X_val,
    y_val,
    scoring="f1",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "feature": X.columns,
    "importance": perm.importances_mean
}).sort_values(by="importance", ascending=False)

print("\n===== Permutation Importance =====")
print(perm_df)

# =============================
# 7. 存檔（報告用）
# =============================
coef_df.to_csv("logistic_coefficients_all_features.csv", index=False)
perm_df.to_csv("permutation_importance_all_features.csv", index=False)

print("\n分析完成（含 6 個工程特徵）")

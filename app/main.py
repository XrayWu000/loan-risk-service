from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os

app = FastAPI()

# =========================
# 載入模型
# =========================
from config import MODEL_PATH, FEATURE_FILE
model = joblib.load(MODEL_PATH)

# =========================
# 載入 feature 順序（如果有）
# =========================

try:
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        FEATURE_COLUMNS = json.load(f)
except:
    FEATURE_COLUMNS = None

# =========================
# API Input Schema
# =========================
class LoanInput(BaseModel):
    person_age: float
    person_gender: str
    person_income: float
    person_emp_exp: int
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: int


# =========================
# Feature Engineering
# =========================
def preprocess(df):
    gender_map = {"男": 1, "女": 0}
    home_map = {"租賃": 0, "自有（尚有貸款）": 1, "自有（無貸款）": 2}
    intent_map = {"個人周轉": 0, "醫療照護": 1, "創業周轉": 2, "教育進修": 3}

    df["person_gender"] = df["person_gender"].map(gender_map)
    df["person_home_ownership"] = df["person_home_ownership"].map(home_map)
    df["loan_intent"] = df["loan_intent"].map(intent_map)

    df["log_income"] = np.log1p(df["person_income"])
    df["interest_pressure"] = (
        df["loan_int_rate"] * df["loan_percent_income"]
    )

    # 模型使用的欄位（你原本那 10 個）
    columns = [
        "person_home_ownership",
        "loan_intent",
        "loan_int_rate",
        "cb_person_cred_hist_length",
        "interest_pressure",
        "person_emp_exp",
        "person_age",
        "person_gender",
        "loan_amnt",
        "log_income"
    ]

    if FEATURE_COLUMNS:
        columns = FEATURE_COLUMNS

    df = df[columns]

    # category 型別
    for col in ["person_home_ownership", "loan_intent", "person_gender"]:
        df[col] = df[col].astype("category")

    return df


# =========================
# API Routes
# =========================
@app.get("/")
def root():
    return {"message": "Loan Risk API Running"}


@app.post("/predict")
def predict(data: LoanInput):
    df = pd.DataFrame([data.dict()])
    df = preprocess(df)

    prob = model.predict_proba(df)[0][1]

    return {
        "probability": float(prob)
    }
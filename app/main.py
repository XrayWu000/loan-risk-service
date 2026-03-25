from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import json
import os

from config import MODEL_PATH, FEATURE_FILE

app = FastAPI()

# =========================
# 載入模型（加錯誤處理）
# =========================
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"模型載入失敗: {str(e)}")

# =========================
# 載入 feature 順序
# =========================
try:
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        FEATURE_COLUMNS = json.load(f)
except:
    FEATURE_COLUMNS = None

# =========================
# API Input Schema（加限制）
# =========================
class LoanInput(BaseModel):
    person_age: float = Field(..., ge=20, le=80)
    person_gender: str
    person_income: float = Field(..., gt=0)
    person_emp_exp: int = Field(..., ge=0)
    person_home_ownership: str
    loan_amnt: float = Field(..., gt=0)
    loan_intent: str
    loan_int_rate: float = Field(..., gt=0)
    loan_percent_income: float = Field(..., gt=0)
    cb_person_cred_hist_length: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=200, le=850)

# =========================
# Feature Engineering
# =========================
def preprocess(df):
    gender_map = {"男": 1, "女": 0}
    home_map = {"租賃": 0, "自有（尚有貸款）": 1, "自有（無貸款）": 2}
    intent_map = {"個人周轉": 0, "醫療照護": 1, "創業周轉": 2, "教育進修": 3}

    # ❗ mapping 檢查（避免 None）
    if df["person_gender"].iloc[0] not in gender_map:
        raise ValueError("無效的性別")

    if df["person_home_ownership"].iloc[0] not in home_map:
        raise ValueError("無效的居住狀況")

    if df["loan_intent"].iloc[0] not in intent_map:
        raise ValueError("無效的貸款用途")

    df["person_gender"] = df["person_gender"].map(gender_map)
    df["person_home_ownership"] = df["person_home_ownership"].map(home_map)
    df["loan_intent"] = df["loan_intent"].map(intent_map)

    df["log_income"] = np.log1p(df["person_income"])
    df["interest_pressure"] = df["loan_int_rate"] * df["loan_percent_income"]

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

    for col in ["person_home_ownership", "loan_intent", "person_gender"]:
        df[col] = df[col].astype("category")

    return df

# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {"message": "Loan Risk API Running"}

@app.post("/predict")
def predict(data: LoanInput):
    try:
        df = pd.DataFrame([data.dict()])
        df = preprocess(df)

        prob = model.predict_proba(df)[0][1]

        return {
            "status": "success",
            "probability": float(prob)
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推論錯誤: {str(e)}")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
from datetime import datetime

# 🔥 pipeline（統一用這個）
from pipeline.feature_engineering import prepare_model_input

# 🔧 config
from config import MODEL_PATH, FEATURE_FILE, CSV_FILE

app = FastAPI()

# =========================
# 載入模型
# =========================
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"模型載入失敗: {str(e)}")

# =========================
# API Input Schema
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
# CSV Logging
# =========================
def save_to_csv(input_data: dict, probability: float):
    row = input_data.copy()
    row["probability"] = probability
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame([row])

    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    if not os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(CSV_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")

# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {"message": "Loan Risk API Running"}

@app.post("/predict")
def predict(data: LoanInput):
    try:
        input_dict = data.dict()

        # =========================
        # 1️⃣ 輸入驗證（保留你原本邏輯）
        # =========================
        valid_gender = {"男", "女"}
        valid_home = {"租賃", "自有（尚有貸款）", "自有（無貸款）"}
        valid_intent = {"個人周轉", "醫療照護", "創業周轉", "教育進修"}

        if input_dict["person_gender"] not in valid_gender:
            raise ValueError("無效的性別")

        if input_dict["person_home_ownership"] not in valid_home:
            raise ValueError("無效的居住狀況")

        if input_dict["loan_intent"] not in valid_intent:
            raise ValueError("無效的貸款用途")

        # =========================
        # 2️⃣ DataFrame
        # =========================
        df = pd.DataFrame([input_dict])

        # =========================
        # 3️⃣ Pipeline（🔥統一）
        # =========================
        df = prepare_model_input(df)

        # =========================
        # 4️⃣ 預測
        # =========================
        prob = model.predict_proba(df)[0][1]

        # =========================
        # 5️⃣ Logging（成功才存）
        # =========================
        save_to_csv(input_dict, float(prob))

        return {
            "status": "success",
            "probability": float(prob)
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推論錯誤: {str(e)}")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

from pipeline.feature_engineering import prepare_model_input
from frontend.services.logger import save_to_csv, update_loan_status
from config import (
    MODEL_PATH,
    GENDER_OPTIONS,
    EDUCATION_OPTIONS,
    HOME_OWNERSHIP_OPTIONS,
    LOAN_INTENT_OPTIONS,
)

app = FastAPI()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"模型載入失敗: {str(e)}")


class LoanInput(BaseModel):
    person_age: float = Field(..., ge=20, le=80)
    person_gender: str
    person_education: str
    person_income: float = Field(..., gt=0)
    person_emp_exp: int = Field(..., ge=0)
    person_home_ownership: str
    loan_amnt: float = Field(..., gt=0)
    loan_intent: str
    loan_int_rate: float = Field(..., gt=0)
    loan_percent_income: float = Field(..., gt=0)
    cb_person_cred_hist_length: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=200, le=850)


class LoanLabelInput(BaseModel):
    case_id: str
    loan_status: int = Field(..., ge=0, le=1)


def get_decision_label(probability: float) -> str:
    if probability >= 0.5:
        return "拒絕貸款申請 (高風險)"
    if probability >= 0.2:
        return "人工審核後再評估 (中風險)"
    return "核准貸款申請 (低風險)"


@app.get("/")
def root():
    return {"message": "Loan Risk API Running"}


@app.post("/predict")
def predict(data: LoanInput):
    try:
        input_dict = data.dict()

        valid_gender = set(GENDER_OPTIONS)
        valid_education = set(EDUCATION_OPTIONS)
        valid_home = set(HOME_OWNERSHIP_OPTIONS)
        valid_intent = set(LOAN_INTENT_OPTIONS)

        if input_dict["person_gender"] not in valid_gender:
            raise ValueError("無效的性別")

        if input_dict["person_education"] not in valid_education:
            raise ValueError("無效的教育程度")

        if input_dict["person_home_ownership"] not in valid_home:
            raise ValueError("無效的居住狀況")

        if input_dict["loan_intent"] not in valid_intent:
            raise ValueError("無效的貸款用途")

        df = pd.DataFrame([input_dict])
        df = prepare_model_input(df)
        prob = model.predict_proba(df)[0][1]

        decision = get_decision_label(float(prob))
        _, case_id = save_to_csv(
            input_dict["person_age"],
            input_dict["person_gender"],
            input_dict["person_education"],
            input_dict["person_income"],
            input_dict["person_emp_exp"],
            input_dict["person_home_ownership"],
            input_dict["loan_amnt"],
            input_dict["loan_intent"],
            input_dict["loan_int_rate"],
            input_dict["loan_percent_income"],
            input_dict["cb_person_cred_hist_length"],
            input_dict["credit_score"],
            float(prob),
            decision,
        )

        return {
            "status": "success",
            "probability": float(prob),
            "case_id": case_id,
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推論錯誤: {str(e)}")


@app.post("/label")
def label(data: LoanLabelInput):
    try:
        return update_loan_status(
            case_id=data.case_id,
            loan_status=data.loan_status,
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"標記錯誤: {str(e)}")

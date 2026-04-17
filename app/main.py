from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.services.gcp_uploader import (
    get_gcp_runtime_status,
    start_gcp_upload_scheduler,
    stop_gcp_upload_scheduler,
    upload_pending_logs_to_gcp,
)
from app.services.notification_service import send_fallback_model_alert
from app.services.request_log_service import save_to_csv, update_loan_status
from config.path_config import MODEL_PATH
from config.ui_config import (
    EDUCATION_OPTIONS,
    GENDER_OPTIONS,
    HOME_OWNERSHIP_OPTIONS,
    LOAN_INTENT_OPTIONS,
)
from pipeline.feature_engineering import prepare_model_input

FALLBACK_PROBABILITY = 0.35


@asynccontextmanager
async def lifespan(_: FastAPI):
    start_gcp_upload_scheduler()
    yield
    stop_gcp_upload_scheduler()


app = FastAPI(lifespan=lifespan)


class FallbackModel:
    def predict(self, x):
        return [0] * len(x)

    def predict_proba(self, x):
        return [[1 - FALLBACK_PROBABILITY, FALLBACK_PROBABILITY] for _ in range(len(x))]


try:
    model = joblib.load(MODEL_PATH)
    model_load_error = None
    fallback_alert_status = "not_needed"
except Exception as exc:
    print(f"模型載入失敗，已切換到 fallback model: {exc}")
    model = FallbackModel()
    model_load_error = str(exc)
    fallback_alert_status = send_fallback_model_alert(str(exc))


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
    return {
        "message": "Loan Risk API Running",
        "model_status": "fallback" if model_load_error else "loaded",
        "fallback_alert_status": fallback_alert_status,
        "gcp_sync": get_gcp_runtime_status(),
    }


@app.post("/predict")
def predict(data: LoanInput):
    try:
        input_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()

        if input_dict["person_gender"] not in set(GENDER_OPTIONS):
            raise ValueError("無效的性別選項。")

        if input_dict["person_education"] not in set(EDUCATION_OPTIONS):
            raise ValueError("無效的教育程度選項。")

        if input_dict["person_home_ownership"] not in set(HOME_OWNERSHIP_OPTIONS):
            raise ValueError("無效的居住狀態選項。")

        if input_dict["loan_intent"] not in set(LOAN_INTENT_OPTIONS):
            raise ValueError("無效的貸款用途選項。")

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
            "model_status": "fallback" if model_load_error else "loaded",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"模型推論失敗: {exc}") from exc


@app.post("/label")
def label(data: LoanLabelInput):
    try:
        return update_loan_status(
            case_id=data.case_id,
            loan_status=data.loan_status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"標記更新失敗: {exc}") from exc


@app.post("/admin/gcp/upload")
def trigger_gcp_upload():
    try:
        return upload_pending_logs_to_gcp()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GCP 上傳失敗: {exc}") from exc

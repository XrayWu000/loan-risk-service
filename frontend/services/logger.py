import pandas as pd
import numpy as np
import os
import shutil
import threading
from datetime import datetime

# ===== 全域 lock =====
csv_lock = threading.Lock()
pending_retry_data = None

from config import CSV_FILE


LOG_COLUMNS = [
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
    "log_income",
    "interest_pressure",
    "loan_status",
    "prediction_probability",
    "prediction_label",
    "case_id",
    "timestamp",
    "is_processed",
]


def _backup_invalid_csv():
    base, ext = os.path.splitext(CSV_FILE)
    backup_path = f"{base}.corrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    shutil.move(CSV_FILE, backup_path)
    return backup_path


def _prepare_csv_for_append():
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        return True

    try:
        existing_columns = list(pd.read_csv(CSV_FILE, nrows=0, encoding="utf-8-sig").columns)
    except Exception:
        _backup_invalid_csv()
        return True

    if existing_columns != LOG_COLUMNS:
        _backup_invalid_csv()
        return True

    return False


def save_to_csv(age, gender, edu, income, emp, home, amnt, intent,
                rate_val, percent_val, cred_len, score, probability, decision):

    current_time = datetime.now()

    case_id = f"CASE_{current_time.strftime('%Y%m%d')}_{np.random.randint(10000, 99999)}"

    log_income = np.log1p(income)
    interest_pressure = rate_val * percent_val

    row = {
        # ===== feature（完全對齊 train）=====
        "person_age": age,
        "person_gender": gender,
        "person_education": edu,
        "person_income": income,
        "person_emp_exp": emp,
        "person_home_ownership": home,
        "loan_amnt": amnt,
        "loan_intent": intent,
        "loan_int_rate": rate_val,
        "loan_percent_income": percent_val,
        "cb_person_cred_hist_length": cred_len,
        "credit_score": score,

        # ===== engineered =====
        "log_income": log_income,
        "interest_pressure": interest_pressure,

        # ===== label（未來用）=====
        "loan_status": None,

        # ===== prediction =====
        "prediction_probability": float(probability),
        "prediction_label": decision,

        # ===== system =====
        "case_id": case_id,
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "is_processed": 0
    }

    df = pd.DataFrame([row], columns=LOG_COLUMNS)

    with csv_lock:
        header = _prepare_csv_for_append()
        df.to_csv(
            CSV_FILE,
            mode='a',
            index=False,
            header=header,
            encoding="utf-8-sig"
        )

    return True, case_id


def update_loan_status(case_id, loan_status):
    with csv_lock:
        if not os.path.exists(CSV_FILE):
            raise ValueError("CSV 檔案不存在")

        df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")

        if "case_id" not in df.columns:
            raise ValueError("CSV 缺少 case_id 欄位")

        if "loan_status" not in df.columns:
            raise ValueError("CSV 缺少 loan_status 欄位")

        mask = df["case_id"] == case_id
        if not mask.any():
            raise ValueError("找不到該 case_id")

        if df.loc[mask, "loan_status"].notna().any():
            raise ValueError("該資料已經標記過")

        df.loc[mask, "loan_status"] = loan_status

        df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")

    return {
        "status": "success",
        "case_id": case_id,
        "loan_status": loan_status,
    }


def retry_write():
    global pending_retry_data

    if pending_retry_data is None:
        return True

    try:
        with csv_lock:
            header = not os.path.exists(CSV_FILE)
            pending_retry_data.to_csv(
                CSV_FILE,
                mode='a',
                index=False,
                header=header,
                encoding="utf_8_sig"
            )

        pending_retry_data = None
        return True

    except Exception:
        return False

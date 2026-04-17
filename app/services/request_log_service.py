import os
import shutil
import threading
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from config.path_config import CSV_FILE

csv_lock = threading.Lock()
pending_retry_data = None

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


def _backup_invalid_csv() -> str:
    base, ext = os.path.splitext(CSV_FILE)
    backup_path = f"{base}.corrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    shutil.move(CSV_FILE, backup_path)
    return backup_path


def _prepare_csv_for_append() -> bool:
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


def _empty_log_df() -> pd.DataFrame:
    return pd.DataFrame(columns=LOG_COLUMNS)


def _normalize_log_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    for column in LOG_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    normalized = normalized[LOG_COLUMNS]
    normalized["is_processed"] = normalized["is_processed"].fillna(0)
    return normalized


def _read_log_df_unlocked() -> pd.DataFrame:
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        return _empty_log_df()

    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    return _normalize_log_df(df)


def _write_log_df_unlocked(df: pd.DataFrame) -> None:
    normalized = _normalize_log_df(df)
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    normalized.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")


def save_to_csv(
    age,
    gender,
    edu,
    income,
    emp,
    home,
    amnt,
    intent,
    rate_val,
    percent_val,
    cred_len,
    score,
    probability,
    decision,
):
    current_time = datetime.now()
    case_id = f"CASE_{current_time.strftime('%Y%m%d')}_{np.random.randint(10000, 99999)}"

    row = {
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
        "log_income": np.log1p(income),
        "interest_pressure": rate_val * percent_val,
        "loan_status": None,
        "prediction_probability": float(probability),
        "prediction_label": decision,
        "case_id": case_id,
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "is_processed": 0,
    }

    df = pd.DataFrame([row], columns=LOG_COLUMNS)

    with csv_lock:
        header = _prepare_csv_for_append()
        df.to_csv(
            CSV_FILE,
            mode="a",
            index=False,
            header=header,
            encoding="utf-8-sig",
        )

    return True, case_id


def update_loan_status(case_id, loan_status):
    with csv_lock:
        if not os.path.exists(CSV_FILE):
            raise ValueError("CSV file does not exist.")

        df = _read_log_df_unlocked()

        if "case_id" not in df.columns:
            raise ValueError("CSV is missing the case_id column.")

        if "loan_status" not in df.columns:
            raise ValueError("CSV is missing the loan_status column.")

        mask = df["case_id"] == case_id
        if not mask.any():
            raise ValueError("Case ID not found.")

        if df.loc[mask, "loan_status"].notna().any():
            raise ValueError("Loan status has already been labeled.")

        df.loc[mask, "loan_status"] = loan_status
        df.loc[mask, "is_processed"] = 0
        _write_log_df_unlocked(df)

    return {
        "status": "success",
        "case_id": case_id,
        "loan_status": loan_status,
    }


def get_pending_upload_rows() -> pd.DataFrame:
    with csv_lock:
        df = _read_log_df_unlocked()

    if df.empty:
        return _empty_log_df()

    pending_mask = df["is_processed"].fillna(0).astype(int) != 1
    return df.loc[pending_mask].copy()


def mark_cases_as_processed(case_ids: Iterable[str]) -> int:
    case_id_list = [case_id for case_id in case_ids if case_id]
    if not case_id_list:
        return 0

    with csv_lock:
        df = _read_log_df_unlocked()
        if df.empty:
            return 0

        mask = df["case_id"].isin(case_id_list)
        updated_rows = int(mask.sum())
        if updated_rows == 0:
            return 0

        df.loc[mask, "is_processed"] = 1
        _write_log_df_unlocked(df)

    return updated_rows


def prune_processed_rows(retention_days: int) -> int:
    if retention_days <= 0:
        return 0

    cutoff = datetime.now() - timedelta(days=retention_days)

    with csv_lock:
        df = _read_log_df_unlocked()
        if df.empty:
            return 0

        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        processed = df["is_processed"].fillna(0).astype(int) == 1
        labeled = df["loan_status"].notna()
        removable_mask = processed & labeled & timestamps.lt(cutoff)
        removed_rows = int(removable_mask.sum())

        if removed_rows == 0:
            return 0

        remaining = df.loc[~removable_mask].copy()
        _write_log_df_unlocked(remaining if not remaining.empty else _empty_log_df())

    return removed_rows


def retry_write():
    global pending_retry_data

    if pending_retry_data is None:
        return True

    try:
        with csv_lock:
            header = _prepare_csv_for_append()
            pending_retry_data.to_csv(
                CSV_FILE,
                mode="a",
                index=False,
                header=header,
                encoding="utf-8-sig",
            )

        pending_retry_data = None
        return True
    except Exception:
        return False

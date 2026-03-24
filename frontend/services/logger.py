import pandas as pd
import numpy as np
import os
import threading
from datetime import datetime

# ===== 全域 lock =====
csv_lock = threading.Lock()
pending_retry_data = None

CSV_FILE = "loan_requests_full.csv"


def save_to_csv(age, gender, edu, income, emp, home, amnt, intent,
                rate_val, percent_val, cred_len, score, probability, status):

    current_time = datetime.now()
    case_id = f"CASE_{current_time.strftime('%Y%m%d')}_{np.random.randint(10000, 99999)}"

    save_data = {
        "申請時間": [current_time.strftime("%Y-%m-%d %H:%M:%S")],
        "案件編號": [case_id],
        "申請人年齡": [age],
        "性別": [gender],
        "教育程度": [edu],
        "年收入": [income],
        "工作年資": [emp],
        "居住狀況": [home],
        "貸款金額": [amnt],
        "貸款用途": [intent],
        "擬定利率": [rate_val],
        "收支負債比": [percent_val],
        "信用歷史年資": [cred_len],
        "信用分數": [score],
        "AI預測結果": [status],
        "違約機率": [f"{probability:.4f}"]
    }

    df = pd.DataFrame(save_data)

    global pending_retry_data

    try:
        with csv_lock:
            header = not os.path.exists(CSV_FILE)
            df.to_csv(
                CSV_FILE,
                mode='a',
                index=False,
                header=header,
                encoding="utf_8_sig"
            )

        pending_retry_data = None
        return True, case_id

    except Exception:
        pending_retry_data = df.copy()
        return False, case_id


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
import pandas as pd
import os

from config.path_config import CSV_FILE

def view_latest_logs():
    try:
        if not os.path.exists(CSV_FILE):
            return pd.DataFrame({"系統訊息": ["尚無任何申請紀錄"]})

        if os.path.getsize(CSV_FILE) == 0:
            return pd.DataFrame({"系統訊息": ["CSV 為空（可能剛被清空）"]})

        df = pd.read_csv(
            CSV_FILE,
            encoding="utf_8_sig",
            on_bad_lines="skip"
        )

        if df.empty:
            return pd.DataFrame({"系統訊息": ["目前沒有資料"]})

        return df.tail(10)

    except PermissionError:
        return pd.DataFrame({"系統錯誤": ["❌ 檔案被 Excel 開啟"]})

    except Exception as e:
        return pd.DataFrame({"系統錯誤": [str(e)]})


def get_log_file():
    try:
        if not os.path.exists(CSV_FILE):
            return None

        if os.path.getsize(CSV_FILE) == 0:
            return None

        # Gradio 6 expects a string filepath for File outputs.
        return str(CSV_FILE)

    except Exception:
        return None

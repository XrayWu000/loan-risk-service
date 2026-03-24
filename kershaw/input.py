import gradio as gr
import pandas as pd
import joblib
import numpy as np
import os
import re
import json
import threading
from datetime import datetime
from google.cloud import storage
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo  # Python 3.9+ 內建時區
import shutil

# ==========================================
# ✅ NEW: 特徵檔與 CSV lock（只加這兩塊）
# ==========================================
FEATURE_FILE = "lgbm_best_model_features.json"
csv_lock = threading.Lock()
pending_retry_data = None

# 讀取特徵順序（如果讀不到就 fallback 用原本 columns）
try:
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        FEATURE_COLUMNS = json.load(f)
    print(f"✅ 特徵順序載入成功: {FEATURE_FILE}")
except Exception as e:
    print(f"⚠️ 特徵順序載入失敗，將使用程式內建 columns: {e}")
    FEATURE_COLUMNS = None

GCS_BUCKET_NAME = "tibame-loan-logs"
GCS_KEY_FILE = "gad258-05-report-storage-bc3f569fb460.json"
GCS_OBJECT_PREFIX = "daily_exports"

# ==========================================
#  對接新版模型
# 1.系統初始化與載入模型
# ==========================================
MODEL_FILE = 'lgbm_best_model.zip'  # 換新版
CSV_FILE = "loan_requests_full.csv"

try:
    model = joblib.load(MODEL_FILE)
    print(f"✅ 模型載入成功: {MODEL_FILE}")
except Exception as e:
    print(f"❌ 模型載入失敗，請確認檔案是否存在: {e}")

    # 建立一個假模型避免程式直接崩潰
    class FakeModel:
        def predict(self, x):
            return [0] * len(x)

        def predict_proba(self, x):
            return [[0.65, 0.35]] * len(x)

    model = FakeModel()


# ==========================================
# 2. 前端輔助函數
# ==========================================
def validate_precision(val):
    if not val: return ""
    val = re.sub(r"[^\d.]", "", str(val))
    if val.count('.') > 1:
        parts = val.split('.')
        val = f"{parts[0]}.{''.join(parts[1:])}"
    if "." in val:
        parts = val.split('.')
        val = f"{parts[0]}.{parts[1][:2]}"
    try:
        num_val = float(val)
        if num_val > 99.99: return "99.99"
        return val
    except ValueError:
        return ""

# ==========================================
# 3. 核心邏輯
# ==========================================
def process_loan_request(age, gender, edu, income, emp, home, amnt, intent, rate, percent, cred_len, score):
    # --- 前端防呆機制 ---
    try:
        rate_val = float(rate) if rate else 0.0
        percent_val = float(percent) if percent else 0.0
        validations = [
            (age is not None and 20 <= age <= 80, "年齡須在 20-80 歲之間"),
            (income is not None and income > 0, "年收入須大於 0"),
            (emp is not None and emp >= 0, "工作年資不可為負"),
            (amnt is not None and amnt >= 0, "貸款金額不可為負"),
            (0.01 < rate_val <= 99.99, "擬定利率須在 0.01 到 99.99 之間"),
            (0.01 < percent_val <= 99.99, "收支負擔率須在 0.01 到 99.99 之間"),
            (cred_len is not None and cred_len >= 0, "信用年資不可為負"),
            (score is not None and 200 <= score <= 800, "信用分數須在 200-800 之間")
        ]
        for condition, msg in validations:
            if not condition: return f"❌ 系統拒絕審核：{msg}"
    except Exception:
        return "❌ 輸入包含非法字元或格式錯誤。"

    # --- A. [Mapping] 類別轉換 ---
    gender_map = {"男": 1, "女": 0}
    home_map = {"租賃": 0, "自有（尚有貸款）": 1, "自有（無貸款）": 2, "其他": 3}
    intent_map = {"個人周轉": 0, "醫療照護": 1, "創業周轉": 2, "教育進修": 3, "債務整合": 4, "家庭裝修": 5}
    # 💡 雙重保險：確保前端不管顯示「大學」還是「學士」，後端都能吃得到
    edu_map = {"高中/職": 0, "副學士(專科)": 1, "學士": 2, "碩士": 3, "博士": 4}

    gender_num = gender_map.get(gender, 0)
    home_num = home_map.get(home, 0)
    intent_num = intent_map.get(intent, 0)

    # --- B. 特徵工程 ---
    try:
        log_income = np.log1p(income)
        interest_pressure = rate_val * percent_val
    except:
        return "❌ 數值計算錯誤，請檢查輸入是否為數字"

    # --- C. [Prediction] 配合模型定義的 10 個特徵與順序 ---
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

    # ✅ NEW: 如果 json 有讀到，就用 json 的順序覆蓋 columns（其餘不改）
    if FEATURE_COLUMNS:
        columns = FEATURE_COLUMNS

    # 依照順序打包
    X_input = pd.DataFrame([[
        home_num,
        intent_num,
        rate_val,
        cred_len,
        interest_pressure,
        emp,
        age,
        gender_num,
        amnt,
        log_income
    ]], columns=columns)

    # 強制轉為 category 型別
    categorical_cols = ["person_home_ownership", "loan_intent", "person_gender"]
    for col in categorical_cols:
        X_input[col] = X_input[col].astype("category")

    try:
        probability = model.predict_proba(X_input)[0][1]
    except Exception as e:
        return f"❌ 預測失敗: {str(e)}"

    # --- 設定高中低風險門檻 ---
    if probability >= 0.50:
        status_emoji = "🚨 建議拒絕 (高風險)"
        csv_status = "建議拒絕 (高風險)"
    elif probability >= 0.20:
        status_emoji = "⚠️ 需人工確認 (中風險)"
        csv_status = "需人工確認 (中風險)"
    else:
        status_emoji = "✅ 建議核貸 (低風險)"
        csv_status = "建議核貸 (低風險)"

    # --- D. [Logging] 完整存檔 ---
    current_time = datetime.now()
    date_str = current_time.strftime("%Y%m%d")
    case_id = f"CASE_{date_str}_{np.random.randint(10000, 99999)}"

    save_data = {
        "申請時間": [current_time.strftime("%Y-%m-%d %H:%M:%S")],
        "案件編號": [case_id],
        "申請人年齡": [age], "性別": [gender], "教育程度": [edu],
        "年收入": [income], "工作年資": [emp], "居住狀況": [home],
        "貸款金額": [amnt], "貸款用途": [intent], "擬定利率": [rate_val],
        "收支負債比": [percent_val], "信用歷史年資": [cred_len], "信用分數": [score],
        "對數收入(計算)": [f"{log_income:.4f}"],
        "利息壓力(計算)": [f"{interest_pressure:.4f}"],
        "AI預測結果": [csv_status],
        "違約機率": [f"{probability:.4f}"]
    }

    df_save = pd.DataFrame(save_data)

    global pending_retry_data

    try:
        with csv_lock:
            header = not os.path.exists(CSV_FILE)
            df_save.to_csv(
                CSV_FILE,
                mode='a',
                index=False,
                header=header,
                encoding="utf_8_sig"
            )
        pending_retry_data = None
        save_status = "✅ 資料已存檔"

    except Exception:
        pending_retry_data = df_save.copy()
        return "WRITE_FAIL"

    # --- E. 回傳給前端的訊息 ---
    return f"【審核完成】\n結果：{status_emoji}\n違約機率：{probability:.2%}\n(案件編號: {case_id})\n{save_status}"

def submit_handler(*args):
    result = process_loan_request(*args)

    if result == "WRITE_FAIL":
        return (
            "",  # 清空 output_text
            gr.update(visible=True)  # 打開 modal
        )

    return (
        result,
        gr.update(visible=False)
    )

# ==========================================
# 4. 後台管理函數
# ==========================================
def view_latest_logs():
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE, encoding='utf_8_sig', on_bad_lines='skip')
            return df.tail(10)
        except PermissionError:
            err_msg = "❌ 無法讀取：檔案正被 Excel 開啟中，請先關閉 Excel 再刷新。"
            return pd.DataFrame({"系統錯誤": [err_msg]})
        except Exception as e:
            return pd.DataFrame({"系統錯誤": [f"❌ 讀取發生未知錯誤: {str(e)}"]})
    return pd.DataFrame({"系統訊息": ["尚無任何申請紀錄"]})

def get_log_file():
    if os.path.exists(CSV_FILE):
        return CSV_FILE
    return None

def retry_write_csv():
    global pending_retry_data

    if pending_retry_data is None:
        return gr.update(visible=False)

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

        # ✅ 成功就關閉 modal
        return gr.update(visible=False)

    except Exception:
        # ❌ 失敗就保持開啟
        return gr.update(visible=True)

# ==========================================
# 5. 自定義 CSS
# ==========================================
custom_css = """
/* 1. 全域背景與字體 */
.gradio-container {
    background-color: #f0f8ff;
    font-family: 'Microsoft JhengHei', 'PingFang TC', sans-serif;
}

/* 2. 主標題區 */
.main-header h1 {
    font-size: 70px !important;  # 馬上有錢 主標題 大小調整處
    font-weight: 800;
    margin-bottom: 10px;  # 主副標題上下間距 調整處
}
.main-header p {
    font-size: 35px;  # 馬上有錢標題下的小字 大小調整處
    color: #555;
}
.main-header { text-align: center; color: #0056b3; padding: 20px; }

/* 3. 題目（Label）字體大小與樣式 */
span[data-testid="block-info"] {
    font-size: 25px !important;    /* 👈 調整題目的字體大小 */
    font-weight: bold !important;  /* 讓題目變粗體 */
    color: #333 !important;        /* 題目顏色 */
}

/* 4. 輸入數值與文字的排版（靠左/置中/靠右） */
input, select, textarea {
    font-size: 20px !important;    /* 👈 調整輸入框內數值的大小 */
    text-align: center !important; /* 👈 選項：left(靠左), center(置中), right(靠右) */
    color: #0056b3 !important;     /* 數值顏色改為深藍色，更有銀行感 */
    font-weight: 500;
}

/* 5. 插畫區與按鈕樣式保持優化 */
.illustration-row {
    display: flex; justify-content: space-around; padding: 20px;
    background: white; border-radius: 20px; margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.submit-btn {
    background-color: #007bff !important; color: white !important;
    border-radius: 30px !important; font-size: 24px !important; /* 按鈕字體也加大 */
    font-weight: bold !important;
}

/* 定位容器，確保「元」字能相對於輸入框排列 */
#unit-yuan-1, #unit-yuan-2 {
    position: relative;
}

/* 在輸入框容器後方加入「元」字 */
#unit-yuan-1::after, #unit-yuan-2::after {
    content: "元";
    position: absolute;
    right: 25px;         /* 調整與右邊邊界的距離 */
    bottom: 23px;        /* 調整上下高度，使其對齊數值中心 */
    font-size: 20px;     /* 字體大小 */
    color: #0056b3;      /* 顏色與數值一致 */
    font-weight: bold;
    pointer-events: none; /* 確保不會擋到滑鼠點擊 */
}

/* 為了避免文字重疊，讓輸入框右側留一點空間 */
#unit-yuan-1 input, #unit-yuan-2 input {
    padding-right: 0px !important;  # 數值越小就往右 ; 數值越大就往左
}
.submit-btn:hover { transform: scale(1.05); transition: 0.3s; box-shadow: 0 5px 15px rgba(0,123,255,0.4); }

#modal_container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: rgba(0,0,0,0.4);
    justify-content: center;
    align-items: center;
    z-index: 10000;
}

#modal_container[style*="display: block"] {
    display: flex !important;
}

#modal_container > * {
    background: white;
    padding: 30px;
    border-radius: 15px;
    width: 400px;
    text-align: center;
}

/* ==========================================
   🏦 銀行藍 3D 立體按鈕
========================================== */

.bank-retry-btn {
    position: relative;
    background: linear-gradient(145deg, #0d6efd, #003f9e) !important;
    color: white !important;
    border-radius: 50px !important;
    font-size: 20px !important;
    font-weight: bold !important;
    padding: 14px 32px !important;
    border: none !important;

    /* 3D 陰影 */
    box-shadow:
        0 6px 0 #002e75,
        0 12px 20px rgba(0, 63, 158, 0.4);

    transition: all 0.15s ease-in-out;
}

/* 上方高光 */
.bank-retry-btn::before {
    content: "";
    position: absolute;
    top: 4px;
    left: 8%;
    width: 84%;
    height: 40%;
    background: rgba(255,255,255,0.25);
    border-radius: 50px;
    pointer-events: none;
}

/* Hover 漂浮 */
.bank-retry-btn:hover {
    transform: translateY(-2px);
    box-shadow:
        0 8px 0 #002e75,
        0 18px 25px rgba(0, 63, 158, 0.5);
}

/* 按壓下沉效果 */
.bank-retry-btn:active {
    transform: translateY(4px);
    box-shadow:
        0 2px 0 #002e75,
        0 6px 12px rgba(0, 63, 158, 0.4);
}
"""

# ==========================================
# 6. 前端介面設計
# ==========================================
with gr.Blocks(css=custom_css, title="銀行信用貸款審核系統") as demo:
    with gr.Tabs():
        # =========== 分頁 1: 信用貸款審核專區 ===========
        with gr.TabItem("👤 信用貸款審核專區"):
            # 標題
            gr.HTML(
                "<div class='main-header'><h1>🏦信用貸款審核系統</h1><p>讓TibaMe銀行助您實現創業、旅遊與安家夢想</p></div>")

            # 插畫區
            with gr.Row(elem_classes="illustration-row"):
                with gr.Column(scale=1):
                    gr.Image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", show_label=False,
                             container=False,
                             height=100)
                    gr.Markdown("<p style='font-size: 24px; text-align: center; font-weight: bold;'>事業啟航</p>")
                with gr.Column(scale=1):
                    gr.Image("https://cdn-icons-png.flaticon.com/512/201/201623.png", show_label=False, container=False,
                             height=100)
                    gr.Markdown("<p style='font-size: 24px; text-align: center; font-weight: bold;'>旅遊規劃</p>")
                with gr.Column(scale=1):
                    gr.Image("https://cdn-icons-png.flaticon.com/512/609/609803.png", show_label=False, container=False,
                             height=100)
                    gr.Markdown("<p style='font-size: 24px; text-align: center; font-weight: bold;'>築夢成家</p>")

            # 主要表單
            with gr.Row():
                with gr.Column(elem_classes="option-style"):
                    gr.HTML(
                        "<style>.option-style .secondary-wrap, .option-style .single-select { font-size: 20px !important; text-align: center !important; }</style>",
                        visible=False)

                    gr.Markdown("<p style='font-size: 26px; text-align: left; font-weight: bold;'>👤 個人基本資料</p>")
                    p_age = gr.Dropdown(label="年齡", choices=list(range(20, 81)), value=20)
                    p_gender = gr.Dropdown(label="生理性別", choices=["男", "女"], value="男")
                    p_edu = gr.Dropdown(label="教育程度", choices=["高中/職", "副學士(專科)", "學士", "碩士", "博士"],
                                        value="學士")
                    p_home = gr.Dropdown(label="居住狀況", choices=["租賃", "自有（尚有貸款）", "自有（無貸款）"],
                                         value="租賃")
                    p_emp = gr.Dropdown(label="工作年資", choices=list(range(0, 66)), value=0)
                    l_intent = gr.Dropdown(label="貸款用途", choices=["個人周轉", "教育進修", "醫療照護", "創業周轉"],
                                           value="個人周轉")

                with gr.Column():
                    gr.Markdown("<p style='font-size: 26px; text-align: left; font-weight: bold;'>💰 貸款資訊</p>")
                    p_income = gr.Number(label="年收入 (美元)", value=100, elem_id="unit-yuan-1")
                    l_amnt = gr.Number(label="貸款金額 (美元)", value=1000, elem_id="unit-yuan-2")
                    l_rate = gr.Textbox(label="擬定利率 (%)", value="5.00")
                    l_percent = gr.Textbox(label="收支負擔率 (%)", value="10.00")

                    gr.Markdown("<p style='font-size: 26px; text-align: left; font-weight: bold;'>💳 信用評分項目</p>")
                    l_cred_len = gr.Number(label="信用歷史年資", value=2)
                    l_score = gr.Number(label="信用分數", value=650)

            # 事件處理 (精度驗證)
            l_rate.blur(fn=validate_precision, inputs=[l_rate], outputs=[l_rate])
            l_percent.blur(fn=validate_precision, inputs=[l_percent], outputs=[l_percent])

            # 按鈕與結果
            submit_btn = gr.Button("🚀 提交審核申請", elem_classes="submit-btn")
            output_text = gr.Textbox(label="處理結果", placeholder="等待審核結果...")
            with gr.Column(visible=False, elem_id="modal_container") as modal_box:
                gr.Markdown("### ⚠️ 檔案寫入失敗\n請關閉 Excel 或確認檔案未被佔用。")
                retry_btn = gr.Button(
                    "重新提交審核申請",
                    elem_classes="bank-retry-btn"
                )

            # 核心函數參數對應
            submit_btn.click(
                fn=submit_handler,
                inputs=[p_age, p_gender, p_edu, p_income, p_emp, p_home, l_amnt, l_intent, l_rate, l_percent,
                        l_cred_len, l_score],
                outputs=[output_text, modal_box]
            )

            retry_btn.click(
                fn=retry_write_csv,
                inputs=[],
                outputs=modal_box
            )

        # =========== 分頁 2: 銀行後台 ===========
        with gr.TabItem("📊報表管理專區"):
            gr.Markdown("### 📊數據監控儀表板")

            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新即時數據")
                download_btn = gr.Button("📥 匯出 CSV 報表")

            log_display = gr.Dataframe(label="最新申請紀錄 (最後 10 筆)", interactive=False)
            file_download = gr.File(label="檔案下載連結")

            refresh_btn.click(fn=view_latest_logs, inputs=[], outputs=log_display)
            download_btn.click(fn=get_log_file, inputs=[], outputs=file_download)

# ==========================================
# 7. 上傳函數
# ==========================================
def upload_daily_csv_to_gcs():
    if not os.path.exists(CSV_FILE):
        print("ℹ️ 今日無 CSV 可上傳")
        return

    tmp_path = CSV_FILE + ".uploading"

    with csv_lock:
        shutil.copyfile(CSV_FILE, tmp_path)

    try:
        client = storage.Client.from_service_account_json(GCS_KEY_FILE)
        bucket = client.bucket(GCS_BUCKET_NAME)

        today = datetime.now().strftime("%Y%m%d")

        # ==============================
        # 1️⃣ Daily 備份
        # ==============================
        daily_object = f"{GCS_OBJECT_PREFIX}/{today}/{CSV_FILE}"
        bucket.blob(daily_object).upload_from_filename(tmp_path)
        print(f"✅ Daily 已上傳: {daily_object}")

        # ==============================
        # 2️⃣ 更新 master_all.csv
        # ==============================
        master_object = "master/master_all.csv"
        master_blob = bucket.blob(master_object)

        if master_blob.exists():
            # 下載現有 master
            master_tmp = "master_tmp.csv"
            master_blob.download_to_filename(master_tmp)

            df_master = pd.read_csv(master_tmp, encoding="utf_8_sig")
            df_new = pd.read_csv(tmp_path, encoding="utf_8_sig")

            df_combined = pd.concat([df_master, df_new], ignore_index=True)
            df_combined.to_csv(master_tmp, index=False, encoding="utf_8_sig")

            master_blob.upload_from_filename(master_tmp)
            os.remove(master_tmp)

            print("📊 master_all.csv 已更新")

        else:
            # 第一次建立 master
            master_blob.upload_from_filename(tmp_path)
            print("📊 master_all.csv 已建立")

        # ==============================
        # 3️⃣ 清空本地 CSV（保留 header）
        # ==============================
        with csv_lock:
            df_header = pd.read_csv(tmp_path, nrows=0)
            df_header.to_csv(CSV_FILE, index=False, encoding="utf_8_sig")

        print("🧹 本地 CSV 已清空")

    except Exception as e:
        print(f"❌ 上傳失敗: {e}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    scheduler = BackgroundScheduler(timezone=ZoneInfo("Asia/Taipei"))
    scheduler.add_job(
        upload_daily_csv_to_gcs,
        CronTrigger(hour=21, minute=0),
        id="upload_csv_9pm",
        replace_existing=True,
    )
    scheduler.start()
    demo.launch(share=True, debug=True)

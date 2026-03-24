import gradio as gr
import pandas as pd
import joblib
import numpy as np
import os
import re
from datetime import datetime

# ==========================================
#  配合前端UI調整第二版
# 1. 系統初始化與載入模型
# ==========================================
MODEL_FILE = 'lgbm_loan_model.zip'
CSV_FILE = "loan_requests_full.csv"

try:
    model = joblib.load(MODEL_FILE)
    print(f"✅ 模型載入成功: {MODEL_FILE}")
except Exception as e:
    print(f"❌ 模型載入失敗，請確認檔案是否存在: {e}")


    # 建立一個假模型避免程式直接崩潰 (僅供測試介面用)
    class FakeModel:
        def predict(self, x):
            return [0] * len(x)

        def predict_proba(self, x):
            # 固定違約機率 35%  一定落在中風險，會人工審核
            return [[0.65, 0.35]] * len(x)


    model = FakeModel()


# ==========================================
# 2. 前端輔助函數 (同學撰寫的精度驗證)
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
# 3. 核心邏輯函數 (結合前端防呆)
# ==========================================
def process_loan_request(age, gender, edu, income, emp, home, amnt, intent, rate, percent, cred_len, score):
    # --- 第一關：前端防呆機制 ---
    try:
        rate_val = float(rate) if rate else 0.0
        percent_val = float(percent) if percent else 0.0
        validations = [
            (age is not None and 20 <= age <= 80, "年齡須在 20-80 歲之間"),
            (income is not None and income > 0, "年收入須大於 0"),
            (emp is not None and emp >= 0, "工作年資不可為負"),
            (amnt is not None and amnt >= 0, "貸款金額不可為負"),
            (1.00 <= rate_val <= 99.99, "擬定利率須在 1.00 到 99.99 之間"),
            (1.00 <= percent_val <= 99.99, "收支負擔率須在 1.00 到 99.99 之間"),
            (cred_len is not None and cred_len >= 0, "信用年資不可為負"),
            (score is not None and 200 <= score <= 800, "信用分數須在 200-800 之間")
        ]
        for condition, msg in validations:
            if not condition: return f"❌ 系統拒絕審核：{msg}"
    except Exception:
        return "❌ 輸入包含非法字元或格式錯誤。"

    # --- A. [Mapping] 類別轉換 ---
    gender_map = {"男": 1, "女": 0}
    edu_map = {"高中/職": 0, "副學士": 1, "學士": 2, "碩士": 3, "博士": 4}
    home_map = {"租賃": 0, "自有（尚有貸款）": 1, "自有（無貸款）": 2, "其他": 3}
    intent_map = {"個人周轉": 0, "醫療照護": 1, "創業周轉": 2, "教育進修": 3, "債務整合": 4, "家庭裝修": 5}

    gender_num = gender_map.get(gender, 0)
    edu_num = edu_map.get(edu, 0)
    home_num = home_map.get(home, 0)
    intent_num = intent_map.get(intent, 0)

    # --- B. [Feature Engineering] 特徵工程 ---
    try:
        log_income = np.log1p(income)
        rate_per_score = rate_val / score if score > 0 else 0
        is_adult = 1 if age > 25 else 0
    except:
        return "❌ 數值計算錯誤，請檢查輸入是否為數字"

    # --- C. [Prediction] 模型預測 ---
    columns = [
        'person_age', 'person_gender', 'person_education', 'person_income',
        'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score',
        'log_person_income', 'interest_rate_per_credit_score', 'age_group'
    ]

    X_input = pd.DataFrame([[
        age, gender_num, edu_num, income, emp, home_num,
        amnt, intent_num, rate_val, percent_val, cred_len, score,
        log_income, rate_per_score, is_adult
    ]], columns=columns)

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
        "對數收入(計算)": [f"{log_income:.4f}"], "利率信用比(計算)": [f"{rate_per_score:.6f}"],
        "是否超過25歲(計算)": [is_adult], "AI預測結果": [csv_status], "違約機率": [f"{probability:.4f}"]
    }

    df_save = pd.DataFrame(save_data)

    try:
        header = not os.path.exists(CSV_FILE)
        df_save.to_csv(CSV_FILE, mode='a', index=False, header=header, encoding="utf_8_sig")
        save_status = "✅ 資料已存檔"
    except PermissionError:
        save_status = "⚠️ 存檔失敗 (檔案可能被 Excel 開啟中，請關閉後重試)"

    # --- E. 回傳給前端的訊息 ---
    return f"【審核完成】\n結果：{status_emoji}\n違約機率：{probability:.2%}\n(案件編號: {case_id})\n{save_status}"


# ==========================================
# 4. 後台管理函數 (修復過長字串斷行問題)
# ==========================================
def view_latest_logs():
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE, encoding='utf_8_sig', on_bad_lines='skip')
            return df.tail(10)
        except PermissionError:
            # 這裡把過長的字串拆開，避免貼上時斷掉
            err_msg = "❌ 無法讀取：檔案正被 Excel 開啟中，請先關閉 Excel 再刷新。"
            return pd.DataFrame({"系統錯誤": [err_msg]})
        except Exception as e:
            return pd.DataFrame({"系統錯誤": [f"❌ 讀取發生未知錯誤: {str(e)}"]})
    return pd.DataFrame({"系統訊息": ["尚無任何申請紀錄"]})


def get_log_file():
    if os.path.exists(CSV_FILE):
        return CSV_FILE
    return None


# ==========================================
# 5. 自定義 CSS (前端)
# ==========================================
custom_css = """
/* 1. 全域背景與字體 */
.gradio-container { background-color: #f0f8ff; font-family: 'Microsoft JhengHei', 'PingFang TC', sans-serif; }
/* 2. 主標題區 */
.main-header h1 { font-size: 100px !important; font-weight: 800; margin-bottom: 10px; }
.main-header p { font-size: 40px; color: #555; }
.main-header { text-align: center; color: #0056b3; padding: 20px; }
/* 3. 題目字體 */
span[data-testid="block-info"] { font-size: 25px !important; font-weight: bold !important; color: #333 !important; }
/* 4. 輸入數值排版 */
input, select, textarea { font-size: 20px !important; text-align: center !important; color: #0056b3 !important; font-weight: 500; }
/* 5. 插畫區與按鈕 */
.illustration-row { display: flex; justify-content: space-around; padding: 20px; background: white; border-radius: 20px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
.submit-btn { background-color: #007bff !important; color: white !important; border-radius: 30px !important; font-size: 24px !important; font-weight: bold !important; }
/* 6. 元字元定位 */
#unit-yuan-1, #unit-yuan-2 { position: relative; }
#unit-yuan-1::after, #unit-yuan-2::after { content: "元"; position: absolute; right: 25px; bottom: 23px; font-size: 20px; color: #0056b3; font-weight: bold; pointer-events: none; }
#unit-yuan-1 input, #unit-yuan-2 input { padding-right: 0px !important; }
.submit-btn:hover { transform: scale(1.05); transition: 0.3s; box-shadow: 0 5px 15px rgba(0,123,255,0.4); }
"""

# ==========================================
# 6. 前端介面設計
# ==========================================
with gr.Blocks(css=custom_css, title="銀行徵信系統") as demo:
    with gr.Tabs():
        # =========== 分頁 1: 信用貸款審核專區 ===========
        with gr.TabItem("👤 信用貸款審核專區"):
            # 分段 HTML 以避免過長斷行
            title_html = "<div class='main-header'><h1>🏦銀行信用貸款審核系統</h1>"
            subtitle_html = "<p>讓德馬銀行的專業，助您實現創業、旅遊與安居樂業夢想</p></div>"
            gr.HTML(title_html + subtitle_html)

            with gr.Row(elem_classes="illustration-row"):
                with gr.Column(scale=1):
                    gr.Image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", show_label=False,
                             container=False, height=100)
                    gr.Markdown("<p style='font-size: 24px; text-align: center; font-weight: bold;'>創業與夢想</p>")
                with gr.Column(scale=1):
                    gr.Image("https://cdn-icons-png.flaticon.com/512/201/201623.png", show_label=False, container=False,
                             height=100)
                    gr.Markdown("<p style='font-size: 24px; text-align: center; font-weight: bold;'>旅遊規劃</p>")
                with gr.Column(scale=1):
                    gr.Image("https://cdn-icons-png.flaticon.com/512/609/609803.png", show_label=False, container=False,
                             height=100)
                    gr.Markdown("<p style='font-size: 24px; text-align: center; font-weight: bold;'>美好居家</p>")

            with gr.Row():
                with gr.Column(elem_classes="option-style"):
                    css_hack = "<style>.option-style .secondary-wrap, .option-style .single-select { font-size: 20px !important; text-align: center !important; }</style>"
                    gr.HTML(css_hack, visible=False)
                    gr.Markdown("<p style='font-size: 26px; text-align: left; font-weight: bold;'>👤 個人基本資料</p>")

                    p_age = gr.Number(label="年齡 (person_age)", value=25)
                    p_gender = gr.Dropdown(label="性別 (person_gender)", choices=["男", "女"])
                    p_edu = gr.Dropdown(label="教育程度 (person_education)",
                                        choices=["高中/職", "大學", "碩士", "博士"])
                    p_income = gr.Number(label="年收入 (person_income)", value=50000)
                    p_emp = gr.Number(label="工作年資 (person_emp_exp)", value=2)
                    p_home = gr.Dropdown(label="居住狀況 (person_home_ownership)",
                                         choices=["租賃", "自有（尚有貸款）", "自有（無貸款）"])

                with gr.Column():
                    gr.Markdown("<p style='font-size: 26px; text-align: left; font-weight: bold;'>💰 貸款與信用資訊</p>")
                    l_amnt = gr.Number(label="貸款金額 (loan_amnt)", value=10000)
                    l_intent = gr.Dropdown(label="貸款用途 (loan_intent)",
                                           choices=["個人周轉", "教育進修", "醫療照護", "創業周轉"])
                    l_rate = gr.Number(label="擬定利率 (loan_int_rate)", value=11.0)
                    l_percent = gr.Number(label="收支負債比 (loan_percent_income)", value=1.0)
                    l_cred_len = gr.Number(label="信用歷史年資 (cb_person_cred_hist_length)", value=5)
                    l_score = gr.Number(label="信用分數 (credit_score)", value=650)

            submit_btn = gr.Button("開始審核", variant="primary")
            output_text = gr.Textbox(label="系統回饋", lines=4)

            submit_btn.click(
                fn=process_loan_request,
                inputs=[p_age, p_gender, p_edu, p_income, p_emp, p_home, l_amnt, l_intent, l_rate, l_percent,
                        l_cred_len, l_score],
                outputs=output_text
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

if __name__ == "__main__":
    demo.launch()
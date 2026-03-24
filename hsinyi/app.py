import gradio as gr
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime

# ==========================================
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
# 2. 核心邏輯函數 (後端處理)
# ==========================================
def process_loan_request(age, gender, edu, income, emp, home, amnt, intent, rate, percent, cred_len, score):
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
        rate_per_score = rate / score if score > 0 else 0
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
        amnt, intent_num, rate, percent, cred_len, score,
        log_income, rate_per_score, is_adult
    ]], columns=columns)

    try:
        probability = model.predict_proba(X_input)[0][1]
    except Exception as e:
        return f"❌ 預測失敗: {str(e)}"

    # --- 【新增邏輯】設定高中低風險門檻 ---
    if probability >= 0.50:
        status_emoji = "🚨 建議拒絕 (高風險)"
        csv_status = "建議拒絕 (高風險)"
    elif probability >= 0.20:
        status_emoji = "⚠️ 需人工確認 (中風險)"
        csv_status = "需人工確認 (中風險)"
    else:
        status_emoji = "✅ 建議核貸 (低風險)"
        csv_status = "建議核貸 (低風險)"

    # --- D. [Logging] 完整存檔 (標題已全面中文化) ---

    # 【更新點】取得當下時間與日期字串，用來產生專業的案件編號
    current_time = datetime.now()
    date_str = current_time.strftime("%Y%m%d")  # 會產生如 20260219 的格式
    case_id = f"CASE_{date_str}_{np.random.randint(10000, 99999)}"

    save_data = {
        # 1. 系統資訊
        "申請時間": [current_time.strftime("%Y-%m-%d %H:%M:%S")],
        "案件編號": [case_id],  # 使用包含日期的案件編號

        # 2. 原始輸入 (User Input)
        "申請人年齡": [age],
        "性別": [gender],
        "教育程度": [edu],
        "年收入": [income],
        "工作年資": [emp],
        "居住狀況": [home],
        "貸款金額": [amnt],
        "貸款用途": [intent],
        "擬定利率": [rate],
        "收支負債比": [percent],
        "信用歷史年資": [cred_len],
        "信用分數": [score],

        # 3. 後端計算特徵 (Calculated Features)
        "對數收入(計算)": [f"{log_income:.4f}"],
        "利率信用比(計算)": [f"{rate_per_score:.6f}"],
        "是否超過25歲(計算)": [is_adult],

        # 4. 模型輸出 (Model Output)
        "AI預測結果": [csv_status],
        "違約機率": [f"{probability:.4f}"]
    }

    df_save = pd.DataFrame(save_data)

    # 寫入 CSV
    try:
        header = not os.path.exists(CSV_FILE)
        df_save.to_csv(CSV_FILE, mode='a', index=False, header=header, encoding="utf_8_sig")
        save_status = "✅ 資料已存檔"
    except PermissionError:
        save_status = "⚠️ 存檔失敗 (檔案可能被 Excel 開啟中，請關閉後重試)"

    # --- E. 回傳給前端的訊息 ---
    return f"【審核完成】\n結果：{status_emoji}\n違約機率：{probability:.2%}\n(案件編號: {case_id})\n{save_status}"


# --- 管理員函數：讀取最新數據 ---
def view_latest_logs():
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE, encoding='utf_8_sig', on_bad_lines='skip')
            return df.tail(10)
        except PermissionError:
            return pd.DataFrame({"系統錯誤": ["❌ 無法讀取：檔案正被 Excel 開啟中，請先關閉 Excel 再刷新。"]})
        except Exception as e:
            return pd.DataFrame({"系統錯誤": [f"❌ 讀取發生未知錯誤: {str(e)}"]})
    return pd.DataFrame({"系統訊息": ["尚無任何申請紀錄"]})


# --- 管理員函數：下載檔案 ---
def get_log_file():
    if os.path.exists(CSV_FILE):
        return CSV_FILE
    return None


# ==========================================
# 3. 前端介面設計 (Gradio Blocks + Tabs)
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(), title="銀行貸款智能審核系統") as demo:
    gr.Markdown("# 🏦 銀行貸款 AI 智能審核系統")

    with gr.Tabs():
        # =========== 分頁 1: 信用貸款審核專區 (Client Side) ===========
        with gr.TabItem("👤 信用貸款審核專區"):
            gr.Markdown("### 請填寫申請人基本資料，系統將進行 AI 初審。")

            # === 同學的 UI 程式碼 ===
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 👤 個人基本資料")
                    p_age = gr.Number(label="年齡 (person_age)", value=25)
                    p_gender = gr.Dropdown(label="性別 (person_gender)", choices=["男", "女"])
                    p_edu = gr.Dropdown(label="教育程度 (person_education)",
                                        choices=["高中/職", "大學", "碩士", "博士"])
                    p_income = gr.Number(label="年收入 (person_income)", value=50000)
                    p_emp = gr.Number(label="工作年資 (person_emp_exp)", value=2)
                    p_home = gr.Dropdown(label="居住狀況 (person_home_ownership)",
                                         choices=["租賃", "自有（尚有貸款）", "自有（無貸款）"])

                with gr.Column():
                    gr.Markdown("### 💰 貸款與信用資訊")
                    l_amnt = gr.Number(label="貸款金額 (loan_amnt)", value=10000)
                    l_intent = gr.Dropdown(label="貸款用途 (loan_intent)",
                                           choices=["個人周轉", "教育進修", "醫療照護", "創業周轉"])
                    l_rate = gr.Number(label="擬定利率 (loan_int_rate)", value=11.0)
                    l_percent = gr.Number(label="收支負債比 (loan_percent_income)", value=0.1)
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

        # =========== 分頁 2: 銀行後台 (Admin Side) ===========
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
import gradio as gr

# ===== 自己模組 =====
from frontend.api_client import predict_from_api
from frontend.utils import validate_precision
from frontend.services.logger import retry_write
from frontend.services.log_service import view_latest_logs, get_log_file
from config import (
    CSS_FILE,
    GENDER_OPTIONS,
    EDUCATION_OPTIONS,
    HOME_OWNERSHIP_OPTIONS,
    LOAN_INTENT_OPTIONS,
)

# =========================
# ⭐ 核心流程（API + logger）
# =========================
def process_loan_request(age, gender, edu, income, emp, home, amnt, intent, rate, percent, cred_len, score):

    try:
        rate_val = float(rate) if rate else 0.0
        percent_val = float(percent) if percent else 0.0
    except:
        return "❌ 輸入格式錯誤"

    # ===== API =====
    try:
        response_data = predict_from_api(
            age, gender, edu, income, emp, home,
            amnt, intent, rate_val, percent_val,
            cred_len, score
        )
        probability = response_data["probability"]
        case_id = response_data["case_id"]
    except Exception as e:
        return f"❌ API錯誤: {str(e)}"

    # ===== 風險判斷 =====
    if probability >= 0.5:
        status = "🚨 建議拒絕 (高風險)"
    elif probability >= 0.2:
        status = "⚠️ 需人工確認 (中風險)"
    else:
        status = "✅ 建議核貸 (低風險)"

    return f"{status}\n違約機率：{probability:.2%}\n(案件編號: {case_id})\n✅ 已存檔"


def submit_handler(*args):
    result = process_loan_request(*args)

    return result, gr.update(visible=False)


def retry_write_csv():
    success = retry_write()

    if success:
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)


# ===== CSS 載入 =====
with open(CSS_FILE, "r", encoding="utf-8") as f:
    custom_css = f.read()


# =========================
# UI（完整還原）
# =========================
with gr.Blocks() as demo:

    with gr.Tabs():
        # =========== 分頁 1 ===========
        with gr.TabItem("👤 信用貸款審核專區"):

            gr.HTML(
                "<div class='main-header'><h1>🏦信用貸款審核系統</h1><p>讓TibaMe銀行助您實現創業、旅遊與安家夢想</p></div>")

            with gr.Row(elem_classes="illustration-row"):
                with gr.Column():
                    gr.Image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", height=100, container=False)
                    gr.Markdown("<p style='text-align:center;font-weight:bold;'>事業啟航</p>")
                with gr.Column():
                    gr.Image("https://cdn-icons-png.flaticon.com/512/201/201623.png", height=100, container=False)
                    gr.Markdown("<p style='text-align:center;font-weight:bold;'>旅遊規劃</p>")
                with gr.Column():
                    gr.Image("https://cdn-icons-png.flaticon.com/512/609/609803.png", height=100, container=False)
                    gr.Markdown("<p style='text-align:center;font-weight:bold;'>築夢成家</p>")

            with gr.Row():
                with gr.Column(elem_classes="option-style"):

                    gr.Markdown("### 👤 個人基本資料")

                    p_age = gr.Dropdown(list(range(20, 81)), value=20, label="年齡")
                    p_gender = gr.Dropdown(GENDER_OPTIONS, value=GENDER_OPTIONS[0], label="生理性別")
                    p_edu = gr.Dropdown(EDUCATION_OPTIONS, value="學士", label="教育程度")
                    p_home = gr.Dropdown(HOME_OWNERSHIP_OPTIONS, value=HOME_OWNERSHIP_OPTIONS[0], label="居住狀況")
                    p_emp = gr.Dropdown(list(range(0, 66)), value=0, label="工作年資")
                    l_intent = gr.Dropdown(LOAN_INTENT_OPTIONS, value=LOAN_INTENT_OPTIONS[0], label="貸款用途")

                with gr.Column():

                    gr.Markdown("### 💰 貸款資訊")

                    p_income = gr.Number(label="年收入 (美元)", value=100)
                    l_amnt = gr.Number(label="貸款金額 (美元)", value=1000)
                    l_rate = gr.Textbox(label="擬定利率 (%)", value="5.00")
                    l_percent = gr.Textbox(label="收支負擔率 (%)", value="10.00")

                    gr.Markdown("### 💳 信用評分項目")

                    l_cred_len = gr.Number(label="信用歷史年資", value=2)
                    l_score = gr.Number(label="信用分數", value=650)

            l_rate.blur(fn=validate_precision, inputs=l_rate, outputs=l_rate)
            l_percent.blur(fn=validate_precision, inputs=l_percent, outputs=l_percent)

            submit_btn = gr.Button("🚀 提交審核申請", elem_classes="submit-btn")
            output_text = gr.Textbox(label="處理結果")

            with gr.Column(visible=False) as modal_box:
                gr.Markdown("### ⚠️ 檔案寫入失敗")
                retry_btn = gr.Button("重新提交審核申請")

            submit_btn.click(
                fn=submit_handler,
                inputs=[p_age, p_gender, p_edu, p_income, p_emp, p_home, l_amnt, l_intent, l_rate, l_percent,
                        l_cred_len, l_score],
                outputs=[output_text, modal_box]
            )

            retry_btn.click(fn=retry_write_csv, outputs=modal_box)

        # =========== 分頁 2 ===========
        with gr.TabItem("📊報表管理專區"):

            gr.Markdown("### 📊數據監控儀表板")

            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新即時數據")
                download_btn = gr.Button("📥 匯出 CSV 報表")

            log_display = gr.Dataframe(label="最新申請紀錄")
            file_download = gr.File()

            refresh_btn.click(fn=view_latest_logs, outputs=log_display)
            download_btn.click(fn=get_log_file, outputs=file_download)


# =========================
# 啟動（Gradio 6 正確寫法）
# =========================
if __name__ == "__main__":
    demo.launch(css=custom_css, server_name="0.0.0.0", server_port=7860)

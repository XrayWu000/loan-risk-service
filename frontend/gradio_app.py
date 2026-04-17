import gradio as gr

from config.path_config import CSS_FILE
from config.ui_config import (
    EDUCATION_OPTIONS,
    GENDER_OPTIONS,
    HOME_OWNERSHIP_OPTIONS,
    LOAN_INTENT_OPTIONS,
)
from frontend.adapters.logging_adapter import retry_write
from frontend.api_client import predict_from_api
from frontend.services.log_service import get_log_file, view_latest_logs
from frontend.utils import validate_precision


def process_loan_request(age, gender, edu, income, emp, home, amnt, intent, rate, percent, cred_len, score):
    try:
        rate_val = float(rate) if rate else 0.0
        percent_val = float(percent) if percent else 0.0
    except Exception:
        return "輸入資料格式錯誤"

    try:
        response_data = predict_from_api(
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
        )
        probability = response_data["probability"]
        case_id = response_data["case_id"]
        model_status = response_data.get("model_status", "loaded")
    except Exception as exc:
        return f"API 發生錯誤: {exc}"

    if probability >= 0.5:
        status = "拒絕貸款申請 (高風險)"
    elif probability >= 0.2:
        status = "人工審核後再評估 (中風險)"
    else:
        status = "核准貸款申請 (低風險)"

    result = f"{status}\n違約機率：{probability:.2%}\n(案件編號: {case_id})\n已完成紀錄"
    if model_status == "fallback":
        result += "\n目前使用 fallback model，請檢查模型檔案與 LightGBM 環境。"
    return result


def submit_handler(*args):
    result = process_loan_request(*args)
    return result, gr.update(visible=False)


def retry_write_csv():
    success = retry_write()
    if success:
        return gr.update(visible=False)
    return gr.update(visible=True)


with open(CSS_FILE, "r", encoding="utf-8") as f:
    custom_css = f.read()


with gr.Blocks(css=custom_css) as demo:
    with gr.Tabs():
        with gr.TabItem("貸款風險評估"):
            gr.HTML(
                "<div class='main-header'><h1>貸款風險評估系統</h1>"
                "<p>輸入申請資料後，系統會即時評估違約風險並給出審核建議。</p></div>"
            )

            with gr.Row(elem_classes="illustration-row"):
                with gr.Column():
                    gr.Image(
                        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
                        height=100,
                        container=False,
                    )
                    gr.Markdown("<p style='text-align:center;font-weight:bold;'>申請人資料輸入</p>")
                with gr.Column():
                    gr.Image(
                        "https://cdn-icons-png.flaticon.com/512/201/201623.png",
                        height=100,
                        container=False,
                    )
                    gr.Markdown("<p style='text-align:center;font-weight:bold;'>AI 風險預測</p>")
                with gr.Column():
                    gr.Image(
                        "https://cdn-icons-png.flaticon.com/512/609/609803.png",
                        height=100,
                        container=False,
                    )
                    gr.Markdown("<p style='text-align:center;font-weight:bold;'>審核結果輸出</p>")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 申請人基本資料")
                    p_age = gr.Dropdown(list(range(20, 81)), value=20, label="年齡")
                    p_gender = gr.Dropdown(GENDER_OPTIONS, value=GENDER_OPTIONS[0], label="性別")
                    p_edu = gr.Dropdown(EDUCATION_OPTIONS, value=EDUCATION_OPTIONS[2], label="教育程度")
                    p_home = gr.Dropdown(HOME_OWNERSHIP_OPTIONS, value=HOME_OWNERSHIP_OPTIONS[0], label="居住狀態")
                    p_emp = gr.Dropdown(list(range(0, 66)), value=0, label="工作年資")
                    l_intent = gr.Dropdown(LOAN_INTENT_OPTIONS, value=LOAN_INTENT_OPTIONS[0], label="貸款用途")

                with gr.Column():
                    gr.Markdown("### 貸款與財務資料")
                    p_income = gr.Number(label="年收入 (元)", value=100000)
                    l_amnt = gr.Number(label="貸款金額 (元)", value=1000)
                    l_rate = gr.Textbox(label="貸款利率 (%)", value="5.00")
                    l_percent = gr.Textbox(label="負債收入比 (%)", value="10.00")

                    gr.Markdown("### 信用資料")
                    l_cred_len = gr.Number(label="信用歷史長度", value=2)
                    l_score = gr.Number(label="信用分數", value=650)

            l_rate.blur(fn=validate_precision, inputs=l_rate, outputs=l_rate)
            l_percent.blur(fn=validate_precision, inputs=l_percent, outputs=l_percent)

            submit_btn = gr.Button("開始風險評估", elem_classes="submit-btn")
            output_text = gr.Textbox(label="評估結果", lines=6)

            with gr.Column(visible=False) as modal_box:
                gr.Markdown("### 資料暫存失敗，請重新嘗試")
                retry_btn = gr.Button("重新寫入資料")

            submit_btn.click(
                fn=submit_handler,
                inputs=[
                    p_age,
                    p_gender,
                    p_edu,
                    p_income,
                    p_emp,
                    p_home,
                    l_amnt,
                    l_intent,
                    l_rate,
                    l_percent,
                    l_cred_len,
                    l_score,
                ],
                outputs=[output_text, modal_box],
            )

            retry_btn.click(fn=retry_write_csv, outputs=modal_box)

        with gr.TabItem("最新資料紀錄"):
            gr.Markdown("### 查看最近的申請紀錄")

            with gr.Row():
                refresh_btn = gr.Button("重新整理資料")
                download_btn = gr.Button("下載 CSV 紀錄")

            log_display = gr.Dataframe(label="最近紀錄")
            file_download = gr.File()

            refresh_btn.click(fn=view_latest_logs, outputs=log_display)
            download_btn.click(fn=get_log_file, outputs=file_download)

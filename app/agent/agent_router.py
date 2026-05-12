from __future__ import annotations

from typing import Any

from app.agent.prediction_tool import extract_prediction_input, predict_from_question
from app.rag.rag_service import answer_question

RAG_KEYWORDS = [
    "規則",
    "流程",
    "條件",
    "資格",
    "人工審核",
    "人工覆核",
    "補件",
    "文件",
    "申請流程",
    "審核標準",
    "怎麼處理",
    "看哪些資料",
]

PREDICTION_KEYWORDS = [
    "風險",
    "預測",
    "能不能貸",
    "能否貸款",
    "違約",
    "信用分數",
    "收入",
    "貸款金額",
    "利率",
    "負債比",
]


def _contains_any(question: str, keywords: list[str]) -> bool:
    lowered = question.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def decide_route(question: str) -> str:
    wants_rag = _contains_any(question, RAG_KEYWORDS)
    wants_prediction = _contains_any(question, PREDICTION_KEYWORDS)
    extracted = extract_prediction_input(question)
    has_any_prediction_data = bool(extracted.payload)
    has_enough_prediction_data = not extracted.missing_fields

    if wants_rag and not has_any_prediction_data:
        return "rag_only"
    if wants_prediction and has_any_prediction_data and not has_enough_prediction_data:
        return "need_more_info"
    if wants_prediction and wants_rag:
        return "predict_then_rag"
    if wants_prediction and has_enough_prediction_data:
        return "predict_only"
    if wants_rag or not has_any_prediction_data:
        return "rag_only"
    return "need_more_info"


def ask_agent(question: str, top_k: int = 3) -> dict[str, Any]:
    route = decide_route(question)

    if route == "rag_only":
        rag_result = answer_question(question, top_k=top_k)
        return {
            "route": route,
            "answer": rag_result.get("answer"),
            "sources": rag_result.get("sources", []),
            **({"error": rag_result["error"]} if "error" in rag_result else {}),
        }

    if route == "need_more_info":
        prediction_result = predict_from_question(question)
        return {
            "route": route,
            "answer": prediction_result.get("answer"),
            "missing_fields": prediction_result.get("missing_fields", []),
        }

    prediction_result = predict_from_question(question)
    if prediction_result.get("status") == "need_more_info":
        return {
            "route": "need_more_info",
            "answer": prediction_result.get("answer"),
            "missing_fields": prediction_result.get("missing_fields", []),
        }

    prediction = {
        "probability": prediction_result.get("probability"),
        "risk_level": prediction_result.get("risk_level"),
        "case_id": prediction_result.get("case_id"),
        "model_status": prediction_result.get("model_status"),
    }

    if route == "predict_only":
        return {
            "route": route,
            "prediction": prediction,
            "answer": (
                f"模型預測違約風險機率為 {prediction['probability']:.2%}，"
                f"風險分級為 {prediction['risk_level']}。"
            ),
            "sources": [],
        }

    followup_question = (
        f"模型預測結果為 {prediction['risk_level']} 風險。"
        "請根據文件說明此風險分級後續審核流程與人工覆核應查看的資料。"
    )
    rag_result = answer_question(followup_question, top_k=top_k)
    return {
        "route": route,
        "prediction": prediction,
        "answer": (
            f"模型預測違約風險機率為 {prediction['probability']:.2%}，"
            f"風險分級為 {prediction['risk_level']}。\n\n"
            f"{rag_result.get('answer')}"
        ),
        "sources": rag_result.get("sources", []),
        **({"error": rag_result["error"]} if "error" in rag_result else {}),
    }

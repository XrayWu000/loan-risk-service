from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


FIELD_LABELS = {
    "person_age": "年齡",
    "person_gender": "性別",
    "person_education": "教育程度",
    "person_income": "收入",
    "person_emp_exp": "就業年資",
    "person_home_ownership": "居住狀態",
    "loan_amnt": "貸款金額",
    "loan_intent": "貸款用途",
    "loan_int_rate": "貸款利率",
    "loan_percent_income": "貸款收入占比",
    "cb_person_cred_hist_length": "信用歷史長度",
    "credit_score": "信用分數",
}


@dataclass(frozen=True)
class ExtractedPredictionInput:
    payload: dict[str, Any]
    missing_fields: list[str]


def _required_fields() -> list[str]:
    from app.main import LoanInput

    if hasattr(LoanInput, "model_fields"):
        return [name for name, field in LoanInput.model_fields.items() if field.is_required()]
    return list(LoanInput.__fields__.keys())


def _find_number(patterns: list[str], question: str, cast=float) -> Any | None:
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return cast(match.group(1).replace(",", ""))
    return None


def _find_choice(choices: dict[str, list[str]], question: str) -> str | None:
    lowered = question.lower()
    for value, keywords in choices.items():
        for keyword in keywords:
            if keyword.lower() in lowered:
                return value
    return None


def extract_prediction_input(question: str) -> ExtractedPredictionInput:
    payload: dict[str, Any] = {}

    payload["person_age"] = _find_number([r"(?:年齡|age)\s*[:：]?\s*(\d+(?:\.\d+)?)", r"(\d+(?:\.\d+)?)\s*歲"], question)
    payload["person_income"] = _find_number([r"(?:收入|月收入|年收入|income)\s*[:：]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)"], question)
    payload["person_emp_exp"] = _find_number([r"(?:就業年資|工作年資|年資|emp(?:loyment)?[_ ]?exp)\s*[:：]?\s*(\d+)"], question, int)
    payload["loan_amnt"] = _find_number([r"(?:貸款金額|貸款|借款金額|loan(?:_amount|_amnt)?)\s*[:：]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)"], question)
    payload["loan_int_rate"] = _find_number([r"(?:利率|interest(?:_rate)?)\s*[:：]?\s*(\d+(?:\.\d+)?)\s*%?"], question)
    payload["loan_percent_income"] = _find_number([r"(?:負債比|貸款收入占比|loan_percent_income|percent_income)\s*[:：]?\s*(\d+(?:\.\d+)?)\s*%?"], question)
    payload["cb_person_cred_hist_length"] = _find_number([r"(?:信用歷史長度|信用年資|cred(?:it)?[_ ]?hist(?:ory)?[_ ]?length)\s*[:：]?\s*(\d+(?:\.\d+)?)"], question)
    payload["credit_score"] = _find_number([r"(?:信用分數|credit(?:_score)?)\s*[:：]?\s*(\d+)"], question, int)

    payload["person_gender"] = _find_choice(
        {
            "男": ["男", "男性", "male"],
            "女": ["女", "女性", "female"],
        },
        question,
    )
    payload["person_education"] = _find_choice(
        {
            "高中/職": ["高中", "高職", "high school"],
            "副學士(專科)": ["副學士", "專科", "associate"],
            "學士": ["學士", "大學", "bachelor"],
            "碩士": ["碩士", "master"],
            "博士": ["博士", "phd", "doctor"],
        },
        question,
    )
    payload["person_home_ownership"] = _find_choice(
        {
            "租賃": ["租賃", "租屋", "rent"],
            "自有（尚有貸款）": ["尚有貸款", "mortgage"],
            "自有（無貸款）": ["無貸款", "自有房", "own"],
            "其他": ["其他", "other"],
        },
        question,
    )
    payload["loan_intent"] = _find_choice(
        {
            "個人周轉": ["個人周轉", "個人", "personal"],
            "醫療照護": ["醫療", "照護", "medical"],
            "創業周轉": ["創業", "venture"],
            "教育進修": ["教育", "進修", "education"],
        },
        question,
    )

    payload = {key: value for key, value in payload.items() if value is not None}
    missing_fields = [field for field in _required_fields() if field not in payload]
    return ExtractedPredictionInput(payload=payload, missing_fields=missing_fields)


def risk_level_from_probability(probability: float) -> str:
    if probability >= 0.5:
        return "high"
    if probability >= 0.2:
        return "medium"
    return "low"


def predict_from_question(question: str) -> dict[str, Any]:
    extracted = extract_prediction_input(question)
    if extracted.missing_fields:
        return {
            "status": "need_more_info",
            "missing_fields": extracted.missing_fields,
            "answer": "目前資料不足，無法進行風險預測。請補充：" + "、".join(
                f"{field}（{FIELD_LABELS.get(field, field)}）" for field in extracted.missing_fields
            ) + "。",
        }

    from app.main import LoanInput, predict

    response = predict(LoanInput(**extracted.payload))
    probability = float(response["probability"])
    return {
        "status": "success",
        "probability": probability,
        "risk_level": risk_level_from_probability(probability),
        "case_id": response.get("case_id"),
        "model_status": response.get("model_status"),
    }

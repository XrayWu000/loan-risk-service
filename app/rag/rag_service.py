from __future__ import annotations

from typing import Any

from app.agent.ollama_client import OllamaUnavailableError, chat_with_ollama
from app.rag.retriever import retriever


SYSTEM_PROMPT = """你是貸款審核文件查詢助理。請遵守以下規則：
1. 僅根據檢索到的文件片段回答。
2. 如果文件沒有足夠資訊，請回答：「目前文件中沒有明確說明」。
3. 不得自行編造貸款政策、利率、審核規則或流程。
4. 回答需附上來源文件與片段資訊。
5. 回答語言使用繁體中文。
"""


def _format_context(sources: list[dict[str, Any]]) -> str:
    blocks = []
    for index, source in enumerate(sources, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[片段 {index}]",
                    f"source_file: {source.get('source_file')}",
                    f"section_title: {source.get('section_title')}",
                    f"chunk_id: {source.get('chunk_id')}",
                    f"text: {source.get('text')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def answer_question(question: str, top_k: int = 3) -> dict[str, Any]:
    sources = retriever.retrieve(question, top_k=top_k)

    if not sources:
        return {
            "answer": "目前文件中沒有明確說明。",
            "sources": [],
        }

    context = _format_context(sources)
    user_prompt = f"""問題：
{question}

可用文件片段：
{context}

請根據上述文件片段回答，並在回答最後列出引用來源。"""

    try:
        answer = chat_with_ollama(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
    except OllamaUnavailableError as exc:
        return {
            "error": str(exc),
            "answer": str(exc),
            "sources": sources,
        }

    return {
        "answer": answer,
        "sources": sources,
    }

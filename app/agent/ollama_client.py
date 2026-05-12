from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:26b"
DEFAULT_OLLAMA_TIMEOUT = 120


class OllamaUnavailableError(RuntimeError):
    pass


def _model_error(model: str) -> str:
    return f"Ollama model {model} is not available. Please run: ollama pull {model}"


def chat_with_ollama(
    messages: list[dict[str, str]],
    model: str | None = None,
    base_url: str | None = None,
    timeout: int | None = None,
) -> str:
    model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    base_url = (base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)).rstrip("/")
    timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", str(DEFAULT_OLLAMA_TIMEOUT)))

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    try:
        response = httpx.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    except httpx.RequestError as exc:
        raise OllamaUnavailableError(
            f"Cannot connect to Ollama at {base_url}. Please start Ollama and pull {model}."
        ) from exc

    if response.status_code == 404:
        raise OllamaUnavailableError(_model_error(model))

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text
        if "model" in detail.lower() and ("not found" in detail.lower() or "not available" in detail.lower()):
            raise OllamaUnavailableError(_model_error(model)) from exc
        raise OllamaUnavailableError(f"Ollama request failed: {detail}") from exc

    data = response.json()
    message = data.get("message", {})
    content = message.get("content")
    if not content:
        raise OllamaUnavailableError("Ollama returned an empty response.")
    return str(content).strip()

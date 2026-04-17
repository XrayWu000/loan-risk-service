import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from time import time


def get_config_path() -> Path:
    env_path = os.getenv("NOTIFICATION_CONFIG_PATH", "").strip()
    if env_path:
        return Path(env_path)

    return Path(__file__).resolve().parents[1] / "notification_settings.json"


def get_notification_state_path() -> Path:
    env_path = os.getenv("NOTIFICATION_STATE_PATH", "").strip()
    if env_path:
        return Path(env_path)

    return Path(__file__).resolve().parent / "notification_state.json"


NOTIFICATION_CONFIG_PATH = get_config_path()
NOTIFICATION_STATE_PATH = get_notification_state_path()
FALLBACK_ALERT_COOLDOWN_SECONDS = int(
    os.getenv("FALLBACK_ALERT_COOLDOWN_SECONDS", "600").strip()
)


def _load_recipient_email() -> str | None:
    if not NOTIFICATION_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {NOTIFICATION_CONFIG_PATH}")

    with NOTIFICATION_CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    email = str(data.get("fallback_alert_email", "")).strip()
    return email or None


def _load_notification_state() -> dict:
    if not NOTIFICATION_STATE_PATH.exists():
        return {}

    try:
        with NOTIFICATION_STATE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_notification_state(state: dict) -> None:
    NOTIFICATION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTIFICATION_STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _is_fallback_alert_rate_limited() -> bool:
    state = _load_notification_state()
    last_sent_at = float(state.get("fallback_alert_last_sent_at", 0) or 0)
    if last_sent_at <= 0:
        return False

    return (time() - last_sent_at) < FALLBACK_ALERT_COOLDOWN_SECONDS


def _mark_fallback_alert_sent() -> None:
    state = _load_notification_state()
    state["fallback_alert_last_sent_at"] = time()
    _save_notification_state(state)


def send_fallback_model_alert(error_message: str) -> str:
    recipient = _load_recipient_email()
    if not recipient:
        return "skipped_missing_recipient"

    if _is_fallback_alert_rate_limited():
        return "skipped_rate_limited"

    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_from = os.getenv("SMTP_FROM", smtp_username).strip()
    use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() not in {"0", "false", "no"}

    if not smtp_username or not smtp_password or not smtp_from:
        return "skipped_missing_smtp_credentials"

    message = EmailMessage()
    message["Subject"] = "[loan-risk-service] Model fallback alert"
    message["From"] = smtp_from
    message["To"] = recipient
    message.set_content(
        "loan-risk-service switched to the fallback model.\n\n"
        f"MODEL_PATH load error:\n{error_message}\n"
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            if use_tls:
                server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(message)

        _mark_fallback_alert_sent()
        return "sent"
    except Exception as exc:
        print(f"Failed to send fallback alert email: {exc}")
        return f"failed:{exc.__class__.__name__}"

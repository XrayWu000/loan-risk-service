import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def get_path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name, "").strip()
    return Path(value) if value else default


def get_bool(env_name: str, default: bool = False) -> bool:
    value = os.getenv(env_name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def get_int(env_name: str, default: int = 0) -> int:
    value = os.getenv(env_name, "").strip()
    if not value:
        return default
    return int(value)

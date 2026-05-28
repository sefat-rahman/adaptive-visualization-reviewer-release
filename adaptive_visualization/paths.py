"""Path helpers for the self-contained Adaptive Visualization app."""

from __future__ import annotations

from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
GENERATED_DATA_DIR = DATA_DIR / "generated_data"
DEFAULT_DATASET_ID = "2016-12"
DEFAULT_DATASET_DIR = GENERATED_DATA_DIR / DEFAULT_DATASET_ID


def ensure_app_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def resolve_data_dir(value: str | Path | None = None) -> Path:
    ensure_app_directories()

    if value is None or str(value).strip() == "":
        return DEFAULT_DATASET_DIR

    raw = Path(value)
    if raw.is_absolute():
        return raw

    candidates = [
        DATA_DIR / raw,
        GENERATED_DATA_DIR / raw,
        Path.cwd() / raw,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return DATA_DIR / raw

"""Production configuration helpers. Secrets are never returned to clients."""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    raw = os.environ.get("DATA_DIR") or os.environ.get("RENDER_DISK_PATH")
    if raw:
        path = Path(raw).expanduser().resolve()
    else:
        path = Path(__file__).resolve().parents[1] / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_persistent_paths() -> dict:
    """
    Ensure SQLite parent directories exist.

    If MARKET_DATA_DB / PAPER_TRADING_DB are unset, default them under DATA_DIR
    so Render/Railway persistent disks work with one mount. This function does
    not mutate the process environment (keeps tests hermetic).
    """
    root = data_dir()
    market = os.environ.get("MARKET_DATA_DB") or str(root / "market.sqlite")
    paper = os.environ.get("PAPER_TRADING_DB") or str(root / "paper.sqlite")
    market_path = Path(market).expanduser().resolve()
    paper_path = Path(paper).expanduser().resolve()
    market_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    return {
        "data_dir": str(root),
        "market_data_db": str(market_path),
        "paper_trading_db": str(paper_path),
    }


def public_runtime_config() -> dict:
    """Safe subset of runtime config for health/diagnostics (no secrets)."""
    paths = ensure_persistent_paths()
    ai_key_set = bool(str(os.environ.get("AI_REASONING_API_KEY", "")).strip())
    broker_set = all(
        str(os.environ.get(key, "")).strip()
        for key in ("BROKER_LIVE_BASE_URL", "BROKER_API_KEY", "BROKER_ACCESS_TOKEN")
    )
    return {
        "environment": os.environ.get("APP_ENV", os.environ.get("ENVIRONMENT", "development")),
        "broker_execution": False,
        "broker_live_configured": broker_set,
        "ai_reasoning_configured": ai_key_set,
        "cors_origins_configured": bool(str(os.environ.get("API_CORS_ORIGINS", "")).strip()),
        "persistence": paths,
        "live_feed_verification": (
            "M3 offline validation is complete. "
            "Live NSE-session quote ingestion remains pending verification during the next market session."
        ),
    }


def cors_origin_list() -> list[str]:
    raw = os.environ.get(
        "API_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173",
    )
    origins = [part.strip() for part in raw.split(",") if part.strip()]
    # Support a deliberate wildcard only when explicitly configured.
    if origins == ["*"]:
        return ["*"]
    return origins

"""Overview and diagnostics endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from api.deps import get_market_store, get_paper_store, paper_engine
from api.serialize import jsonable
from market_data.providers import BrokerLiveProvider, YFinanceHistoryProvider
from market_data.session import nse_cash_status
from paper_trading.engine import PAPER_ACCOUNT_ID

router = APIRouter(prefix="/api", tags=["overview"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "multi-factor-financial-decision-system",
        "time": datetime.now(timezone.utc).isoformat(),
        "broker_execution": False,
        "note": "API exposes research and paper trading only. Real broker orders are disabled.",
    }


@router.get("/overview")
def overview() -> dict:
    now = datetime.now(timezone.utc)
    market = get_market_store().stats()
    paper = paper_engine(mode="paper").portfolio_snapshot()
    history = YFinanceHistoryProvider().status()
    broker = BrokerLiveProvider().status()
    return {
        "checked_at": now.isoformat(),
        "nse_cash_session": nse_cash_status(now),
        "yahoo_freshness": history.freshness,
        "broker_configured": broker.configured,
        "broker_freshness": broker.freshness,
        "market_database": market,
        "paper_account": {
            "account_id": PAPER_ACCOUNT_ID,
            "cash": paper["cash"],
            "equity": paper["equity"],
            "positions": len(paper["positions"]),
            "total_return": paper["total_return"],
        },
        "paper_database": get_paper_store().stats(),
        "disclaimer": (
            "Research and paper-trading system. Not financial advice. "
            "Not a live execution platform."
        ),
        "live_verification": (
            "Live NSE-session quote ingestion remains pending verification during the next market session."
        ),
    }


@router.get("/diagnostics")
def diagnostics() -> dict:
    import os

    keys = [
        "MARKET_DATA_DB",
        "MARKET_DATA_WATCHLIST",
        "BROKER_LIVE_BASE_URL",
        "BROKER_API_KEY",
        "BROKER_ACCESS_TOKEN",
        "PAPER_TRADING_DB",
        "API_CORS_ORIGINS",
        "AI_REASONING_BASE_URL",
        "AI_REASONING_API_KEY",
        "AI_REASONING_MODEL",
    ]
    env = []
    for key in keys:
        raw = os.environ.get(key)
        secret = any(part in key for part in ("KEY", "TOKEN"))
        env.append(
            {
                "variable": key,
                "set": bool(raw),
                "display": "«configured, value hidden»" if raw and secret else (raw or "empty"),
            }
        )
    from reasoning.service import provider_status

    return {
        "environment": env,
        "market_store": jsonable(get_market_store().stats()),
        "paper_store": jsonable(get_paper_store().stats()),
        "broker_execution_enabled": False,
        "ai_reasoning": provider_status(),
    }

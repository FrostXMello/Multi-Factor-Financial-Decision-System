"""Market-data HTTP routes. Wraps existing market_data modules."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from api.deps import get_market_store
from api.serialize import jsonable
from market_data.ingest import IngestService
from market_data.providers import BrokerLiveProvider, YFinanceHistoryProvider
from market_data.session import nse_cash_status

router = APIRouter(prefix="/api/market", tags=["market"])


def _watchlist() -> list[str]:
    raw = os.environ.get("MARKET_DATA_WATCHLIST", "RELIANCE.NS,TCS.NS,HDFCBANK.NS")
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


@router.get("/status")
def market_status() -> dict:
    now = datetime.now(timezone.utc)
    store = get_market_store()
    history = YFinanceHistoryProvider().status()
    broker = BrokerLiveProvider().status()
    session = nse_cash_status(now)
    stats = store.stats()
    quotes = store.latest_quotes(_watchlist())
    if session == "closed":
        data_mode = "HISTORICAL" if stats.get("candles", 0) else ("UNCONFIGURED" if not broker.configured else "CLOSED")
    elif not quotes.empty and set(quotes["freshness"].astype(str)) == {"live"}:
        data_mode = "LIVE"
    elif not broker.configured:
        data_mode = "UNCONFIGURED"
    else:
        data_mode = "DELAYED"
    return {
        "checked_at": now.isoformat(),
        "nse_cash_session": session,
        "data_mode": data_mode,
        "providers": [
            {
                "provider_id": history.provider_id,
                "kind": history.kind,
                "configured": history.configured,
                "connected": history.connected,
                "freshness": history.freshness,
                "websocket": history.websocket,
                "detail": history.detail,
            },
            {
                "provider_id": broker.provider_id,
                "kind": broker.kind,
                "configured": broker.configured,
                "connected": broker.connected,
                "freshness": broker.freshness,
                "websocket": broker.websocket,
                "detail": broker.detail,
            },
        ],
        "database": stats,
        "note": (
            "M3 offline validation is complete. "
            "Live NSE-session quote ingestion remains pending verification during the next market session."
        ),
    }


@router.get("/quotes")
def market_quotes() -> dict:
    store = get_market_store()
    watchlist = _watchlist()
    quotes = store.latest_quotes(watchlist)
    return {
        "watchlist": watchlist,
        "quotes": jsonable(quotes),
        "label": "Stored quotes keep their provider freshness. Missing rows are not invented as live.",
    }


@router.get("/candles/{symbol}")
def market_candles(symbol: str, timeframe: str = "1d", limit: int = 60) -> dict:
    store = get_market_store()
    candles = store.read_candles(symbol.strip().upper(), timeframe)
    if candles.empty:
        return {
            "symbol": symbol.strip().upper(),
            "timeframe": timeframe,
            "candles": [],
            "label": "HISTORICAL",
            "detail": "No candles stored for this symbol/timeframe.",
        }
    return {
        "symbol": symbol.strip().upper(),
        "timeframe": timeframe,
        "candles": jsonable(candles.tail(max(1, min(limit, 500)))),
        "label": "HISTORICAL",
        "detail": "Rows are from the local SQLite store (Yahoo research backfill), not a live tape.",
    }


@router.get("/coverage")
def market_coverage() -> dict:
    return {"coverage": jsonable(get_market_store().coverage())}


@router.get("/quality")
def market_quality(limit: int = 30) -> dict:
    store = get_market_store()
    return {
        "quality_events": jsonable(store.quality_events(limit)),
        "provider_events": jsonable(store.provider_events(limit)),
    }


@router.post("/collect")
def market_collect() -> dict:
    store = get_market_store()
    broker = BrokerLiveProvider()
    service = IngestService(store, broker, _watchlist())
    result = service.collect_once(datetime.now(timezone.utc))
    payload = {
        "market_status": result.market_status,
        "fetched": result.fetched,
        "freshness": result.freshness,
        "detail": result.detail,
        "error": result.error,
    }
    if result.error:
        raise HTTPException(status_code=502, detail=payload)
    return payload

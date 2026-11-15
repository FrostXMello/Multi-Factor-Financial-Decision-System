"""Shared API dependencies. Never fabricates live market data."""

from __future__ import annotations

import os
from functools import lru_cache

from market_data.store import MarketStore, open_store
from paper_trading.demo import DEMO_ACCOUNT_ID, DEMO_INITIAL_CAPITAL
from paper_trading.engine import PAPER_ACCOUNT_ID, PaperTradingEngine
from paper_trading.price_sources import quote_from_market_store
from paper_trading.store import PaperStore, open_paper_store


@lru_cache(maxsize=1)
def get_market_store() -> MarketStore:
    return open_store()


@lru_cache(maxsize=1)
def get_paper_store() -> PaperStore:
    return open_paper_store()


def clear_store_caches() -> None:
    get_market_store.cache_clear()
    get_paper_store.cache_clear()


def paper_engine(*, mode: str = "paper") -> PaperTradingEngine:
    mode_u = str(mode).strip().lower()
    store = get_paper_store()
    if mode_u == "demo":
        return PaperTradingEngine.demo(store)
    return PaperTradingEngine(
        store,
        account_id=PAPER_ACCOUNT_ID,
        mode="paper",
        price_lookup=lambda symbol: quote_from_market_store(symbol, store=get_market_store()),
    )


def cors_origins() -> list[str]:
    raw = os.environ.get("API_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [part.strip() for part in raw.split(",") if part.strip()]

"""Shared API dependencies. Never fabricates live market data."""

from __future__ import annotations

from functools import lru_cache

from api.settings import cors_origin_list, ensure_persistent_paths
from market_data.store import MarketStore, open_store
from paper_trading.engine import PAPER_ACCOUNT_ID, PaperTradingEngine
from paper_trading.price_sources import quote_from_market_store
from paper_trading.store import PaperStore, open_paper_store


@lru_cache(maxsize=1)
def get_market_store() -> MarketStore:
    paths = ensure_persistent_paths()
    return open_store(paths["market_data_db"])


@lru_cache(maxsize=1)
def get_paper_store() -> PaperStore:
    paths = ensure_persistent_paths()
    return open_paper_store(paths["paper_trading_db"])


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
    return cors_origin_list()

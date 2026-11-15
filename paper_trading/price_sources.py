"""Resolve paper marks from the market-data store without inventing live prints."""

from __future__ import annotations

from datetime import timezone

import pandas as pd

from market_data.store import MarketStore, open_store
from paper_trading.prices import PriceQuote


def quote_from_market_store(symbol: str, store: MarketStore | None = None) -> PriceQuote | None:
    """
    Prefer the latest stored quote. Fall back to the latest daily candle close.

    Returned freshness is never upgraded to live.
    """
    db = store or open_store()
    symbol_u = str(symbol).strip().upper()
    quotes = db.latest_quotes([symbol_u])
    if not quotes.empty:
        row = quotes.iloc[0]
        raw = str(row.get("freshness", "delayed")).lower()
        if raw == "live":
            freshness, live_authorized = "live", True
        elif raw in {"delayed", "cached", "historical", "unavailable"}:
            freshness, live_authorized = raw, False
        elif raw == "simulated":
            freshness, live_authorized = "cached", False
        else:
            freshness, live_authorized = "delayed", False
        return PriceQuote(
            symbol=symbol_u,
            price=float(row["last_price"]),
            source=str(row.get("provider_id", "market_store_quote")),
            quote_time=pd.to_datetime(row["quote_time"], utc=True).to_pydatetime(),
            freshness=freshness,
            live_authorized=live_authorized,
        )
    candles = db.read_candles(symbol_u, "1d")
    if candles.empty:
        return None
    last = candles.iloc[-1]
    bar_time = pd.to_datetime(last["Date"], utc=True).to_pydatetime()
    if bar_time.tzinfo is None:
        bar_time = bar_time.replace(tzinfo=timezone.utc)
    return PriceQuote(
        symbol=symbol_u,
        price=float(last["Close"]),
        source="market_store_candle_close",
        quote_time=bar_time,
        freshness="historical",
        live_authorized=False,
    )

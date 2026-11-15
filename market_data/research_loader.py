"""
Read research candles from the store, backfilling gaps from Yahoo Finance.

Returned frames keep the Date / OHLCV columns the feature builder already expects.
Bars after `as_of` are dropped so a later cache cannot leak into an earlier study.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from data_loader import DEFAULT_PERIOD_BY_INTERVAL, DownloadConfig, _normalize_interval, download_stock_data
from market_data.providers import YFinanceHistoryProvider
from market_data.quality import find_duplicate_timestamps, find_missing_sessions
from market_data.store import MarketStore, open_store

_HISTORY = YFinanceHistoryProvider()


def load_research_timeframes(
    ticker: str,
    *,
    intervals: tuple[str, ...],
    auto_adjust: bool = True,
    store: MarketStore | None = None,
    as_of: datetime | pd.Timestamp | None = None,
    period_by_interval: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    if not intervals:
        raise ValueError("intervals must contain at least one interval")
    book = store or open_store()
    book.register_provider(
        _HISTORY.provider_id,
        _HISTORY.kind,
        "Yahoo Finance research history. Not a live market feed.",
    )
    period_map = dict(DEFAULT_PERIOD_BY_INTERVAL)
    if period_by_interval:
        period_map.update({str(key).strip().lower(): value for key, value in period_by_interval.items()})
    cutoff = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    out: dict[str, pd.DataFrame] = {}
    for interval in intervals:
        norm = _normalize_interval(interval)
        out[norm] = _sync_interval(
            book,
            ticker.strip().upper(),
            norm,
            period=period_map.get(norm, "1y"),
            auto_adjust=auto_adjust,
            cutoff=cutoff,
        )
    return out


def _sync_interval(
    store: MarketStore,
    symbol: str,
    timeframe: str,
    *,
    period: str,
    auto_adjust: bool,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    stored = store.read_candles(symbol, timeframe, adjusted=auto_adjust)
    start = None
    if len(stored):
        start = pd.Timestamp(stored["Date"].iloc[-1]).strftime("%Y-%m-%d")
    fresh = _download_or_empty(symbol, timeframe, period=period, auto_adjust=auto_adjust, start=start, have_stored=len(stored) > 0)
    combined = _merge_frames(stored, fresh)
    if combined.empty:
        raise ValueError(f"No candles available for {symbol} {timeframe}")
    combined = combined.loc[pd.to_datetime(combined["Date"]) <= cutoff].reset_index(drop=True)
    if combined.empty:
        raise ValueError(f"No candles on or before {cutoff} for {symbol} {timeframe}")
    if not fresh.empty:
        incoming = fresh.loc[pd.to_datetime(fresh["Date"]) <= cutoff]
        if not incoming.empty:
            store.upsert_candles(
                incoming,
                symbol=symbol,
                timeframe=timeframe,
                provider_id=_HISTORY.provider_id,
                adjusted=auto_adjust,
            )
            _record_gaps(store, symbol, timeframe, combined)
    visible = combined.drop(columns=[c for c in ("QualityFlags",) if c in combined.columns], errors="ignore")
    return visible.reset_index(drop=True)


def _download_or_empty(
    symbol: str,
    timeframe: str,
    *,
    period: str,
    auto_adjust: bool,
    start: str | None,
    have_stored: bool,
) -> pd.DataFrame:
    try:
        return download_stock_data(
            symbol,
            cfg=DownloadConfig(period=period, interval=timeframe, auto_adjust=auto_adjust, start=start),
        )
    except Exception:
        if have_stored:
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        raise


def _merge_frames(stored: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    parts = []
    for frame in (stored, fresh):
        if frame is None or frame.empty:
            continue
        part = frame.copy()
        part["Date"] = pd.to_datetime(part["Date"])
        parts.append(part[columns])
    if not parts:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return combined.reset_index(drop=True)


def _record_gaps(store: MarketStore, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
    duplicates = find_duplicate_timestamps(frame)
    for stamp in duplicates:
        store.record_provider_event(_HISTORY.provider_id, "warning", f"duplicate {symbol} {timeframe} {stamp}")
    missing = find_missing_sessions(frame, timeframe)
    if missing:
        sample = ", ".join(str(item.date()) for item in missing[:5])
        store.record_provider_event(
            _HISTORY.provider_id,
            "info",
            f"{symbol} {timeframe} has {len(missing)} weekday or intrabar gaps (holidays included). First: {sample}",
        )

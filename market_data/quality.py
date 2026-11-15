"""Candle and quote checks. Flags are recorded; they are not trading signals."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from market_data.session import IST


def ohlc_problems(open_: float, high: float, low: float, close: float, volume: float) -> list[str]:
    problems: list[str] = []
    values = (open_, high, low, close, volume)
    if not all(np.isfinite(v) for v in values):
        return ["incomplete_candle"]
    if open_ <= 0 or close <= 0 or high <= 0 or low <= 0:
        problems.append("invalid_price")
    if volume < 0:
        problems.append("invalid_volume")
    if high < low:
        problems.append("high_below_low")
    if high + 1e-8 < max(open_, close) or low - 1e-8 > min(open_, close):
        problems.append("ohlc_range")
    return problems


def find_duplicate_timestamps(frame: pd.DataFrame, time_col: str = "Date") -> list[pd.Timestamp]:
    if frame is None or frame.empty or time_col not in frame.columns:
        return []
    stamps = pd.to_datetime(frame[time_col])
    duplicated = stamps[stamps.duplicated(keep=False)].drop_duplicates()
    return [pd.Timestamp(v) for v in duplicated.tolist()]


def find_missing_sessions(frame: pd.DataFrame, timeframe: str, time_col: str = "Date") -> list[pd.Timestamp]:
    """
    Weekday gaps for daily bars. Intraday gaps larger than one bar during the same session.

    NSE holidays look like missing weekdays. The report says so; it does not invent a holiday list.
    """
    if frame is None or len(frame) < 2 or time_col not in frame.columns:
        return []
    stamps = pd.to_datetime(frame[time_col]).sort_values().drop_duplicates()
    token = str(timeframe).lower()
    missing: list[pd.Timestamp] = []
    if token in {"1d", "1day", "d"}:
        expected = pd.bdate_range(stamps.iloc[0], stamps.iloc[-1])
        have = set(pd.Timestamp(v).normalize() for v in stamps)
        for day in expected:
            if pd.Timestamp(day).normalize() not in have:
                missing.append(pd.Timestamp(day))
        return missing
    step = _bar_step(token)
    if step is None:
        return []
    values = list(stamps)
    for earlier, later in zip(values, values[1:]):
        gap = later - earlier
        if gap <= step:
            continue
        if earlier.tzinfo is None:
            same_day = earlier.normalize() == later.normalize()
        else:
            same_day = earlier.astimezone(IST).date() == later.astimezone(IST).date()
        if same_day:
            cursor = earlier + step
            while cursor < later:
                missing.append(pd.Timestamp(cursor))
                cursor += step
    return missing


def find_price_jumps(frame: pd.DataFrame, *, threshold: float = 0.25, time_col: str = "Date") -> list[dict]:
    if frame is None or len(frame) < 2 or "Close" not in frame.columns:
        return []
    ordered = frame.sort_values(time_col)
    close = ordered["Close"].astype(float)
    previous = close.shift(1)
    change = close / previous - 1.0
    hits = ordered.loc[change.abs() > float(threshold)]
    rows = []
    for idx in hits.index:
        rows.append(
            {
                "Date": pd.Timestamp(ordered.loc[idx, time_col]),
                "return": float(change.loc[idx]),
            }
        )
    return rows


def quote_is_stale(quote_time: datetime | pd.Timestamp, now: datetime | pd.Timestamp, max_age: timedelta) -> bool:
    left = pd.Timestamp(quote_time)
    right = pd.Timestamp(now)
    if left.tzinfo is None and right.tzinfo is not None:
        left = left.tz_localize(right.tzinfo)
    if right.tzinfo is None and left.tzinfo is not None:
        right = right.tz_localize(left.tzinfo)
    return (right - left) > max_age


def _bar_step(timeframe: str) -> pd.Timedelta | None:
    token = timeframe.lower().strip()
    mapping = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "60m": pd.Timedelta(minutes=60),
        "1h": pd.Timedelta(hours=1),
    }
    return mapping.get(token)

"""SQLite schema, idempotent candles, and validation. No network and no live feed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from market_data.quality import find_duplicate_timestamps, find_missing_sessions, quote_is_stale
from market_data.store import MarketStore
from market_data.types import Quote


def _bars(days: int, close_at=None) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days)
    close = [100 + i for i in range(days)]
    if close_at:
        for idx, price in close_at.items():
            close[idx] = price
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": [c + 1 for c in close],
            "Low": [c - 1 for c in close],
            "Close": close,
            "Volume": [1_000] * days,
        }
    )


def test_schema_survives_reopen(tmp_path):
    path = tmp_path / "market.sqlite"
    store = MarketStore(path)
    assert store.migrate() == 1
    store.upsert_candles(_bars(3), symbol="RELIANCE.NS", timeframe="1d", provider_id="yfinance")
    store.close()
    again = MarketStore(path)
    assert again.migrate() == 1
    assert again.stats()["candles"] == 3
    assert again.stats()["schema_version"] == 1


def test_duplicate_bars_update_in_place(tmp_path):
    store = MarketStore(tmp_path / "market.sqlite")
    store.upsert_candles(_bars(4), symbol="TCS.NS", timeframe="1d", provider_id="yfinance")
    revised = _bars(4, close_at={1: 250})
    revised.loc[1, "Open"] = 250
    revised.loc[1, "High"] = 251
    revised.loc[1, "Low"] = 249
    store.upsert_candles(revised, symbol="TCS.NS", timeframe="1d", provider_id="yfinance")
    stored = store.read_candles("TCS.NS", "1d")
    assert len(stored) == 4
    assert stored["Close"].iloc[1] == pytest.approx(250)
    assert find_duplicate_timestamps(stored) == []


def test_incremental_range_does_not_duplicate_overlap(tmp_path):
    store = MarketStore(tmp_path / "market.sqlite")
    store.upsert_candles(_bars(5), symbol="INFY.NS", timeframe="1d", provider_id="yfinance")
    extra_dates = pd.bdate_range("2024-01-05", periods=4)
    close = [104, 105, 106, 107]
    extra = pd.DataFrame(
        {
            "Date": extra_dates,
            "Open": close,
            "High": [c + 1 for c in close],
            "Low": [c - 1 for c in close],
            "Close": close,
            "Volume": [2_000] * 4,
        }
    )
    store.upsert_candles(extra, symbol="INFY.NS", timeframe="1d", provider_id="yfinance")
    stored = store.read_candles("INFY.NS", "1d")
    assert len(stored) == 7
    assert stored["Date"].is_monotonic_increasing
    assert stored["Close"].iloc[-1] == pytest.approx(107)


def test_invalid_ohlc_is_rejected_and_recorded(tmp_path):
    store = MarketStore(tmp_path / "market.sqlite")
    frame = _bars(2)
    frame.loc[1, "High"] = 1
    frame.loc[1, "Low"] = 50
    result = store.upsert_candles(frame, symbol="HDFCBANK.NS", timeframe="1d", provider_id="yfinance")
    assert result["written"] == 1
    assert result["rejected"] == 1
    assert store.stats()["candles"] == 1
    events = store.quality_events()
    assert "high_below_low" in set(events["check_name"])


def test_missing_weekday_and_stale_quote_detection():
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]),
            "Close": [1, 2, 3],
        }
    )
    missing = find_missing_sessions(frame, "1d")
    assert pd.Timestamp("2024-01-04") in missing
    intraday = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02 09:15", "2024-01-02 09:16", "2024-01-02 09:18"]),
            "Close": [1, 2, 3],
        }
    )
    gaps = find_missing_sessions(intraday, "1m")
    assert pd.Timestamp("2024-01-02 09:17") in gaps
    now = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
    assert quote_is_stale(datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc), now, timedelta(minutes=30))
    assert not quote_is_stale(datetime(2024, 1, 2, 9, 50, tzinfo=timezone.utc), now, timedelta(minutes=30))


def test_simulated_quote_cannot_be_labeled_live(tmp_path):
    store = MarketStore(tmp_path / "market.sqlite")
    with pytest.raises(ValueError, match="live"):
        Quote(
            symbol="RELIANCE.NS",
            quote_time="2024-01-02T10:00:00+00:00",
            last_price=100,
            provider_id="simulated",
            freshness="live",
            live_authorized=False,
        )
    store.upsert_quotes(
        [
            Quote(
                symbol="RELIANCE.NS",
                quote_time="2024-01-02T10:00:00+00:00",
                last_price=100,
                provider_id="simulated",
                freshness="simulated",
            )
        ]
    )
    latest = store.latest_quotes(["RELIANCE.NS"])
    assert latest.iloc[0]["freshness"] == "simulated"
    assert store.stats()["quotes"] == 1

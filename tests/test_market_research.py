"""Stored history feeds the existing feature builder. Downloads in these tests are simulated."""

from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering import MultiTimeframeDatasetConfig, build_multi_timeframe_dataset
from market_data.research_loader import load_research_timeframes
from market_data.store import MarketStore


def _history(periods: int, start: str = "2023-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    close = pd.Series(range(periods), dtype=float) + 100
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000,
        }
    )


def test_loader_uses_stored_bars_when_the_provider_fails(tmp_path, monkeypatch):
    store = MarketStore(tmp_path / "market.sqlite")
    store.upsert_candles(_history(80), symbol="RELIANCE.NS", timeframe="1d", provider_id="yfinance")

    def _fail(*args, **kwargs):
        raise RuntimeError("simulated provider outage")

    monkeypatch.setattr("market_data.research_loader.download_stock_data", _fail)
    loaded = load_research_timeframes(
        "RELIANCE.NS",
        intervals=("1d",),
        store=store,
        as_of=pd.Timestamp("2024-12-31"),
    )
    assert len(loaded["1d"]) == 80
    assert list(loaded["1d"].columns)[:6] == ["Date", "Open", "High", "Low", "Close", "Volume"]


def test_incremental_request_starts_at_the_last_stored_bar_and_as_of_drops_the_future(tmp_path, monkeypatch):
    store = MarketStore(tmp_path / "market.sqlite")
    store.upsert_candles(_history(5), symbol="TCS.NS", timeframe="1d", provider_id="yfinance")
    seen = {}

    def _download(ticker, cfg=None):
        seen["start"] = cfg.start
        seen["ticker"] = ticker
        return _history(3, start="2023-01-06")

    monkeypatch.setattr("market_data.research_loader.download_stock_data", _download)
    loaded = load_research_timeframes(
        "TCS.NS",
        intervals=("1d",),
        store=store,
        as_of=pd.Timestamp("2023-01-09"),
    )
    assert seen["ticker"] == "TCS.NS"
    assert seen["start"] == "2023-01-06"
    assert loaded["1d"]["Date"].max() <= pd.Timestamp("2023-01-09")
    assert loaded["1d"]["Date"].max() < pd.Timestamp("2023-01-10")
    assert store.read_candles("TCS.NS", "1d")["Date"].is_monotonic_increasing
    assert store.read_candles("TCS.NS", "1d")["Date"].duplicated().sum() == 0


def test_stored_daily_bars_still_build_the_technical_feature_frame(tmp_path, monkeypatch):
    store = MarketStore(tmp_path / "market.sqlite")
    history = _history(120)
    store.upsert_candles(history, symbol="INFY.NS", timeframe="1d", provider_id="yfinance")
    monkeypatch.setattr(
        "market_data.research_loader.download_stock_data",
        lambda *args, **kwargs: history.iloc[0:0],
    )
    loaded = load_research_timeframes("INFY.NS", intervals=("1d",), store=store, as_of=pd.Timestamp("2024-12-31"))
    model_df, columns = build_multi_timeframe_dataset(
        {"1d": loaded["1d"]},
        cfg=MultiTimeframeDatasetConfig(
            base_interval="1d",
            context_intervals=(),
            horizon_bars=2,
            technical_only=True,
            technical_min_forward_return=0.0,
        ),
    )
    assert "Target" in model_df.columns
    assert model_df["Date"].is_monotonic_increasing
    assert columns
    assert model_df["Target"].isna().sum() >= 2

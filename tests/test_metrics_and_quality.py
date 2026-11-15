"""Annualization, performance fields, and OHLCV integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import BacktestConfig, backtest_buy_and_hold, backtest_long_cash
from backtest_analytics import build_full_analytics, infer_periods_per_year
from data_loader import sanitize_ohlcv
from transaction_costs import IndiaEquityCostConfig


def test_annualization_matches_bar_spacing():
    daily = pd.bdate_range("2020-01-01", periods=40)
    weekly = pd.date_range("2020-01-03", periods=20, freq="W-FRI")
    hourly = pd.date_range("2024-01-02 09:15", periods=40, freq="h")
    assert infer_periods_per_year(daily) == pytest.approx(252.0)
    assert infer_periods_per_year(weekly) == pytest.approx(52.0)
    assert infer_periods_per_year(hourly) > 252.0


def test_backtest_reports_gross_net_and_buy_and_hold():
    dates = pd.bdate_range("2024-01-02", periods=6)
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": [10, 10, 11, 12, 13, 14],
            "High": [10.5, 11, 12, 13, 14, 15],
            "Low": [9.5, 9.5, 10.5, 11.5, 12.5, 13.5],
            "Close": [10.2, 10.8, 11.5, 12.4, 13.2, 14.1],
            "Signal": ["BUY", "HOLD", "HOLD", "SELL", "HOLD", "HOLD"],
            "Volume": np.full(6, 1_000_000.0),
        }
    )
    cfg = BacktestConfig(initial_capital=10_000.0, apply_costs=True)
    strategy = backtest_long_cash(df, cfg)
    bench = backtest_buy_and_hold(df, cfg)
    assert strategy["net_return_pct"] == pytest.approx(strategy["total_return_pct"])
    assert strategy["gross_return_pct"] > strategy["net_return_pct"]
    assert np.isfinite(strategy["sharpe"])
    assert np.isfinite(strategy["cagr"])
    assert strategy["periods_per_year"] == pytest.approx(252.0)
    assert bench["fill_count"] == 1
    assert bench["final_portfolio_value"] != strategy["final_portfolio_value"]
    analytics = build_full_analytics(strategy, initial_capital=10_000.0, benchmark=bench)
    assert "gross_return_pct" in analytics["performance"]
    assert "net_return_pct" in analytics["performance"]
    assert np.isfinite(analytics["benchmark"]["bh_total_return_pct"])


def test_invalid_ohlc_bars_are_dropped():
    df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-02", periods=3),
            "Open": [10.0, 10.0, 10.0],
            "High": [11.0, 9.0, 11.0],
            "Low": [9.0, 8.0, 9.0],
            "Close": [10.5, 10.5, 10.5],
            "Volume": [100.0, 100.0, -5.0],
        }
    )
    cleaned = sanitize_ohlcv(df)
    assert len(cleaned) == 1
    assert cleaned.attrs["dropped_invalid_bars"] == 2
    assert float(cleaned.iloc[0]["Close"]) == pytest.approx(10.5)

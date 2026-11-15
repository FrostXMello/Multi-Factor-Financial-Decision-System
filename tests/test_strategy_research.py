"""Offline M7 analytics tests. Fixtures are synthetic historical series, not live data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_research.portfolio import portfolio_analytics
from strategy_research.regimes import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS, classify_regimes, evaluate_returns_by_regime
from strategy_research.robustness import walk_forward_robustness
from strategy_research.strategy_compare import compare_strategies


def _price_path(n: int = 120, start: float = 100.0, drift: float = 0.002) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = [start]
    for _ in range(1, n):
        close.append(close[-1] * (1.0 + drift))
    return pd.DataFrame({"Date": dates, "Close": close})


def test_regime_classification_and_no_lookahead_for_returns():
    # Strong uptrend should eventually label bull once lookback is filled.
    prices = _price_path(100, drift=0.01)
    regimes = classify_regimes(prices, lookback=20, bull_threshold=0.05, bear_threshold=-0.05)
    assert regimes.attrs["lookahead_safe"] is True
    assert regimes["regime_end_of_day"].iloc[:20].eq("insufficient_data").all()
    assert regimes["regime_end_of_day"].iloc[-1] == REGIME_BULL
    # Causal label used for return t equals end-of-day regime from t-1.
    assert regimes["regime_for_return"].iloc[21] == regimes["regime_end_of_day"].iloc[20]
    assert regimes["regime_for_return"].iloc[0] == "insufficient_data"

    # Downtrend
    down = _price_path(100, drift=-0.01)
    bear = classify_regimes(down, lookback=20, bull_threshold=0.05, bear_threshold=-0.05)
    assert bear["regime_end_of_day"].iloc[-1] == REGIME_BEAR

    # Flat path -> sideways after warm-up
    flat = _price_path(80, drift=0.0)
    side = classify_regimes(flat, lookback=20, bull_threshold=0.05, bear_threshold=-0.05)
    assert side["regime_end_of_day"].iloc[-1] == REGIME_SIDEWAYS


def test_evaluate_returns_by_regime_marks_insufficient_samples():
    idx = pd.bdate_range("2024-01-02", periods=10)
    returns = pd.Series(np.linspace(-0.01, 0.01, 10), index=idx)
    regimes = pd.Series(
        ["bull", "bull", "bear", "sideways", "sideways", "sideways", "sideways", "sideways", "sideways", "sideways"],
        index=idx,
    )
    out = evaluate_returns_by_regime(returns, regimes, min_observations=5)
    assert out["by_regime"]["bull"]["available"] is False
    assert out["by_regime"]["sideways"]["available"] is True
    assert "insufficient" in out["weaknesses"][0]


def test_portfolio_allocation_concentration_and_unavailable_sector():
    result = portfolio_analytics(
        [
            {"symbol": "TCS.NS", "quantity": 10, "mark": 100.0},
            {"symbol": "HDFCBANK.NS", "quantity": 10, "mark": 100.0},
            {"symbol": "UNKNOWN.ZZ", "quantity": 5, "mark": 100.0},
        ],
        cash=500.0,
    )
    assert result["available"] is True
    assert result["nav"] == pytest.approx(3000.0)
    weights = {row["symbol"]: row["weight"] for row in result["allocation"]}
    assert weights["CASH"] == pytest.approx(500.0 / 3000.0)
    assert result["concentration"]["hhi"] is not None
    assert "UNKNOWN.ZZ" in result["unavailable_sectors"]
    sectors = {row["sector"] for row in result["sector_exposure"]}
    assert "IT" in sectors
    assert "BANK" in sectors


def test_strategy_comparison_uses_net_returns_after_costs():
    dates = pd.bdate_range("2024-01-02", periods=30)
    eq_a = pd.DataFrame({"Date": dates, "PortfolioValue": np.linspace(100_000, 110_000, 30)})
    eq_b = pd.DataFrame({"Date": dates, "PortfolioValue": np.linspace(100_000, 105_000, 30)})
    compared = compare_strategies(
        {
            "alpha": {
                "equity_curve": eq_a,
                "net_return_pct": 8.0,
                "gross_return_pct": 10.0,
                "total_transaction_costs": 200.0,
                "max_drawdown_pct": -5.0,
                "final_portfolio_value": 110_000,
            },
            "beta": {
                "equity_curve": eq_b,
                "net_return_pct": 3.0,
                "gross_return_pct": 4.0,
                "total_transaction_costs": 50.0,
                "max_drawdown_pct": -2.0,
                "final_portfolio_value": 105_000,
            },
            "empty": {},
        }
    )
    assert compared["leader"] == "alpha"
    assert compared["ranking_by_net_return"][0] == "alpha"
    assert any("empty" in w for w in compared["weaknesses"])


def test_walk_forward_robustness_is_diagnostic_only():
    weak = walk_forward_robustness(fold_scores=[0.7])
    assert weak["available"] is False
    strong = walk_forward_robustness(fold_scores=[0.62, 0.58, 0.60, 0.55, 0.63])
    assert strong["available"] is True
    assert strong["mean"] == pytest.approx(np.mean([0.62, 0.58, 0.60, 0.55, 0.63]))
    assert "no parameter search" in strong["note"].lower()

"""Next-open fills, risk exits, and Indian equity cost assumptions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import BacktestConfig, backtest_long_cash
from risk_model import apply_risk_gating
from transaction_costs import (
    IndiaEquityCostConfig,
    buy_charges,
    cost_config_for_segment,
    sell_charges,
)


def _frame(opens, closes, signals) -> pd.DataFrame:
    n = len(opens)
    dates = pd.bdate_range("2024-01-02", periods=n)
    open_ = np.asarray(opens, dtype=float)
    close = np.asarray(closes, dtype=float)
    high = np.nanmax(np.vstack([open_, close]), axis=0) + 0.1
    low = np.nanmin(np.vstack([np.where(np.isfinite(open_), open_, close), close]), axis=0) - 0.1
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Signal": signals,
            "PositionSize": np.ones(n),
        }
    )


def test_signal_fills_at_the_next_open_not_the_signal_close():
    df = _frame([10, 20, 30, 40], [11, 21, 31, 41], ["BUY", "HOLD", "SELL", "HOLD"])
    bt = backtest_long_cash(
        df,
        BacktestConfig(initial_capital=1000.0, apply_costs=False, cost_config=IndiaEquityCostConfig.zero()),
    )
    trades = bt["trades_df"]
    assert len(trades) == 1
    assert trades.iloc[0]["EntryPx"] == pytest.approx(20.0)
    assert trades.iloc[0]["ExitPx"] == pytest.approx(40.0)
    assert trades.iloc[0]["EntryPx"] != pytest.approx(11.0)
    assert trades.iloc[0]["ExitPx"] != pytest.approx(31.0)
    assert bt["final_portfolio_value"] == pytest.approx(2000.0)
    assert bt["gross_return_pct"] == pytest.approx(bt["net_return_pct"])


def test_last_bar_signal_is_not_filled_and_missing_open_is_skipped():
    held = _frame([10, 20, 30], [11, 21, 31], ["BUY", "HOLD", "SELL"])
    bt = backtest_long_cash(
        held,
        BacktestConfig(initial_capital=1000.0, apply_costs=False, cost_config=IndiaEquityCostConfig.zero()),
    )
    assert bt["trades_df"].empty
    assert bt["fill_count"] == 1  # the buy filled; the terminal sell did not
    assert bt["final_portfolio_value"] == pytest.approx(50.0 * 31.0)

    skipped = _frame([10, np.nan, 30, 40], [11, 21, 31, 41], ["BUY", "HOLD", "HOLD", "HOLD"])
    missed = backtest_long_cash(
        skipped,
        BacktestConfig(initial_capital=1000.0, apply_costs=False, cost_config=IndiaEquityCostConfig.zero()),
    )
    assert missed["fill_count"] == 0
    assert missed["final_portfolio_value"] == pytest.approx(1000.0)

    lone = _frame([10], [11], ["BUY"])
    one = backtest_long_cash(
        lone,
        BacktestConfig(initial_capital=1000.0, apply_costs=False, cost_config=IndiaEquityCostConfig.zero()),
    )
    assert one["fill_count"] == 0
    assert one["final_portfolio_value"] == pytest.approx(1000.0)


def test_costs_reduce_net_equity_versus_gross():
    df = _frame([10, 20, 30, 40], [11, 21, 31, 41], ["BUY", "HOLD", "SELL", "HOLD"])
    bt = backtest_long_cash(df, BacktestConfig(initial_capital=100_000.0, apply_costs=True))
    assert bt["total_transaction_costs"] > 0
    assert bt["final_portfolio_value"] < bt["gross_final_portfolio_value"]
    assert bt["net_return_pct"] < bt["gross_return_pct"]
    assert bt["total_return_pct"] == pytest.approx(bt["net_return_pct"])
    assert bt["trade_count"] == 1
    assert bt["win_rate"] == pytest.approx(1.0)


def test_delivery_cost_components_match_the_documented_formula():
    cfg = cost_config_for_segment("equity_delivery")
    notional = 100_000.0
    brokerage = notional * cfg.brokerage_rate
    exchange = notional * cfg.exchange_rate
    sebi = notional * cfg.sebi_rate
    gst = cfg.gst_rate * (brokerage + exchange + sebi)
    buy = brokerage + notional * cfg.stt_buy_rate + exchange + sebi + notional * cfg.stamp_buy_rate + gst
    sell = brokerage + notional * cfg.stt_sell_rate + exchange + sebi + notional * cfg.stamp_sell_rate + gst
    assert buy_charges(notional, cfg) == pytest.approx(buy)
    assert sell_charges(notional, cfg) == pytest.approx(sell)
    assert cfg.stamp_sell_rate == 0.0
    intra = IndiaEquityCostConfig.equity_intraday()
    assert intra.stt_buy_rate == 0.0
    assert intra.stt_sell_rate > 0


def test_risk_gate_blocks_entries_and_still_allows_exits():
    dates = pd.bdate_range("2024-01-02", periods=4)
    signals = _frame([10, 20, 30, 40], [11, 21, 31, 41], ["BUY", "HOLD", "SELL", "HOLD"])
    risk = pd.DataFrame(
        {
            "Date": dates,
            "RealizedVol": [0.01, 0.01, 0.2, 0.01],
            "RollingDrawdown": [-0.01, -0.01, -0.3, -0.01],
            "RiskScore": [0.1, 0.1, 0.9, 0.1],
            "NoTrade": [False, False, True, False],
            "PositionSize": [1.0, 1.0, 1.0, 1.0],
        }
    )
    gated = apply_risk_gating(signals, risk)
    assert gated.iloc[0]["Signal"] == "BUY"
    assert gated.iloc[2]["Signal"] == "SELL"
    assert bool(gated.iloc[2]["EntryBlocked"]) is False

    blocked = risk.copy()
    blocked["NoTrade"] = True
    gated_all = apply_risk_gating(signals, blocked)
    assert gated_all.iloc[0]["Signal"] == "HOLD"
    assert bool(gated_all.iloc[0]["EntryBlocked"]) is True
    assert gated_all.iloc[2]["Signal"] == "SELL"

    closed = backtest_long_cash(
        gated,
        BacktestConfig(initial_capital=1000.0, apply_costs=False, cost_config=IndiaEquityCostConfig.zero()),
    )
    assert len(closed["trades_df"]) == 1
    assert closed["trades_df"].iloc[0]["ExitPx"] == pytest.approx(40.0)

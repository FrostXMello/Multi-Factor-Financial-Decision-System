"""
Walk-forward backtest for the multi-layer selector.

At each rebalance (default: month-end), re-run Layer 1→2→3 on data **only up to that date**,
then hold an equal-weight basket until the next rebalance. Benchmark = equal-weight buy & hold
of the full downloaded universe on the same calendar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from multi_layer_config import MULTI_LAYER_MAX_REBALANCE_POINTS, MULTI_LAYER_WARMUP_BARS, MultiLayerPipelineConfig
from multi_layer_data import truncate_prices_asof
from multi_layer_pipeline import run_multi_layer_pipeline


def build_close_panel(prices_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide Close matrix, union calendar, forward-filled."""
    series_list: list[pd.Series] = []
    for sym, df in prices_by_ticker.items():
        if df is None or df.empty:
            continue
        s = df.set_index(pd.to_datetime(df["Date"]))["Close"].astype(float).sort_index()
        s.name = sym
        series_list.append(s)
    if not series_list:
        return pd.DataFrame()
    panel = pd.concat(series_list, axis=1).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    return panel.ffill()


def _month_end_dates(index: pd.DatetimeIndex, min_pos: int) -> list[pd.Timestamp]:
    if len(index) <= min_pos:
        return []
    s = pd.Series(np.arange(len(index)), index=index)
    month = pd.DatetimeIndex(index).to_period("M")
    last_pos = s.groupby(month).max()
    dates = [index[int(i)] for i in last_pos.values if int(i) >= min_pos]
    return sorted(set(dates))


def _portfolio_bt_dict(
    equity_curve: pd.DataFrame,
    *,
    initial_capital: float,
    trades_df: pd.DataFrame,
    exposure_pct: float,
    total_signal_bars: int,
) -> dict:
    eq = equity_curve.copy()
    eq["Date"] = pd.to_datetime(eq["Date"])
    eq = eq.sort_values("Date").reset_index(drop=True)
    if eq.empty:
        raise ValueError("Empty equity curve.")
    final_value = float(eq["PortfolioValue"].iloc[-1])
    total_return_pct = (final_value / float(initial_capital) - 1.0) * 100.0
    v = eq["PortfolioValue"].astype(float)
    rolling_max = v.cummax()
    drawdown = v / rolling_max - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0)
    daily_ret = v.pct_change().dropna()
    sharpe = (
        float(np.sqrt(252) * daily_ret.mean() / daily_ret.std())
        if len(daily_ret) > 1 and daily_ret.std() > 0
        else float("nan")
    )
    return {
        "equity_curve": eq,
        "final_portfolio_value": final_value,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "trades_df": trades_df if isinstance(trades_df, pd.DataFrame) else pd.DataFrame(),
        "exposure_pct": float(exposure_pct),
        "total_signal_bars": int(total_signal_bars),
    }


def backtest_multi_layer_walk_forward(
    prices_by_ticker: dict[str, pd.DataFrame],
    pipe_cfg: MultiLayerPipelineConfig | None = None,
    *,
    initial_capital: float = 100_000.0,
    warmup_bars: int = MULTI_LAYER_WARMUP_BARS,
    max_rebalance_points: int | None = None,
    backtest_years: float | None = 1.0,
) -> dict:
    """
    Simulate the full multi-layer pipeline on a rolling calendar.

    Returns `portfolio_backtest` / `benchmark_backtest` dicts compatible with
    `backtest_analytics.build_full_analytics` and `render_portfolio_quant_dashboard`.

    ``backtest_years``:
        If set (e.g. 1.0), only month-end rebalance dates on or after
        (last panel date − that many calendar years) are simulated.
        If ``None``, uses the full rebalance schedule (still subject to ``max_rebalance_points``).
    """
    if pipe_cfg is None:
        pipe_cfg = MultiLayerPipelineConfig()
    panel = build_close_panel(prices_by_ticker)
    if panel.empty or len(panel) < warmup_bars + 30:
        raise ValueError("Not enough overlapping price history for multi-layer backtest.")

    max_pts = int(max_rebalance_points) if max_rebalance_points is not None else int(pipe_cfg.rebalance_max_points)
    reb_dates = _month_end_dates(panel.index, warmup_bars)
    if len(reb_dates) > max_pts:
        reb_dates = reb_dates[-max_pts:]

    if backtest_years is not None and float(backtest_years) > 0:
        last_d = pd.Timestamp(panel.index.max()).normalize()
        window_start = last_d - pd.DateOffset(years=float(backtest_years))
        filtered = [pd.Timestamp(d).normalize() for d in reb_dates if pd.Timestamp(d) >= window_start]
        if len(filtered) >= 2:
            reb_dates = filtered
        elif len(reb_dates) >= 2:
            reb_dates = reb_dates[-min(13, len(reb_dates)) :]

    if len(reb_dates) < 2:
        raise ValueError("Need at least two rebalance dates after warmup and window filter.")

    strategy_rows: list[dict] = []
    bench_rows: list[dict] = []
    rebalance_log: list[dict] = []
    trades_log: list[dict] = []

    all_syms = [c for c in panel.columns if panel[c].notna().any()]
    nav_s = float(initial_capital)
    nav_b = float(initial_capital)
    days_with_holdings = 0
    n_trading_days = 0

    for k in range(len(reb_dates) - 1):
        d0 = pd.Timestamp(reb_dates[k]).normalize()
        d1 = pd.Timestamp(reb_dates[k + 1]).normalize()
        trimmed = truncate_prices_asof(prices_by_ticker, d0, min_rows=200)
        try:
            out = run_multi_layer_pipeline(trimmed, cfg=pipe_cfg)
            picks = list(out["final_tickers"])
        except Exception:
            picks = []

        rebalance_log.append({"Date": d0, "Picks": picks, "N": len(picks)})

        mask = (panel.index > d0) & (panel.index <= d1)
        chunk_idx = panel.index[mask]
        if len(chunk_idx) == 0:
            continue

        if picks:
            sub = panel.loc[chunk_idx, [p for p in picks if p in panel.columns]]
            if sub.shape[1] == 0:
                r_strat = pd.Series(0.0, index=chunk_idx)
            else:
                r_strat = sub.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0.0)
        else:
            r_strat = pd.Series(0.0, index=chunk_idx)

        sub_b = panel.loc[chunk_idx, all_syms]
        r_bench = sub_b.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0.0)

        nav_start_seg = nav_s
        first_dt = chunk_idx[0]
        last_dt = chunk_idx[-1]
        for dt in chunk_idx:
            rsv = float(r_strat.loc[dt])
            rbv = float(r_bench.loc[dt])
            nav_s *= 1.0 + rsv
            nav_b *= 1.0 + rbv
            strategy_rows.append({"Date": dt, "PortfolioValue": nav_s})
            bench_rows.append({"Date": dt, "PortfolioValue": nav_b})

        if picks:
            days_with_holdings += len(chunk_idx)
        n_trading_days += len(chunk_idx)

        pnl = float(nav_s - nav_start_seg)
        pnl_pct = float((nav_s / nav_start_seg - 1.0) * 100.0) if nav_start_seg > 0 else 0.0
        trades_log.append(
            {
                "EntryDate": first_dt,
                "ExitDate": last_dt,
                "EntryCost": float(nav_start_seg),
                "Proceeds": float(nav_s),
                "PnL": pnl,
                "PnLPct": pnl_pct,
                "Win": bool(pnl > 0),
            }
        )

    if not strategy_rows:
        raise ValueError("Walk-forward produced no daily equity points (check rebalance window vs data).")

    eq_s = pd.DataFrame(strategy_rows)
    eq_b = pd.DataFrame(bench_rows)
    trades_df = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()

    exposure_pct = 100.0 * float(days_with_holdings) / float(n_trading_days) if n_trading_days else 0.0

    portfolio_bt = _portfolio_bt_dict(
        eq_s,
        initial_capital=initial_capital,
        trades_df=trades_df,
        exposure_pct=exposure_pct,
        total_signal_bars=n_trading_days,
    )
    benchmark_bt = _portfolio_bt_dict(
        eq_b,
        initial_capital=initial_capital,
        trades_df=pd.DataFrame(),
        exposure_pct=100.0,
        total_signal_bars=n_trading_days,
    )

    seg_wins = int(trades_df["Win"].sum()) if len(trades_df) > 0 and "Win" in trades_df.columns else 0
    seg_n = len(trades_df)
    win_rate = float(seg_wins / seg_n) if seg_n else float("nan")

    _ds = eq_s[["Date", "PortfolioValue"]].copy()
    _ds["Date"] = pd.to_datetime(_ds["Date"])
    daily_rs = _ds.set_index("Date").sort_index()["PortfolioValue"].astype(float).pct_change().dropna()

    return {
        "portfolio_backtest": portfolio_bt,
        "benchmark_backtest": benchmark_bt,
        "rebalance_log": rebalance_log,
        "rebalance_count": len(rebalance_log),
        "rebalance_segments": seg_n,
        "rebalance_win_rate": win_rate,
        "backtest_years_requested": backtest_years,
        "equity_strategy": portfolio_bt["equity_curve"],
        "equity_benchmark": benchmark_bt["equity_curve"],
        "total_return_pct_strategy": float(portfolio_bt["total_return_pct"]),
        "total_return_pct_benchmark": float(benchmark_bt["total_return_pct"]),
        "max_drawdown_pct_strategy": float(portfolio_bt["max_drawdown_pct"]),
        "max_drawdown_pct_benchmark": float(benchmark_bt["max_drawdown_pct"]),
        "sharpe_strategy": float(portfolio_bt["sharpe"]),
        "sharpe_benchmark": float(benchmark_bt["sharpe"]),
        "daily_returns_strategy": daily_rs,
    }

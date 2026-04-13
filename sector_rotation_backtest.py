"""
Walk-forward backtest: weekly rebalance, sector rank → top sectors → 1 stock each → equal weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import DownloadConfig, download_stock_data
from multi_layer_backtest import _portfolio_bt_dict
from sector_rotation_config import SectorRotationConfig
from sector_rotation_pipeline import run_sector_rotation_walk_forward_step


def _week_anchor_dates(index: pd.DatetimeIndex, min_pos: int) -> list[pd.Timestamp]:
    if len(index) <= min_pos:
        return []
    s = pd.Series(np.arange(len(index)), index=index)
    lab = pd.DatetimeIndex(index).to_period("W-FRI")
    last_pos = s.groupby(lab).max()
    dates = [index[int(i)] for i in last_pos.values if int(i) >= min_pos]
    return sorted(set(dates))


def build_close_panel(prices_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series_list: list[pd.Series] = []
    for sym, df in prices_by_ticker.items():
        if df is None or df.empty:
            continue
        ser = df.set_index(pd.to_datetime(df["Date"]))["Close"].astype(float).sort_index()
        ser.name = sym
        series_list.append(ser)
    if not series_list:
        return pd.DataFrame()
    panel = pd.concat(series_list, axis=1).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    return panel.ffill()


def backtest_sector_rotation_walk_forward(
    sector_index_frames: dict[str, pd.DataFrame],
    stock_prices_full: dict[str, pd.DataFrame],
    cfg: SectorRotationConfig | None = None,
    *,
    initial_capital: float = 100_000.0,
    benchmark_ticker: str = "^NSEI",
) -> dict:
    """
    Simulate weekly rebalance; returns portfolio_backtest / benchmark_backtest compatible dicts.
    """
    if cfg is None:
        cfg = SectorRotationConfig()
    panel = build_close_panel(stock_prices_full)
    if panel.empty or len(panel) < cfg.warmup_bars + 20:
        raise ValueError("Not enough stock history for sector rotation backtest.")

    max_pts = int(cfg.rebalance_max_points)
    reb_dates = _week_anchor_dates(panel.index, cfg.warmup_bars)
    if len(reb_dates) > max_pts:
        reb_dates = reb_dates[-max_pts:]

    if cfg.backtest_years is not None and float(cfg.backtest_years) > 0:
        last_d = pd.Timestamp(panel.index.max()).normalize()
        window_start = last_d - pd.DateOffset(years=float(cfg.backtest_years))
        filtered = [pd.Timestamp(d).normalize() for d in reb_dates if pd.Timestamp(d) >= window_start]
        if len(filtered) >= 2:
            reb_dates = filtered
        elif len(reb_dates) >= 2:
            reb_dates = reb_dates[-min(55, len(reb_dates)) :]

    if len(reb_dates) < 2:
        raise ValueError("Need at least two weekly rebalance dates after warmup.")

    bench_df: pd.DataFrame | None = None
    bench_daily_ret: pd.Series | None = None
    try:
        bench_df = download_stock_data(
            benchmark_ticker,
            cfg=DownloadConfig(period="max", interval="1d", auto_adjust=False),
        )
        b = bench_df.copy()
        b["Date"] = pd.to_datetime(b["Date"])
        bc = b.sort_values("Date").set_index("Date")["Close"].astype(float)
        bench_daily_ret = bc.reindex(panel.index, method="ffill").pct_change().fillna(0.0)
    except Exception:
        bench_df = None
        bench_daily_ret = None

    strategy_rows: list[dict] = []
    bench_rows: list[dict] = []
    trades_log: list[dict] = []
    rebalance_meta: list[dict] = []

    nav_s = float(initial_capital)
    nav_b = float(initial_capital)
    n_days_invested = 0
    n_days_total = 0

    for k in range(len(reb_dates) - 1):
        d0 = pd.Timestamp(reb_dates[k]).normalize()
        d1 = pd.Timestamp(reb_dates[k + 1]).normalize()
        snap: dict = {}
        try:
            snap = run_sector_rotation_walk_forward_step(
                sector_index_frames,
                stock_prices_full,
                d0,
                cfg,
                min_rows=120,
            )
        except Exception:
            snap = {}
        picks = list(snap.get("final_tickers") or [])
        rebalance_meta.append(
            {
                "Date": d0,
                "Picks": picks,
                "N": len(picks),
                "Sectors": list(snap.get("selected_sectors") or []),
            }
        )

        mask = (panel.index > d0) & (panel.index <= d1)
        chunk_idx = panel.index[mask]
        if len(chunk_idx) == 0:
            continue

        if picks:
            sub = panel.loc[chunk_idx, [p for p in picks if p in panel.columns]]
            if sub.shape[1] == 0:
                r_strat = pd.Series(0.0, index=chunk_idx)
            else:
                r_strat = sub.pct_change().mean(axis=1, skipna=True).fillna(0.0)
        else:
            r_strat = pd.Series(0.0, index=chunk_idx)

        if bench_daily_ret is not None:
            r_bench = bench_daily_ret.loc[chunk_idx]
        else:
            r_bench = pd.Series(0.0, index=chunk_idx)

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
            n_days_invested += len(chunk_idx)
        n_days_total += len(chunk_idx)

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
        raise ValueError("Sector rotation backtest produced no daily points.")

    eq_s = pd.DataFrame(strategy_rows)
    eq_b = pd.DataFrame(bench_rows)
    trades_df = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()
    exposure_pct = 100.0 * float(n_days_invested) / float(n_days_total) if n_days_total else 0.0

    portfolio_bt = _portfolio_bt_dict(
        eq_s,
        initial_capital=initial_capital,
        trades_df=trades_df,
        exposure_pct=exposure_pct,
        total_signal_bars=n_days_total,
    )
    benchmark_bt = _portfolio_bt_dict(
        eq_b,
        initial_capital=initial_capital,
        trades_df=pd.DataFrame(),
        exposure_pct=100.0,
        total_signal_bars=n_days_total,
    )

    seg_wins = int(trades_df["Win"].sum()) if len(trades_df) > 0 and "Win" in trades_df.columns else 0
    seg_n = len(trades_df)
    win_rate = float(seg_wins / seg_n) if seg_n else float("nan")

    return {
        "portfolio_backtest": portfolio_bt,
        "benchmark_backtest": benchmark_bt,
        "rebalance_log": rebalance_meta,
        "rebalance_count": len(rebalance_meta),
        "rebalance_segments": seg_n,
        "rebalance_win_rate": win_rate,
        "equity_strategy": portfolio_bt["equity_curve"],
        "equity_benchmark": benchmark_bt["equity_curve"],
        "total_return_pct_strategy": float(portfolio_bt["total_return_pct"]),
        "total_return_pct_benchmark": float(benchmark_bt["total_return_pct"]),
    }

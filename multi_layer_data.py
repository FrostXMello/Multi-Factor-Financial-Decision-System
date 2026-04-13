"""
Download and quality-filter OHLCV for a large symbol universe.

Yahoo Finance is used (same as `data_loader`). Failures are skipped so scans stay robust.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_loader import DownloadConfig, download_stock_data
from multi_layer_config import MIN_AVG_DOLLAR_VOLUME_20D, MIN_OHLCV_ROWS


@dataclass(frozen=True)
class UniverseDownloadConfig:
    period: str = "5y"
    interval: str = "1d"
    min_rows: int = MIN_OHLCV_ROWS
    min_avg_dollar_volume_20d: float = MIN_AVG_DOLLAR_VOLUME_20D
    sleep_sec: float = 0.03  # light pacing for Yahoo
    auto_adjust: bool = False


def _avg_dollar_volume_20d(df: pd.DataFrame) -> float:
    if df is None or len(df) < 25:
        return 0.0
    x = df.tail(25).copy()
    dv = (x["Close"].astype(float) * x["Volume"].astype(float)).tail(20)
    return float(dv.mean()) if len(dv) else 0.0


def prefilter_stock_quality(df: pd.DataFrame, cfg: UniverseDownloadConfig | None = None) -> bool:
    """
    Pre-Layer-1 gate: enough history and not chronically illiquid (20d avg Close*Volume).
    """
    if cfg is None:
        cfg = UniverseDownloadConfig()
    if df is None or len(df) < cfg.min_rows:
        return False
    if _avg_dollar_volume_20d(df) < cfg.min_avg_dollar_volume_20d:
        return False
    if df["Volume"].tail(60).astype(float).mean() <= 0:
        return False
    return True


def download_universe_daily(
    tickers: list[str],
    cfg: UniverseDownloadConfig | None = None,
    *,
    progress_callback=None,
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for every ticker; skip symbols that error or fail quality prefilter.

    progress_callback: optional callable(done: int, total: int, ticker: str)
    """
    if cfg is None:
        cfg = UniverseDownloadConfig()
    cleaned = [t.strip().upper() for t in tickers if t and str(t).strip()]
    cleaned = list(dict.fromkeys(cleaned))
    out: dict[str, pd.DataFrame] = {}
    n = len(cleaned)
    for i, sym in enumerate(cleaned):
        if progress_callback:
            progress_callback(i + 1, n, sym)
        try:
            df = download_stock_data(
                sym,
                cfg=DownloadConfig(period=cfg.period, interval=cfg.interval, auto_adjust=cfg.auto_adjust),
            )
        except Exception:
            time.sleep(cfg.sleep_sec)
            continue
        if not prefilter_stock_quality(df, cfg):
            time.sleep(cfg.sleep_sec)
            continue
        out[sym] = df.reset_index(drop=True)
        time.sleep(cfg.sleep_sec)
    return out


def truncate_prices_asof(
    prices_by_ticker: dict[str, pd.DataFrame],
    asof: pd.Timestamp,
    *,
    min_rows: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Cut each series to dates <= asof (for walk-forward / backtest)."""
    a = pd.Timestamp(asof).normalize()
    need = int(min_rows) if min_rows is not None else int(MIN_OHLCV_ROWS)
    trimmed: dict[str, pd.DataFrame] = {}
    for t, df in prices_by_ticker.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        d["Date"] = pd.to_datetime(d["Date"])
        sub = d[d["Date"] <= a].reset_index(drop=True)
        if len(sub) >= need:
            trimmed[t] = sub
    return trimmed

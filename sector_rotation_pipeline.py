"""
Sector → stock selection pipeline: rank sectors from index momentum/vol, then 1 ML pick per sector.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from data_loader import DownloadConfig, download_stock_data
from feature_engineering import MultiTimeframeDatasetConfig, build_multi_timeframe_dataset
from model import ModelConfig, train_model
from multi_layer_data import truncate_prices_asof
from sector_rotation_config import SectorRotationConfig
from sector_universe import SECTOR_DISPLAY_NAME, SECTOR_INDEX_YAHOO, STOCKS_BY_SECTOR


def download_sector_index_ohlcv(
    cfg: SectorRotationConfig | None = None,
    *,
    period: str = "5y",
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for each configured sector index."""
    if cfg is None:
        cfg = SectorRotationConfig()
    out: dict[str, pd.DataFrame] = {}
    dc = DownloadConfig(period=period, interval="1d", auto_adjust=False)
    for key, ysym in SECTOR_INDEX_YAHOO.items():
        try:
            df = download_stock_data(ysym, cfg=dc)
            out[key] = df.reset_index(drop=True)
        except Exception:
            continue
        time.sleep(0.05)
    return out


def _closes_to_series(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    return d.set_index("Date")["Close"].astype(float).sort_index()


def sector_engine(
    sector_index_frames: dict[str, pd.DataFrame],
    cfg: SectorRotationConfig | None = None,
    *,
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Rank sectors by risk-adjusted momentum score:
    Score = (w5 * 5d return + w10 * 10d return) / (rolling daily vol + eps)
    """
    if cfg is None:
        cfg = SectorRotationConfig()
    w5 = float(cfg.sector_score_ret5_weight)
    w10 = float(cfg.sector_score_ret10_weight)
    vw = int(cfg.sector_vol_window)
    eps = 1e-8
    rows: list[dict[str, Any]] = []
    a = pd.Timestamp(asof).normalize() if asof is not None else None

    for key, df in sector_index_frames.items():
        if df is None or df.empty:
            continue
        s = _closes_to_series(df)
        if a is not None:
            s = s.loc[:a]
        if len(s) < max(15, vw + 2):
            continue
        c = s.dropna()
        r5 = float(c.iloc[-1] / c.iloc[-6] - 1.0) if len(c) >= 6 else float("nan")
        r10 = float(c.iloc[-1] / c.iloc[-11] - 1.0) if len(c) >= 11 else float("nan")
        rets = c.pct_change().dropna()
        vol = float(rets.tail(vw).std()) if len(rets) >= vw else float("nan")
        if not np.isfinite(vol) or vol <= 0:
            continue
        raw = w5 * r5 + w10 * r10
        score = raw / (vol + eps)
        rows.append(
            {
                "SectorKey": key,
                "SectorName": SECTOR_DISPLAY_NAME.get(key, key),
                "IndexSymbol": SECTOR_INDEX_YAHOO.get(key, ""),
                "Return5d": r5,
                "Return10d": r10,
                "VolDaily": vol,
                "RawMomentum": raw,
                "SectorScore": score,
            }
        )

    if not rows:
        return pd.DataFrame()
    tab = pd.DataFrame(rows).sort_values("SectorScore", ascending=False).reset_index(drop=True)
    tab["Rank"] = np.arange(1, len(tab) + 1)
    return tab


def get_sector_stocks(sector_keys: list[str], *, max_per_sector: int = 10) -> dict[str, list[str]]:
    """Predefined liquid universe per sector (capped)."""
    out: dict[str, list[str]] = {}
    cap = max(5, int(max_per_sector))
    for k in sector_keys:
        lst = STOCKS_BY_SECTOR.get(str(k).upper(), [])
        out[k] = lst[:cap]
    return out


def feature_engineering_stock(df: pd.DataFrame, cfg: SectorRotationConfig | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Technical-only daily features + binary target (forward return > threshold)."""
    if cfg is None:
        cfg = SectorRotationConfig()
    return build_multi_timeframe_dataset(
        {"1d": df},
        cfg=MultiTimeframeDatasetConfig(
            base_interval="1d",
            context_intervals=(),
            horizon_bars=int(cfg.horizon_bars),
            technical_only=True,
            technical_min_forward_return=float(cfg.min_forward_return),
            roc_period=int(cfg.roc_period),
        ),
    )


def model_training(model_df: pd.DataFrame, feat_cols: list[str], cfg: SectorRotationConfig | None = None):
    """Single classifier (GB or RF), chronological split, predict_proba on test tail."""
    if cfg is None:
        cfg = SectorRotationConfig()
    return train_model(
        model_df,
        cfg=ModelConfig(
            model_type=str(cfg.model_type).lower(),
            min_rows=int(cfg.min_rows_model),
            random_state=42,
        ),
        feature_cols=feat_cols,
    )


def _recent_max_drawdown(close: pd.Series) -> float:
    c = close.dropna().astype(float)
    if len(c) < 10:
        return 0.0
    peak = c.cummax()
    dd = (c / peak - 1.0).min()
    return float(dd)


def risk_allows_stock(df: pd.DataFrame, cfg: SectorRotationConfig) -> bool:
    """Block names in deep recent drawdown (simple path risk)."""
    lb = int(cfg.risk_lookback_days)
    c = df["Close"].astype(float).tail(lb)
    if len(c) < 15:
        return True
    dd = _recent_max_drawdown(c)
    return dd >= -float(cfg.max_recent_drawdown)


def select_stock_per_sector(
    sector_key: str,
    tickers: list[str],
    prices_by_ticker: dict[str, pd.DataFrame],
    cfg: SectorRotationConfig | None = None,
) -> dict[str, Any] | None:
    """
    Train/score each name; keep those with Close > MA50 on latest test row and risk OK.
    Return best by probability.
    """
    if cfg is None:
        cfg = SectorRotationConfig()
    candidates: list[dict[str, Any]] = []
    for sym in tickers:
        df = prices_by_ticker.get(sym)
        if df is None or len(df) < cfg.min_rows_model:
            continue
        if not risk_allows_stock(df, cfg):
            continue
        try:
            model_df, feat_cols = feature_engineering_stock(df, cfg)
            if len(model_df) < cfg.min_rows_model:
                continue
            res = model_training(model_df, feat_cols, cfg)
            last = res.test_df.iloc[-1]
            px = float(last["Close"])
            ma50 = float(last["MA50"])
            if not (np.isfinite(px) and np.isfinite(ma50) and px > ma50):
                continue
            vol10 = float(last.get("Rolling_Volatility_10", np.nan))
            if np.isfinite(vol10) and vol10 > float(cfg.max_rolling_vol_10):
                continue
            candidates.append(
                {
                    "Ticker": sym,
                    "SectorKey": sector_key,
                    "Proba": float(res.test_proba[-1]),
                    "Close": px,
                    "MA50": ma50,
                    "TestAccuracy": float(res.test_accuracy),
                }
            )
        except Exception:
            continue

    if not candidates:
        return None
    best = max(candidates, key=lambda x: x["Proba"])
    return {**best, "Candidates": candidates}


def portfolio_construction(selections: list[dict[str, Any]]) -> tuple[list[str], pd.DataFrame]:
    """Equal weight across 2–3 names (one per sector)."""
    picks = [s["Ticker"] for s in selections if s and s.get("Ticker")]
    picks = list(dict.fromkeys(picks))
    n = len(picks)
    w = 100.0 / n if n else 0.0
    tab = pd.DataFrame(
        [
            {
                "Ticker": s["Ticker"],
                "SectorKey": s.get("SectorKey", ""),
                "WeightPct": w,
                "Proba": s.get("Proba", float("nan")),
            }
            for s in selections
            if s and s.get("Ticker")
        ]
    )
    return picks, tab


def run_sector_rotation_snapshot(
    sector_index_frames: dict[str, pd.DataFrame],
    stock_prices: dict[str, pd.DataFrame],
    cfg: SectorRotationConfig | None = None,
    *,
    asof: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """
    Full snapshot: sector ranks → top K sectors → best stock per sector → equal-weight book.
    """
    if cfg is None:
        cfg = SectorRotationConfig()
    ranked = sector_engine(sector_index_frames, cfg, asof=asof)
    if ranked.empty:
        return {
            "sector_rank_table": ranked,
            "selected_sectors": [],
            "selections": [],
            "final_tickers": [],
            "portfolio_table": pd.DataFrame(),
        }

    k = int(cfg.top_sectors)
    k = max(int(cfg.min_top_sectors), min(k, len(ranked)))
    top_keys = ranked.head(k)["SectorKey"].tolist()
    stock_map = get_sector_stocks(top_keys, max_per_sector=cfg.max_stocks_scanned_per_sector)

    selections: list[dict[str, Any]] = []
    for sk in top_keys:
        tix = stock_map.get(sk, [])
        sel = select_stock_per_sector(sk, tix, stock_prices, cfg)
        if sel is not None:
            selections.append(sel)

    final_tickers, ptab = portfolio_construction(selections)
    return {
        "sector_rank_table": ranked,
        "selected_sectors": top_keys,
        "selections": selections,
        "final_tickers": final_tickers,
        "portfolio_table": ptab,
    }


def prepare_stock_prices_for_backtest(
    tickers: list[str],
    *,
    period: str = "5y",
    progress_callback=None,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for sector stock universe (light pacing)."""
    dc = DownloadConfig(period=period, interval="1d", auto_adjust=False)
    out: dict[str, pd.DataFrame] = {}
    n = len(tickers)
    for i, sym in enumerate(tickers):
        if progress_callback:
            progress_callback(i + 1, n, sym)
        try:
            df = download_stock_data(sym.strip().upper(), cfg=dc)
            out[sym.strip().upper()] = df.reset_index(drop=True)
        except Exception:
            pass
        time.sleep(0.04)
    return out


def run_sector_rotation_walk_forward_step(
    sector_index_frames: dict[str, pd.DataFrame],
    stock_prices_full: dict[str, pd.DataFrame],
    asof: pd.Timestamp,
    cfg: SectorRotationConfig | None = None,
    *,
    min_rows: int = 120,
) -> dict[str, Any]:
    """One rebalance date: truncate data to asof, then snapshot."""
    if cfg is None:
        cfg = SectorRotationConfig()
    st_trim = truncate_prices_asof(stock_prices_full, asof, min_rows=min_rows)
    ix_trim = truncate_prices_asof(sector_index_frames, asof, min_rows=15)
    return run_sector_rotation_snapshot(ix_trim, st_trim, cfg, asof=None)

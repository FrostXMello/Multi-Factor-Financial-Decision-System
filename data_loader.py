from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from utils import validate_columns


@dataclass(frozen=True)
class DownloadConfig:
    period: str = "5y"
    interval: str = "1d"
    # Yahoo adjusted OHLC so splits and dividends stay on the same basis as Close.
    auto_adjust: bool = True


DEFAULT_PERIOD_BY_INTERVAL: dict[str, str] = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "90m": "730d",
    "1h": "730d",
    "1d": "5y",
    "1wk": "10y",
}


def _normalize_interval(interval: str) -> str:
    token = str(interval).strip().lower()
    if token == "1h":
        return "60m"
    return token


def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop bars that cannot be traded or marked: non-positive prices, negative volume,
    or a high/low range that does not contain the open and close.
    """
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing}")
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    o = out["Open"].astype(float)
    h = out["High"].astype(float)
    l = out["Low"].astype(float)
    c = out["Close"].astype(float)
    v = out["Volume"].astype(float)
    ok = (
        np.isfinite(o)
        & np.isfinite(h)
        & np.isfinite(l)
        & np.isfinite(c)
        & np.isfinite(v)
        & (o > 0)
        & (c > 0)
        & (h >= l)
        & (h + 1e-8 >= np.maximum(o, c))
        & (l - 1e-8 <= np.minimum(o, c))
        & (v >= 0)
    )
    cleaned = out.loc[ok].reset_index(drop=True)
    cleaned.attrs["dropped_invalid_bars"] = int((~ok).sum())
    return cleaned


def download_stock_data(ticker: str, cfg: DownloadConfig | None = None) -> pd.DataFrame:
    """
    Download daily OHLCV data using yfinance.

    Returns a DataFrame with columns:
    Date, Open, High, Low, Close, Volume
    """
    if cfg is None:
        cfg = DownloadConfig()

    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")

    ticker = ticker.strip().upper()

    interval = _normalize_interval(cfg.interval)

    df = yf.download(
        tickers=ticker,
        period=cfg.period,
        interval=interval,
        auto_adjust=cfg.auto_adjust,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # yfinance returns DateTimeIndex; columns are often Open/High/Low/Close/Adj Close/Volume.
    df = df.reset_index()

    # yfinance frequently returns MultiIndex columns like (Price, Ticker) for single-ticker downloads.
    # We only need the first level (e.g., "Open", "High", "Close", ..., "Date").
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]) for col in df.columns]

    # Some yfinance versions name the date column "Date"; others can be "index".
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "Date"})

    needed = ["Open", "High", "Low", "Close", "Volume"]
    validate_columns(df, needed, df_name="raw download")
    df = df[["Date"] + needed].copy()

    # Clean numeric issues, sort, remove duplicate dates.
    df = df.replace([np.inf, -np.inf], np.nan)
    # Normalize all timestamps to a single representation to avoid merge errors
    # between timezone-aware and timezone-naive datetime columns.
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    # Drop rows with missing OHLCV values, then drop structurally invalid bars.
    df = df.dropna(subset=needed).reset_index(drop=True)
    df = sanitize_ohlcv(df)
    if df.empty:
        raise ValueError(f"No valid OHLCV rows left for ticker: {ticker}")

    return df


def download_multi_timeframe_data(
    ticker: str,
    *,
    intervals: tuple[str, ...] = ("1m", "5m", "15m", "60m", "1d"),
    period_by_interval: dict[str, str] | None = None,
    auto_adjust: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for multiple intervals.

    The returned dict keys are normalized interval tokens (e.g. "60m" for "1h").
    """
    if not intervals:
        raise ValueError("intervals must contain at least one interval")

    period_map = dict(DEFAULT_PERIOD_BY_INTERVAL)
    if period_by_interval:
        period_map.update({str(k).strip().lower(): v for k, v in period_by_interval.items()})

    out: dict[str, pd.DataFrame] = {}
    for interval in intervals:
        norm_interval = _normalize_interval(interval)
        period = period_map.get(norm_interval, "1y")
        df = download_stock_data(
            ticker,
            cfg=DownloadConfig(period=period, interval=norm_interval, auto_adjust=auto_adjust),
        )
        out[norm_interval] = df

    return out


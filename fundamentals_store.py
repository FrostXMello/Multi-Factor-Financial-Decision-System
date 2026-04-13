from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from utils import NIFTY50_TICKERS


@dataclass(frozen=True)
class FundamentalsConfig:
    cache_path: str = ".cache/nifty50_fundamentals.csv"
    max_age_hours: int = 24
    tickers: tuple[str, ...] = tuple(NIFTY50_TICKERS)


FUNDAMENTAL_FIELDS: tuple[str, ...] = (
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "returnOnEquity",
    "debtToEquity",
    "currentRatio",
    "profitMargins",
    "operatingMargins",
    "revenueGrowth",
    "earningsGrowth",
    "freeCashflow",
    "beta",
)


def _cache_is_fresh(path: Path, max_age_hours: int) -> bool:
    if not path.exists() or max_age_hours <= 0:
        return False
    age_seconds = pd.Timestamp.utcnow().timestamp() - path.stat().st_mtime
    return age_seconds <= float(max_age_hours * 3600)


def _safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _fetch_one_ticker_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    row = {
        "Date": pd.Timestamp.utcnow().normalize(),
        "Ticker": str(ticker).strip().upper(),
    }
    for field in FUNDAMENTAL_FIELDS:
        row[field] = _safe_float(info.get(field))

    valid = sum(1 for field in FUNDAMENTAL_FIELDS if pd.notna(row[field]))
    row["DataQualityScore"] = float(valid / max(1, len(FUNDAMENTAL_FIELDS)))
    return row


def collect_nifty50_fundamentals(
    cfg: FundamentalsConfig | None = None,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Collect NIFTY50 fundamentals and store/read from a local CSV cache.
    """
    if cfg is None:
        cfg = FundamentalsConfig()

    cache = Path(cfg.cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if not force_refresh and _cache_is_fresh(cache, cfg.max_age_hours):
        return pd.read_csv(cache)

    rows = [_fetch_one_ticker_fundamentals(t) for t in cfg.tickers]
    out = pd.DataFrame(rows)

    # Deterministic ordering and simple cross-sectional imputation fallback.
    out = out.sort_values("Ticker").reset_index(drop=True)
    numeric_cols = [c for c in out.columns if c not in {"Date", "Ticker"}]
    for col in numeric_cols:
        med = out[col].median(skipna=True)
        out[col] = out[col].fillna(med)

    out.to_csv(cache, index=False)
    return out


def build_fundamentals_quality_report(funda_df: pd.DataFrame) -> pd.DataFrame:
    if funda_df is None or funda_df.empty:
        return pd.DataFrame(
            [
                {
                    "Tickers": 0,
                    "AvgQualityScore": float("nan"),
                    "MinQualityScore": float("nan"),
                    "MaxQualityScore": float("nan"),
                    "AsOfDate": None,
                }
            ]
        )

    date_val = pd.to_datetime(funda_df["Date"], errors="coerce").max() if "Date" in funda_df.columns else None
    score = funda_df["DataQualityScore"] if "DataQualityScore" in funda_df.columns else pd.Series(dtype=float)

    return pd.DataFrame(
        [
            {
                "Tickers": int(funda_df["Ticker"].nunique()) if "Ticker" in funda_df.columns else int(len(funda_df)),
                "AvgQualityScore": float(score.mean()) if len(score) else float("nan"),
                "MinQualityScore": float(score.min()) if len(score) else float("nan"),
                "MaxQualityScore": float(score.max()) if len(score) else float("nan"),
                "AsOfDate": date_val,
            }
        ]
    )


def get_fundamentals_for_ticker(funda_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if funda_df is None or funda_df.empty:
        return pd.DataFrame()
    t = str(ticker).strip().upper()
    out = funda_df[funda_df["Ticker"].astype(str).str.upper() == t].copy()
    return out.reset_index(drop=True)


def subset_fundamentals(funda_df: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    if funda_df is None or funda_df.empty:
        return pd.DataFrame()
    keep = {str(t).strip().upper() for t in tickers if str(t).strip()}
    out = funda_df[funda_df["Ticker"].astype(str).str.upper().isin(keep)].copy()
    return out.reset_index(drop=True)

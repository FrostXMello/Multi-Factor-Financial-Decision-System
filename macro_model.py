from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf

from data_loader import DownloadConfig, download_stock_data


POSITIVE_WORDS = {
    "growth",
    "cooling inflation",
    "rate cut",
    "beat",
    "surge",
    "record high",
    "soft landing",
    "optimism",
}
NEGATIVE_WORDS = {
    "recession",
    "war",
    "hike",
    "hot inflation",
    "downgrade",
    "crash",
    "selloff",
    "slowdown",
}


@dataclass(frozen=True)
class MacroConfig:
    market_ticker: str = "^NSEI"
    lookback_period: str = "5y"
    horizon_days: int = 1


def _headline_sentiment_score(headlines: list[str]) -> float:
    if not headlines:
        return 0.0
    score = 0.0
    for h in headlines:
        text = h.lower()
        pos = sum(1 for w in POSITIVE_WORDS if w in text)
        neg = sum(1 for w in NEGATIVE_WORDS if w in text)
        score += (pos - neg)
    return float(score / max(1, len(headlines)))


def _get_macro_headlines(cfg: MacroConfig) -> list[str]:
    try:
        news = yf.Ticker(cfg.market_ticker).news or []
    except Exception:
        return []
    titles: list[str] = []
    for item in news[:40]:
        title = item.get("title") if isinstance(item, dict) else None
        if title:
            titles.append(str(title))
    return titles


@lru_cache(maxsize=32)
def _build_macro_features_cached(market_ticker: str, lookback_period: str, horizon_days: int) -> pd.DataFrame:
    cfg = MacroConfig(market_ticker=market_ticker, lookback_period=lookback_period, horizon_days=horizon_days)
    idx = download_stock_data(
        cfg.market_ticker,
        DownloadConfig(period=cfg.lookback_period, interval="1d", auto_adjust=False),
    )
    data = idx[["Date", "Close", "Volume"]].copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    data["MktRet1"] = data["Close"].pct_change()
    data["MktRet5"] = data["Close"].pct_change(5)
    data["MktVol20"] = data["MktRet1"].rolling(20, min_periods=20).std()
    data["MktMA20"] = data["Close"].rolling(20, min_periods=20).mean()
    data["MktTrend"] = data["Close"] / data["MktMA20"] - 1.0
    data["VolumeZ"] = (data["Volume"] - data["Volume"].rolling(20, min_periods=20).mean()) / data["Volume"].rolling(
        20, min_periods=20
    ).std()

    sent = _headline_sentiment_score(_get_macro_headlines(cfg))
    data["MacroNewsSent"] = sent

    data["MacroTarget"] = (data["Close"].shift(-cfg.horizon_days) / data["Close"] - 1.0 > 0).astype(int)
    keep = ["Date", "MktRet1", "MktRet5", "MktVol20", "MktTrend", "VolumeZ", "MacroNewsSent", "MacroTarget"]
    data = data[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return data


def build_macro_features(cfg: MacroConfig | None = None) -> pd.DataFrame:
    """
    Build macro regime features from a benchmark index plus headline sentiment.
    """
    if cfg is None:
        cfg = MacroConfig()
    return _build_macro_features_cached(cfg.market_ticker, cfg.lookback_period, cfg.horizon_days).copy()


def infer_macro_probability(
    macro_features: pd.DataFrame,
    signal_dates: pd.Series,
) -> pd.Series:
    """
    Return macro up probabilities aligned to given signal dates.
    Uses a simple logistic transform of macro features (no heavy training dependency).
    """
    req = ["Date", "MktRet1", "MktRet5", "MktVol20", "MktTrend", "VolumeZ", "MacroNewsSent"]
    missing = [c for c in req if c not in macro_features.columns]
    if missing:
        raise ValueError(f"macro_features missing columns: {missing}")

    data = macro_features.copy()
    # Hand-tuned linear score; intentionally conservative.
    score = (
        0.9 * data["MktRet1"]
        + 0.7 * data["MktRet5"]
        + 0.6 * data["MktTrend"]
        - 2.2 * data["MktVol20"]
        + 0.2 * data["VolumeZ"].clip(-3, 3)
        + 0.12 * data["MacroNewsSent"]
    )
    data["MacroProba"] = 1.0 / (1.0 + np.exp(-score))
    series = data.set_index(pd.to_datetime(data["Date"]))["MacroProba"].sort_index()
    aligned = series.reindex(pd.to_datetime(signal_dates)).ffill().bfill()
    return aligned.reset_index(drop=True)

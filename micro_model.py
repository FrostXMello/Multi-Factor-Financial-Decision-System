from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass(frozen=True)
class MicroConfig:
    random_state: int = 42
    min_rows: int = 120
    max_depth: int = 4
    max_iter: int = 250


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def fetch_fundamental_snapshot(ticker: str) -> dict[str, float]:
    """
    Fetch lightweight point-in-time fundamental snapshot from Yahoo metadata.
    """
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    return {
        "PE": _safe_float(info.get("trailingPE"), 0.0),
        "PB": _safe_float(info.get("priceToBook"), 0.0),
        "ROE": _safe_float(info.get("returnOnEquity"), 0.0),
        "DebtToEquity": _safe_float(info.get("debtToEquity"), 0.0),
        "ProfitMargin": _safe_float(info.get("profitMargins"), 0.0),
        "RevenueGrowth": _safe_float(info.get("revenueGrowth"), 0.0),
    }


@lru_cache(maxsize=256)
def _cached_fundamental_snapshot(ticker: str) -> tuple[float, float, float, float, float, float]:
    snap = fetch_fundamental_snapshot(ticker)
    return (
        float(snap.get("PE", 0.0)),
        float(snap.get("PB", 0.0)),
        float(snap.get("ROE", 0.0)),
        float(snap.get("DebtToEquity", 0.0)),
        float(snap.get("ProfitMargin", 0.0)),
        float(snap.get("RevenueGrowth", 0.0)),
    )


def _company_news_sentiment(ticker: str) -> float:
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        news = []
    if not news:
        return 0.0
    positive = ("beat", "growth", "upgrade", "expands", "strong", "surge")
    negative = ("miss", "downgrade", "fraud", "weak", "decline", "cuts")
    score = 0.0
    for item in news[:30]:
        title = str(item.get("title", "")).lower() if isinstance(item, dict) else ""
        score += sum(1 for w in positive if w in title)
        score -= sum(1 for w in negative if w in title)
    return float(score / max(1, len(news[:30])))


@lru_cache(maxsize=256)
def _cached_company_news_sentiment(ticker: str) -> float:
    return float(_company_news_sentiment(ticker))


def infer_micro_probability(model_df: pd.DataFrame, ticker: str, cfg: MicroConfig | None = None) -> pd.Series:
    """
    Build a company-specific probability using fundamentals + technical base features.
    """
    if cfg is None:
        cfg = MicroConfig()
    if len(model_df) < cfg.min_rows:
        raise ValueError(f"Not enough rows for micro model: {len(model_df)}")

    required = ["Date", "Target", "MA20", "MA50", "RSI14", "Daily_Return", "Rolling_Volatility_10"]
    missing = [c for c in required if c not in model_df.columns]
    if missing:
        raise ValueError(f"model_df missing columns: {missing}")

    data = model_df.sort_values("Date").reset_index(drop=True).copy()
    pe, pb, roe, dte, pm, rg = _cached_fundamental_snapshot(str(ticker).strip().upper())
    funda = {
        "PE": pe,
        "PB": pb,
        "ROE": roe,
        "DebtToEquity": dte,
        "ProfitMargin": pm,
        "RevenueGrowth": rg,
    }
    news_sent = _cached_company_news_sentiment(str(ticker).strip().upper())

    for k, v in funda.items():
        data[f"F_{k}"] = v
    data["F_CompanyNewsSent"] = news_sent

    feature_cols = [
        "MA20",
        "MA50",
        "RSI14",
        "Daily_Return",
        "Rolling_Volatility_10",
        "F_PE",
        "F_PB",
        "F_ROE",
        "F_DebtToEquity",
        "F_ProfitMargin",
        "F_RevenueGrowth",
        "F_CompanyNewsSent",
    ]

    split_idx = int(0.8 * len(data))
    train_df = data.iloc[:split_idx]
    full_df = data

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=cfg.max_depth,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
    )
    clf.fit(train_df[feature_cols].values, train_df["Target"].values)
    proba = clf.predict_proba(full_df[feature_cols].values)[:, 1]
    return pd.Series(proba, index=full_df.index, name="MicroProba")


def clear_micro_caches() -> None:
    _cached_fundamental_snapshot.cache_clear()
    _cached_company_news_sentiment.cache_clear()

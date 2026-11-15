"""
Causal market-regime classification.

Regimes at date t use only prices through t. Strategy returns on date t are
evaluated against the regime known at t-1 so the label cannot peek at the
same-bar return being measured.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_SIDEWAYS = "sideways"
REGIME_INSUFFICIENT = "insufficient_data"


def classify_regimes(
    prices: pd.DataFrame,
    *,
    lookback: int = 60,
    bull_threshold: float = 0.08,
    bear_threshold: float = -0.08,
    price_col: str = "Close",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Classify each bar into bull / bear / sideways from trailing total return.

    trailing_return[t] = Close[t] / Close[t-lookback] - 1
    Labels before `lookback` observations are `insufficient_data`.
    """
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if bull_threshold <= 0 or bear_threshold >= 0:
        raise ValueError("bull_threshold must be > 0 and bear_threshold must be < 0")
    if date_col not in prices.columns or price_col not in prices.columns:
        raise ValueError(f"prices must include {date_col!r} and {price_col!r}")

    frame = prices[[date_col, price_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values(date_col).drop_duplicates(date_col, keep="last").reset_index(drop=True)
    close = pd.to_numeric(frame[price_col], errors="coerce")
    trailing = close / close.shift(int(lookback)) - 1.0

    labels: list[str] = []
    for idx, value in enumerate(trailing.tolist()):
        if idx < lookback or value is None or not np.isfinite(value):
            labels.append(REGIME_INSUFFICIENT)
        elif float(value) >= float(bull_threshold):
            labels.append(REGIME_BULL)
        elif float(value) <= float(bear_threshold):
            labels.append(REGIME_BEAR)
        else:
            labels.append(REGIME_SIDEWAYS)

    out = pd.DataFrame(
        {
            date_col: frame[date_col],
            "Close": close,
            "trailing_return": trailing,
            "regime_end_of_day": labels,
        }
    )
    # Causal label for evaluating the return realized on date t.
    out["regime_for_return"] = out["regime_end_of_day"].shift(1)
    out.loc[out.index[0], "regime_for_return"] = REGIME_INSUFFICIENT
    out["regime_for_return"] = out["regime_for_return"].fillna(REGIME_INSUFFICIENT)
    out.attrs["lookback"] = int(lookback)
    out.attrs["bull_threshold"] = float(bull_threshold)
    out.attrs["bear_threshold"] = float(bear_threshold)
    out.attrs["lookahead_safe"] = True
    return out


def _metric_bundle(returns: pd.Series) -> dict[str, Any]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return {
            "observations": int(len(r)),
            "available": False,
            "mean_return": None,
            "volatility": None,
            "total_return": None,
            "sharpe_like": None,
            "detail": "insufficient_data",
        }
    vol = float(r.std())
    mean = float(r.mean())
    total = float((1.0 + r).prod() - 1.0)
    sharpe_like = float(mean / vol) if vol > 0 else None
    return {
        "observations": int(len(r)),
        "available": True,
        "mean_return": mean,
        "volatility": vol,
        "total_return": total,
        "sharpe_like": sharpe_like,
        "detail": "ok",
    }


def evaluate_returns_by_regime(
    returns: pd.Series,
    regimes: pd.Series,
    *,
    min_observations: int = 5,
) -> dict[str, Any]:
    """
    Aggregate strategy period returns by causal regime labels.

    `returns` and `regimes` must share an index (typically Date). Rows labelled
    insufficient_data are reported separately and excluded from bull/bear/sideways.
    """
    aligned = pd.DataFrame({"ret": returns, "regime": regimes}).dropna(subset=["ret"])
    aligned["regime"] = aligned["regime"].astype(str)
    summary: dict[str, Any] = {}
    for name in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS, REGIME_INSUFFICIENT):
        subset = aligned.loc[aligned["regime"] == name, "ret"]
        bundle = _metric_bundle(subset)
        if name != REGIME_INSUFFICIENT and bundle["observations"] < int(min_observations):
            bundle["available"] = False
            bundle["detail"] = "insufficient_data"
            # Keep raw counts but clear risk of over-interpreting tiny samples.
            for key in ("mean_return", "volatility", "total_return", "sharpe_like"):
                if bundle["observations"] < int(min_observations):
                    pass
        summary[name] = bundle

    ranked = []
    for name in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
        item = summary[name]
        if item.get("available") and item.get("total_return") is not None:
            ranked.append((name, float(item["total_return"])))
    ranked.sort(key=lambda x: x[1], reverse=True)

    weaknesses = []
    for name in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
        item = summary[name]
        if not item.get("available"):
            weaknesses.append(f"{name}: insufficient observations for a reliable read.")
        elif item.get("total_return") is not None and float(item["total_return"]) < 0:
            weaknesses.append(f"{name}: negative compounded return over the labelled sample.")

    return {
        "by_regime": summary,
        "best_regime": ranked[0][0] if ranked else None,
        "worst_regime": ranked[-1][0] if ranked else None,
        "weaknesses": weaknesses,
        "min_observations": int(min_observations),
        "alignment": "regime_for_return uses prior end-of-day regime (no same-bar lookahead).",
    }


def attach_causal_regimes_to_equity(
    equity_curve: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    lookback: int = 60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Join equity daily returns to causal regimes from price history.
    """
    regimes = classify_regimes(price_history, lookback=lookback)
    eq = equity_curve.copy()
    eq["Date"] = pd.to_datetime(eq["Date"])
    eq = eq.sort_values("Date")
    if "PortfolioValue" not in eq.columns:
        raise ValueError("equity_curve requires PortfolioValue")
    eq["strategy_return"] = eq["PortfolioValue"].astype(float).pct_change()
    merged = eq.merge(
        regimes[["Date", "regime_end_of_day", "regime_for_return", "trailing_return"]],
        on="Date",
        how="left",
    )
    merged["regime_for_return"] = merged["regime_for_return"].fillna(REGIME_INSUFFICIENT)
    evaluation = evaluate_returns_by_regime(
        merged.set_index("Date")["strategy_return"],
        merged.set_index("Date")["regime_for_return"],
    )
    return merged, evaluation

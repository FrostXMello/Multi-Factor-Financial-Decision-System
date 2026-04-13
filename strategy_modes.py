"""
Strategy selection for the trading pipeline.

Each strategy uses a disjoint feature set built in `feature_engineering` (no cross-strategy
columns on the training matrix) to avoid accidental leakage between modes.

- **multi_factor**: Macro + micro + technical fusion stack (existing “AI stack”).
- **momentum**: Trend-following style common in systematic / CTA books — ride strength
  when medium-term trend aligns and momentum (RSI band) confirms.
- **mean_reversion**: Statistical mean-reversion / bounded-price intuition — bet on
  snap-back when price is far below/above a short rolling mean (Z-score), similar in spirit
  to stat-arb sleeves (simplified, single-name).

Runtime switch: `FourModelConfig.strategy_mode` in {"multi_factor", "momentum", "mean_reversion"}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical values for `strategy_mode` / config / APIs.
STRATEGY_MULTI_FACTOR = "multi_factor"
STRATEGY_MOMENTUM = "momentum"
STRATEGY_MEAN_REVERSION = "mean_reversion"

VALID_STRATEGY_MODES = frozenset({STRATEGY_MULTI_FACTOR, STRATEGY_MOMENTUM, STRATEGY_MEAN_REVERSION})

# UI labels -> canonical mode
_STRATEGY_ALIASES: dict[str, str] = {
    "multi_factor": STRATEGY_MULTI_FACTOR,
    "multi-factor": STRATEGY_MULTI_FACTOR,
    "multifactor": STRATEGY_MULTI_FACTOR,
    "momentum": STRATEGY_MOMENTUM,
    "mean_reversion": STRATEGY_MEAN_REVERSION,
    "mean-reversion": STRATEGY_MEAN_REVERSION,
    "meanreversion": STRATEGY_MEAN_REVERSION,
}


def normalize_strategy_mode(mode: str | None) -> str:
    if mode is None or (isinstance(mode, str) and not str(mode).strip()):
        return STRATEGY_MULTI_FACTOR
    key = str(mode).lower().strip().replace(" ", "_").replace("-", "_")
    if key in _STRATEGY_ALIASES:
        return _STRATEGY_ALIASES[key]
    if key in VALID_STRATEGY_MODES:
        return key
    raise ValueError(f"Unknown strategy_mode={mode!r}; expected one of {sorted(VALID_STRATEGY_MODES)}")


def momentum_trend_up(row: pd.Series) -> bool:
    """Strong uptrend: price above long MA and short MA above long MA."""
    return float(row["Close"]) > float(row["MA50"]) and float(row["MA20"]) > float(row["MA50"])


def momentum_trend_down(row: pd.Series) -> bool:
    """Symetric downtrend for exit side."""
    return float(row["Close"]) < float(row["MA50"]) and float(row["MA20"]) < float(row["MA50"])


def momentum_rsi_confirms(row: pd.Series, *, rsi_col: str = "RSI14", lo: float = 55.0, hi: float = 75.0) -> bool:
    """Momentum confirmation band (not overbought panic, not deep oversold)."""
    rsi = float(row[rsi_col])
    return lo <= rsi <= hi


def attach_momentum_strategy_signals(
    test_df: pd.DataFrame,
    test_proba: np.ndarray,
    *,
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Momentum / trend-following execution layer.

    Combines classifier probability (P(upward move)) with **structure**:
    - BUY only if model is bullish *and* trend is up *and* RSI confirms (55–75).
    - SELL if model is bearish *and* trend is down.

    This captures delayed, persistent participation in directional moves — the kind of
    rule-of-thumb used in many quant equity momentum sleeves (simplified for teaching).
    """
    out = test_df.copy()
    p = np.asarray(test_proba, dtype=float)
    p = np.clip(np.where(np.isfinite(p), p, 0.5), 0.0, 1.0)
    out["Proba"] = p

    sig: list[str] = []
    for i in range(len(out)):
        row = out.iloc[i]
        pv = float(p[i])
        if pv > buy_threshold and momentum_trend_up(row) and momentum_rsi_confirms(row):
            sig.append("BUY")
        elif pv < sell_threshold and momentum_trend_down(row):
            sig.append("SELL")
        else:
            sig.append("HOLD")
    out["Signal"] = sig
    return out


def attach_mean_reversion_strategy_signals(
    test_df: pd.DataFrame,
    test_proba: np.ndarray | None = None,
    *,
    z_col: str = "MR_ZScore",
    z_buy: float = -2.0,
    z_sell: float = 2.0,
) -> pd.DataFrame:
    """
    Mean-reversion / bounded-price signal layer.

    Uses **Z-score** distance from a short rolling mean:
    - Z < -2 → BUY (expect rebound).
    - Z > +2 → SELL (expect fade).

    Optional `test_proba` fills `Proba` for dashboard consistency when a classifier is
    trained on the same mean-reversion features; **trades are driven only by Z**, matching
    the rule-based spec (classifier is diagnostic only in this mode).
    """
    out = test_df.copy()
    if z_col not in out.columns:
        raise ValueError(f"Mean reversion signals require column {z_col!r}")

    if test_proba is not None:
        p = np.asarray(test_proba, dtype=float)
        p = np.clip(np.where(np.isfinite(p), p, 0.5), 0.0, 1.0)
        out["Proba"] = p
    else:
        out["Proba"] = 0.5

    sig: list[str] = []
    for i in range(len(out)):
        z = float(out.iloc[i][z_col])
        if not np.isfinite(z):
            sig.append("HOLD")
        elif z < z_buy:
            sig.append("BUY")
        elif z > z_sell:
            sig.append("SELL")
        else:
            sig.append("HOLD")
    out["Signal"] = sig
    return out

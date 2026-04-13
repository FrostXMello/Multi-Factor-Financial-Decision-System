"""
CORE_MODE — evaluation-ready, low-complexity trading pipeline.

TECHNICAL_MODE — **primary simplification flag** (technical-only, no multi-model stack).

When TECHNICAL_MODE is True (recommended default):
- **Disabled at runtime**: macro, micro, fusion, meta-fusion, news/fundamentals in the main path.
  Those modules stay importable for research but are not called from `four_model_pipeline`.
- **Data**: daily bars as base; optional single higher timeframe (weekly) for context features only.
- **Features**: price-only — MA20/MA50, price vs MA50, RSI14, ROC, daily return, 10d vol (+ optional weekly).
- **Target**: forward return over `CORE_HORIZON_BARS` exceeding `TECHNICAL_MIN_FORWARD_RETURN` (meaningful move, not noise).
- **Model**: one gradient boosting classifier, chronological split, `predict_proba`.
- **Signals**: P>buy & Close>MA50 → BUY; P<sell & Close<MA50 → SELL; else HOLD (no RSI band gate).
- **Portfolio**: optional daily top-N by probability with a single capital pool (see `backtest`).
- **Risk**: volatility / drawdown / position sizing kept, thresholds tuned to avoid over-blocking.
- **Optimizer / walk-forward / intraday churn**: off in this mode.

When CORE_MODE is True (legacy evaluation switch; typically aligned with TECHNICAL_MODE):
- Fixed intervals, fixed thresholds, fast path, etc.

Re-enabling the full research stack
- Set TECHNICAL_MODE = False and CORE_MODE = False in this file.
"""

from __future__ import annotations

# Master switches — defined first so `from core_config import TECHNICAL_MODE` always succeeds
# (before any other imports that might pull this module in transitively).
TECHNICAL_MODE: bool = True
CORE_MODE: bool = True

__all__ = [
    "TECHNICAL_MODE",
    "CORE_MODE",
    "CORE_BASE_INTERVAL",
    "CORE_CONTEXT_INTERVALS",
    "CORE_HORIZON_BARS",
    "TECHNICAL_MIN_FORWARD_RETURN",
    "CORE_MODEL_TYPE",
    "CORE_BUY_THRESHOLD",
    "CORE_SELL_THRESHOLD",
    "CORE_TECH_FUSION_WEIGHT",
    "CORE_MACRO_FUSION_WEIGHT",
    "CORE_MAX_TRADES_PER_DAY",
    "CORE_MIN_MINUTES_BETWEEN_TRADES",
    "CORE_RISK_CONFIG",
    "TECHNICAL_PORTFOLIO_TOP_N",
    "assert_optimizer_allowed",
]

from risk_model import RiskConfig

# --- Technical-mode data & label (multi-day horizon avoids 1-bar noise) ---
CORE_BASE_INTERVAL: str = "1d"
CORE_CONTEXT_INTERVALS: tuple[str, ...] = ("1wk",)
# Forward horizon in **base bars** (trading days when base is 1d). 5–10d range; default 7.
CORE_HORIZON_BARS: int = 7
# Binary target: 1 if forward return over horizon exceeds this (e.g. 1% meaningful move).
TECHNICAL_MIN_FORWARD_RETURN: float = 0.01

# Single model type in technical mode (HistGradientBoosting in sklearn).
CORE_MODEL_TYPE: str = "gb"

CORE_BUY_THRESHOLD: float = 0.6
CORE_SELL_THRESHOLD: float = 0.4

# Fusion weights kept for API compatibility; unused when TECHNICAL_MODE (no fusion call).
CORE_TECH_FUSION_WEIGHT: float = 1.0
CORE_MACRO_FUSION_WEIGHT: float = 0.0

CORE_MAX_TRADES_PER_DAY: int = 10_000
CORE_MIN_MINUTES_BETWEEN_TRADES: int = 0

# Slightly relaxed vs ultra-tight gates so risk does not over-block legitimate trades.
CORE_RISK_CONFIG = RiskConfig(
    vol_window=20,
    drawdown_window=60,
    max_vol_for_full_risk=0.022,
    hard_vol_stop=0.07,
    hard_drawdown_stop=0.20,
    base_position_size=1.0,
    min_position_size=0.18,
)

# Portfolio: concentrate into this many names by highest P(BUY) among valid BUY signals.
TECHNICAL_PORTFOLIO_TOP_N: int = 3


def assert_optimizer_allowed() -> None:
    """Call from optimizer entry points; raises if technical / core mode forbids tuning loops."""
    if TECHNICAL_MODE:
        raise RuntimeError(
            "Optimizer is disabled when core_config.TECHNICAL_MODE is True. "
            "Set TECHNICAL_MODE = False for offline tuning, or edit thresholds in core_config."
        )
    if CORE_MODE:
        raise RuntimeError(
            "Optimizer is disabled when core_config.CORE_MODE is True. "
            "Set CORE_MODE = False for offline tuning, or tune parameters manually in core_config."
        )

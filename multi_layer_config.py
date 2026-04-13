"""
Multi-layer stock selection (quant-style pipeline).

When MULTI_LAYER_MODE is True, the app can run:
  Universe → Layer 1 (fast filters) → Layer 2 (ML scores) → Layer 3 (portfolio) → risk/backtest.

Set MULTI_LAYER_MODE = False to hide/disable this path in favor of the standard single/compact portfolio UI only.
"""

from __future__ import annotations

from dataclasses import dataclass

from risk_model import RiskConfig

# Master switch: enable "Technical Multi-Layer" universe mode in Streamlit and helpers.
MULTI_LAYER_MODE: bool = True

# --- Pre-filter (before Layer 1) ---
MIN_OHLCV_ROWS: int = 400
# Approximate minimum 20-day average *dollar* liquidity (Close * Volume, currency of listing).
MIN_AVG_DOLLAR_VOLUME_20D: float = 2.0e6

# --- Layer 1: fast universe reduction (~300 → ~40–60) ---
LAYER1_MOMENTUM_DAYS: int = 20
LAYER1_MA_TREND: int = 50
LAYER1_VOL_WINDOW: int = 20
LAYER1_MIN_ANN_VOL: float = 0.08
LAYER1_MAX_ANN_VOL: float = 0.90
LAYER1_OUTPUT_MIN: int = 40
LAYER1_OUTPUT_MAX: int = 60

# --- Layer 2: ML scoring (→ ~20–25) ---
LAYER2_TOP_K: int = 25
LAYER2_MODEL_MIN_ROWS: int = 160
LAYER2_HORIZON_BARS: int = 7
LAYER2_MIN_FORWARD_RETURN: float = 0.01
LAYER2_USE_RISK_ADJUSTED_SCORE: bool = False  # Score = Proba / ann_vol when True

# --- Layer 3: final book (3–5 names) ---
LAYER3_PORTFOLIO_N_DEFAULT: int = 4
LAYER3_MIN_PROBA: float = 0.6
LAYER3_MAX_PAIRWISE_CORR: float | None = 0.88  # None to skip correlation pruning

# --- Backtest ---
MULTI_LAYER_REBALANCE: str = "M"  # month-end frequency token for pandas
MULTI_LAYER_MAX_REBALANCE_POINTS: int = 36  # cap runtime for large scans
MULTI_LAYER_WARMUP_BARS: int = 252

DEFAULT_RISK = RiskConfig(
    vol_window=20,
    drawdown_window=60,
    max_vol_for_full_risk=0.022,
    hard_vol_stop=0.08,
    hard_drawdown_stop=0.22,
    base_position_size=1.0,
    min_position_size=0.18,
)


@dataclass(frozen=True)
class MultiLayerPipelineConfig:
    """Runtime overrides (Streamlit can build this from sliders)."""

    layer1_out_max: int = LAYER1_OUTPUT_MAX
    layer1_out_min: int = LAYER1_OUTPUT_MIN
    layer2_top_k: int = LAYER2_TOP_K
    layer3_n: int = LAYER3_PORTFOLIO_N_DEFAULT
    min_proba: float = LAYER3_MIN_PROBA
    use_risk_adjusted: bool = LAYER2_USE_RISK_ADJUSTED_SCORE
    max_pairwise_corr: float | None = LAYER3_MAX_PAIRWISE_CORR
    rebalance_max_points: int = MULTI_LAYER_MAX_REBALANCE_POINTS
    horizon_bars: int = LAYER2_HORIZON_BARS
    min_forward_return: float = LAYER2_MIN_FORWARD_RETURN
    model_min_rows: int = LAYER2_MODEL_MIN_ROWS

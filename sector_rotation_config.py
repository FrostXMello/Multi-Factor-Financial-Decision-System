"""
Sector rotation pipeline — configuration (capital-flow style, daily/weekly, not intraday).
"""

from __future__ import annotations

from dataclasses import dataclass

# Master switch: show "Sector Rotation (Flow)" in Streamlit universe list.
SECTOR_ROTATION_MODE: bool = True


@dataclass(frozen=True)
class SectorRotationConfig:
    """Runtime knobs (Streamlit can mirror these)."""

    top_sectors: int = 3
    min_top_sectors: int = 2
    max_stocks_scanned_per_sector: int = 10
    horizon_bars: int = 7
    min_forward_return: float = 0.01
    roc_period: int = 10
    model_type: str = "gb"  # "gb" | "rf"
    min_rows_model: int = 160
    sector_vol_window: int = 20
    sector_score_ret5_weight: float = 0.6
    sector_score_ret10_weight: float = 0.4
    max_rolling_vol_10: float = 0.045
    risk_lookback_days: int = 60
    max_recent_drawdown: float = 0.22
    rebalance_max_points: int = 40
    backtest_years: float | None = 1.0
    warmup_bars: int = 200

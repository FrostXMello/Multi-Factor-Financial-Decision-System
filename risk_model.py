from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    vol_window: int = 20
    drawdown_window: int = 60
    max_vol_for_full_risk: float = 0.015
    hard_vol_stop: float = 0.05
    hard_drawdown_stop: float = 0.15
    base_position_size: float = 1.0
    min_position_size: float = 0.10


def _rolling_drawdown(close: pd.Series, window: int) -> pd.Series:
    roll_max = close.rolling(window=window, min_periods=window).max()
    dd = close / roll_max - 1.0
    return dd


def compute_risk_frame(prices: pd.DataFrame, cfg: RiskConfig | None = None) -> pd.DataFrame:
    """
    Compute deterministic risk diagnostics per date.
    """
    if cfg is None:
        cfg = RiskConfig()

    required = ["Date", "Close"]
    missing = [c for c in required if c not in prices.columns]
    if missing:
        raise ValueError(f"prices missing columns: {missing}")

    data = prices[["Date", "Close"]].copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    data["Daily_Return"] = data["Close"].pct_change()
    data["RealizedVol"] = data["Daily_Return"].rolling(window=cfg.vol_window, min_periods=cfg.vol_window).std()
    data["RollingDrawdown"] = _rolling_drawdown(data["Close"], window=cfg.drawdown_window)

    # Risk score in [0,1]: higher means more risk.
    vol_component = (data["RealizedVol"] / cfg.max_vol_for_full_risk).clip(lower=0.0, upper=1.0)
    dd_component = (-data["RollingDrawdown"] / cfg.hard_drawdown_stop).clip(lower=0.0, upper=1.0)
    data["RiskScore"] = (0.6 * vol_component + 0.4 * dd_component).clip(lower=0.0, upper=1.0)

    # Hard risk gate and smooth sizing multiplier.
    data["NoTrade"] = (data["RealizedVol"] >= cfg.hard_vol_stop) | (data["RollingDrawdown"] <= -cfg.hard_drawdown_stop)
    raw_size = cfg.base_position_size * (1.0 - data["RiskScore"])
    data["PositionSize"] = raw_size.clip(lower=cfg.min_position_size, upper=cfg.base_position_size)

    return data[["Date", "RealizedVol", "RollingDrawdown", "RiskScore", "NoTrade", "PositionSize"]]


def apply_risk_gating(
    signals_df: pd.DataFrame,
    risk_frame: pd.DataFrame,
    *,
    buy_label: str = "BUY",
    sell_label: str = "SELL",
) -> pd.DataFrame:
    """
    Apply risk constraints to precomputed model signals.

    Rules:
    - If NoTrade is True -> force HOLD.
    - Else keep BUY/SELL/HOLD but attach PositionSize for execution sizing.
    """
    required = ["Date", "Signal"]
    missing = [c for c in required if c not in signals_df.columns]
    if missing:
        raise ValueError(f"signals_df missing columns: {missing}")

    merged = signals_df.copy()
    merged["Date"] = pd.to_datetime(merged["Date"])
    risk = risk_frame.copy()
    risk["Date"] = pd.to_datetime(risk["Date"])
    merged = merged.merge(risk, on="Date", how="left")

    for col in ["RiskScore", "NoTrade", "PositionSize"]:
        if col not in merged.columns:
            raise ValueError(f"risk_frame merge missing '{col}'")

    merged["NoTrade"] = merged["NoTrade"].fillna(True)
    merged["PositionSize"] = merged["PositionSize"].fillna(0.0)
    merged["RiskScore"] = merged["RiskScore"].fillna(1.0)

    new_signal = merged["Signal"].astype(str).str.upper()
    merged["SignalModel"] = new_signal
    new_signal = np.where(merged["NoTrade"], "HOLD", new_signal)
    merged["Signal"] = new_signal

    # Position sizing is only meaningful for active directions.
    active = merged["Signal"].isin([buy_label, sell_label])
    merged.loc[~active, "PositionSize"] = 0.0

    return merged

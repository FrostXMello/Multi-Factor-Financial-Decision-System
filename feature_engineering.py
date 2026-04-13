from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from strategy_modes import STRATEGY_MEAN_REVERSION, STRATEGY_MOMENTUM, STRATEGY_MULTI_FACTOR, normalize_strategy_mode
from utils import validate_columns


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using a simple rolling average of gains/losses.

    Note: This uses information up to and including the current day `t` (no look-ahead).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


@dataclass(frozen=True)
class MultiTimeframeDatasetConfig:
    base_interval: str = "1m"
    context_intervals: tuple[str, ...] = ("5m", "15m", "60m", "1d")
    horizon_bars: int = 5
    target_return_threshold: float = 0.0
    ma_short_window: int = 20
    ma_long_window: int = 50
    rsi_period: int = 14
    vol_window: int = 10
    tf_ma_window: int = 20
    tf_vol_window: int = 10
    downcast_float32: bool = True
    # CORE_MODE: keep only MA20, MA50, RSI14, Daily_Return, Rolling_Vol on base bars (no TF_* merges).
    core_mode: bool = False
    # strategy_mode: multi_factor (default stack), momentum (trend features), mean_reversion (Z-score features).
    strategy_mode: str = STRATEGY_MULTI_FACTOR
    momentum_roc_period: int = 10
    mean_reversion_window: int = 20
    # TECHNICAL_MODE / simplified pipeline: one feature path, meaningful-move target, optional one higher TF only.
    technical_only: bool = False
    technical_min_forward_return: float = 0.01
    roc_period: int = 10


def _interval_token(interval: str) -> str:
    return "".join(ch for ch in str(interval).lower() if ch.isalnum())


def _compute_close_feature_block(
    close: pd.Series,
    *,
    ma_window: int,
    rsi_period: int,
    vol_window: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    ret1 = close.pct_change()
    ma = close.rolling(window=ma_window, min_periods=ma_window).mean()
    trend = close / ma - 1.0
    rsi = _compute_rsi(close, period=rsi_period)
    vol = ret1.rolling(window=vol_window, min_periods=vol_window).std()
    return ret1, trend, rsi, vol


def _build_interval_feature_frame(df: pd.DataFrame, interval: str, cfg: MultiTimeframeDatasetConfig) -> tuple[pd.DataFrame, list[str]]:
    required = ["Date", "Close"]
    validate_columns(df, required, df_name=f"timeframe[{interval}]")

    token = _interval_token(interval)
    data = df[["Date", "Close"]].copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    ret1, trend, rsi, vol = _compute_close_feature_block(
        data["Close"],
        ma_window=cfg.tf_ma_window,
        rsi_period=cfg.rsi_period,
        vol_window=cfg.tf_vol_window,
    )
    data[f"TF_{token}_Ret1"] = ret1
    data[f"TF_{token}_Trend"] = trend
    data[f"TF_{token}_RSI{cfg.rsi_period}"] = rsi
    data[f"TF_{token}_Vol{cfg.tf_vol_window}"] = vol

    feat_cols = [
        f"TF_{token}_Ret1",
        f"TF_{token}_Trend",
        f"TF_{token}_RSI{cfg.rsi_period}",
        f"TF_{token}_Vol{cfg.tf_vol_window}",
    ]

    out = data[["Date"] + feat_cols].replace([np.inf, -np.inf], np.nan)
    return out, feat_cols


def build_multi_timeframe_dataset(
    timeframe_data: dict[str, pd.DataFrame],
    cfg: MultiTimeframeDatasetConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build an intraday-capable, multi-timeframe feature matrix aligned to `base_interval` bars.

    Returns:
    - dataset DataFrame (OHLCV-derived technical features + TF_* multi-timeframe columns)
    - technical feature columns list for model training
    """
    if cfg is None:
        cfg = MultiTimeframeDatasetConfig()
    if cfg.horizon_bars <= 0:
        raise ValueError("horizon_bars must be >= 1")
    if cfg.base_interval not in timeframe_data:
        raise ValueError(f"Missing base interval data: {cfg.base_interval}")

    base = timeframe_data[cfg.base_interval].copy()
    validate_columns(base, ["Date", "Open", "High", "Low", "Close", "Volume"], df_name="base timeframe")

    data = base.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    # --- Technical-only path (price features, no strategy_mode / multifactor / MR / momentum splits) ---
    if cfg.technical_only:
        ma_short = data["Close"].rolling(window=cfg.ma_short_window, min_periods=cfg.ma_short_window).mean()
        ma_long = data["Close"].rolling(window=cfg.ma_long_window, min_periods=cfg.ma_long_window).mean()
        rsi = _compute_rsi(data["Close"], period=cfg.rsi_period)
        roc_n = max(1, int(cfg.roc_period))
        roc_col = f"ROC_{roc_n}"
        roc = data["Close"].pct_change(roc_n)
        ret1 = data["Close"].pct_change()
        vol10 = ret1.rolling(window=cfg.vol_window, min_periods=cfg.vol_window).std()
        price_vs_ma50 = data["Close"] / ma_long.replace(0, np.nan) - 1.0
        data["MA20"] = ma_short
        data["MA50"] = ma_long
        data["Price_vs_MA50"] = price_vs_ma50
        data["RSI14"] = rsi
        data[roc_col] = roc
        data["Daily_Return"] = ret1
        data["Rolling_Volatility_10"] = vol10
        merged = data
        technical_cols = [
            "MA20",
            "MA50",
            "Price_vs_MA50",
            "RSI14",
            roc_col,
            "Daily_Return",
            "Rolling_Volatility_10",
        ]
        # Optional single higher timeframe (e.g. weekly): backward asof alignment, no duplicate base interval.
        seen_tf: set[str] = {cfg.base_interval}
        for interval in cfg.context_intervals:
            if interval in seen_tf:
                continue
            seen_tf.add(interval)
            frame = timeframe_data.get(interval)
            if frame is None or frame.empty:
                continue
            tf_features, tf_cols = _build_interval_feature_frame(frame, interval, cfg)
            technical_cols.extend(tf_cols)
            merged = pd.merge_asof(
                merged.sort_values("Date"),
                tf_features.sort_values("Date"),
                on="Date",
                direction="backward",
            )
            break  # one higher TF is enough for context; avoids feature bloat

        merged["Forward_Return"] = merged["Close"].shift(-cfg.horizon_bars) / merged["Close"] - 1.0
        merged["Target"] = (merged["Forward_Return"] > float(cfg.technical_min_forward_return)).astype(int)
        merged["Next_Close"] = merged["Close"].shift(-cfg.horizon_bars)
        merged["Next_Date"] = merged["Date"].shift(-cfg.horizon_bars)

        deduped_cols = list(dict.fromkeys(technical_cols))
        required_for_training = deduped_cols + ["Target", "Next_Close", "Next_Date", "Forward_Return"]
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=required_for_training).reset_index(drop=True)

        if cfg.downcast_float32:
            float_cols = merged.select_dtypes(include=["float64", "float32"]).columns
            merged[float_cols] = merged[float_cols].astype(np.float32)

        return merged, deduped_cols

    smode = normalize_strategy_mode(cfg.strategy_mode)
    merged = data
    technical_cols: list[str]

    if smode == STRATEGY_MOMENTUM:
        # Momentum-only features (no multifactor / MR columns — avoids leakage across strategies).
        ma_short = data["Close"].rolling(window=cfg.ma_short_window, min_periods=cfg.ma_short_window).mean()
        ma_long = data["Close"].rolling(window=cfg.ma_long_window, min_periods=cfg.ma_long_window).mean()
        rsi = _compute_rsi(data["Close"], period=cfg.rsi_period)
        roc_n = max(1, int(cfg.momentum_roc_period))
        roc_col = f"Momentum_ROC_{roc_n}"
        roc = data["Close"].pct_change(roc_n)
        price_vs_ma50 = data["Close"] / ma_long.replace(0, np.nan) - 1.0
        data["MA20"] = ma_short
        data["MA50"] = ma_long
        data["RSI14"] = rsi
        data[roc_col] = roc
        data["Price_vs_MA50"] = price_vs_ma50
        merged = data
        technical_cols = ["MA20", "MA50", "RSI14", roc_col, "Price_vs_MA50"]

    elif smode == STRATEGY_MEAN_REVERSION:
        w = max(2, int(cfg.mean_reversion_window))
        mr_mean = data["Close"].rolling(window=w, min_periods=w).mean()
        mr_std = data["Close"].rolling(window=w, min_periods=w).std()
        rsi = _compute_rsi(data["Close"], period=cfg.rsi_period)
        z = (data["Close"] - mr_mean) / mr_std.replace(0, np.nan)
        data["MR_Mean20"] = mr_mean
        data["MR_Std20"] = mr_std
        data["MR_ZScore"] = z
        data["RSI14"] = rsi
        merged = data
        technical_cols = ["MR_Mean20", "MR_Std20", "MR_ZScore", "RSI14"]

    else:
        # Multi-factor / legacy stack: broad technicals + optional higher-TF merges.
        ma_short = data["Close"].rolling(window=cfg.ma_short_window, min_periods=cfg.ma_short_window).mean()
        ma_long = data["Close"].rolling(window=cfg.ma_long_window, min_periods=cfg.ma_long_window).mean()
        ret1, _, rsi, vol = _compute_close_feature_block(
            data["Close"],
            ma_window=cfg.ma_short_window,
            rsi_period=cfg.rsi_period,
            vol_window=cfg.vol_window,
        )
        data["MA20"] = ma_short
        data["MA50"] = ma_long
        data["RSI14"] = rsi
        data["Daily_Return"] = ret1
        data["Rolling_Volatility_10"] = vol
        merged = data
        technical_cols = ["MA20", "MA50", "RSI14", "Daily_Return", "Rolling_Volatility_10"]

    # Merge higher-timeframe context into each base bar using backward asof alignment.
    # Only **multi_factor** uses TF_*; momentum/mean_reversion stay on base-bar features only.
    if smode == STRATEGY_MULTI_FACTOR and not cfg.core_mode:
        seen: set[str] = set()
        intervals: list[str] = [cfg.base_interval] + list(cfg.context_intervals)
        for interval in intervals:
            if interval in seen:
                continue
            seen.add(interval)
            frame = timeframe_data.get(interval)
            if frame is None or frame.empty:
                continue
            tf_features, tf_cols = _build_interval_feature_frame(frame, interval, cfg)
            technical_cols.extend(tf_cols)
            merged = pd.merge_asof(
                merged.sort_values("Date"),
                tf_features.sort_values("Date"),
                on="Date",
                direction="backward",
            )

    merged["Forward_Return"] = merged["Close"].shift(-cfg.horizon_bars) / merged["Close"] - 1.0
    merged["Target"] = (merged["Forward_Return"] > cfg.target_return_threshold).astype(int)
    merged["Next_Close"] = merged["Close"].shift(-cfg.horizon_bars)
    merged["Next_Date"] = merged["Date"].shift(-cfg.horizon_bars)

    deduped_cols = list(dict.fromkeys(technical_cols))
    required_for_training: Iterable[str] = deduped_cols + ["Target", "Next_Close", "Next_Date", "Forward_Return"]
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=required_for_training).reset_index(drop=True)

    if cfg.downcast_float32:
        float_cols = merged.select_dtypes(include=["float64", "float32"]).columns
        merged[float_cols] = merged[float_cols].astype(np.float32)

    return merged, deduped_cols


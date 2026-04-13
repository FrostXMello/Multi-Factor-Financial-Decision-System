from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_loader import download_multi_timeframe_data
from feature_engineering import MultiTimeframeDatasetConfig, build_multi_timeframe_dataset
from strategy_modes import STRATEGY_MULTI_FACTOR


@dataclass(frozen=True)
class DataAPIConfig:
    base_interval: str = "1m"
    context_intervals: tuple[str, ...] = ("5m", "15m", "60m", "1d")
    horizon_bars: int = 5
    auto_adjust: bool = False
    core_mode: bool = False
    strategy_mode: str = STRATEGY_MULTI_FACTOR


def build_price_quality_report(timeframe_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for interval, df in timeframe_data.items():
        if df is None or df.empty:
            rows.append(
                {
                    "Interval": interval,
                    "Rows": 0,
                    "Start": None,
                    "End": None,
                    "MissingPct": 100.0,
                }
            )
            continue

        date_ser = pd.to_datetime(df["Date"], errors="coerce") if "Date" in df.columns else pd.Series(dtype="datetime64[ns]")
        rows.append(
            {
                "Interval": interval,
                "Rows": int(len(df)),
                "Start": date_ser.min(),
                "End": date_ser.max(),
                "MissingPct": float(df.isna().mean().mean() * 100.0),
            }
        )

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out.sort_values("Interval").reset_index(drop=True)
    return out


def load_candles_features_metadata(ticker: str, cfg: DataAPIConfig | None = None) -> dict:
    """
    Clean data API that returns:
    - raw multi-timeframe candles
    - merged model dataset
    - feature list
    - metadata quality tables
    """
    if cfg is None:
        cfg = DataAPIConfig()

    intervals = tuple(dict.fromkeys((cfg.base_interval,) + tuple(cfg.context_intervals)))
    timeframe_data = download_multi_timeframe_data(ticker, intervals=intervals, auto_adjust=cfg.auto_adjust)

    model_df, technical_feature_cols = build_multi_timeframe_dataset(
        timeframe_data,
        cfg=MultiTimeframeDatasetConfig(
            base_interval=cfg.base_interval,
            context_intervals=cfg.context_intervals,
            horizon_bars=cfg.horizon_bars,
            core_mode=cfg.core_mode,
            strategy_mode=cfg.strategy_mode,
        ),
    )

    price_quality = build_price_quality_report(timeframe_data)
    return {
        "ticker": str(ticker).strip().upper(),
        "timeframe_data": timeframe_data,
        "base_prices": timeframe_data[cfg.base_interval],
        "model_df": model_df,
        "technical_feature_cols": technical_feature_cols,
        "price_quality_report": price_quality,
        "metadata": {
            "base_interval": cfg.base_interval,
            "context_intervals": cfg.context_intervals,
            "horizon_bars": cfg.horizon_bars,
            "rows_model_df": int(len(model_df)),
        },
    }

"""Label separation, latest-bar scoring, and horizon embargo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering import MultiTimeframeDatasetConfig, build_multi_timeframe_dataset
from model import ModelConfig, purged_split_bounds, score_features, train_model
from strategy import prob_to_signal_quantile


def _ohlcv(n: int = 420, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.012, n)
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.001, n - 1))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2016-01-04", periods=n),
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(100_000, 500_000, n).astype(float),
        }
    )


def _labeled_frame(horizon: int = 7):
    raw = _ohlcv()
    model_df, cols = build_multi_timeframe_dataset(
        {"1d": raw},
        cfg=MultiTimeframeDatasetConfig(
            base_interval="1d",
            context_intervals=(),
            horizon_bars=horizon,
            technical_only=True,
            technical_min_forward_return=0.0,
        ),
    )
    return model_df, cols


def test_latest_bars_keep_features_and_drop_only_the_forward_label():
    horizon = 7
    model_df, cols = _labeled_frame(horizon)
    assert model_df["Target"].isna().sum() == horizon
    assert model_df["LabelComplete"].iloc[-1] == False  # noqa: E712
    assert bool(model_df["LabelComplete"].iloc[-(horizon + 1)])
    assert model_df.loc[model_df.index[-1], cols].notna().all()


def test_latest_bar_scores_without_a_target_column():
    model_df, cols = _labeled_frame(7)
    result = train_model(
        model_df,
        cfg=ModelConfig(model_type="gb", min_rows=120, test_fraction=0.2, horizon_bars=7, random_state=1),
        feature_cols=cols,
    )
    live = model_df.tail(1).drop(columns=["Target"])
    assert "Target" not in live.columns
    scored = score_features(result.model, live, cols)
    assert scored.shape == (1,)
    assert np.isfinite(scored[0])
    assert pd.isna(result.test_df["Target"].iloc[-1])
    assert np.isfinite(result.test_proba[-1])
    assert pd.Timestamp(model_df["Date"].iloc[-1]) >= pd.Timestamp(result.test_start_date)


def test_horizon_embargo_keeps_training_labels_out_of_the_test_window():
    horizon = 7
    train_end, test_start = purged_split_bounds(200, 0.2, horizon)
    assert test_start - train_end == horizon
    # Last trained label index is train_end - 1; its forward close is still before test_start.
    assert (train_end - 1) + horizon < test_start

    model_df, cols = _labeled_frame(horizon)
    result = train_model(
        model_df,
        cfg=ModelConfig(model_type="gb", min_rows=120, test_fraction=0.2, horizon_bars=horizon, random_state=1),
        feature_cols=cols,
    )
    dates = pd.to_datetime(model_df["Date"])
    positions = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    p = positions[pd.Timestamp(result.train_end_date)]
    q = positions[pd.Timestamp(result.test_start_date)]
    assert p + horizon < q
    assert result.embargo_bars == horizon


def test_future_probabilities_do_not_change_earlier_quantile_signals():
    history = np.linspace(0.2, 0.85, 40)
    original = prob_to_signal_quantile(history, buy_quantile=0.8, sell_quantile=0.2, min_history=8)
    extended = prob_to_signal_quantile(
        np.r_[history, [0.01, 0.99, 0.99, 0.01]],
        buy_quantile=0.8,
        sell_quantile=0.2,
        min_history=8,
    )
    assert list(original) == list(extended[: len(original)])

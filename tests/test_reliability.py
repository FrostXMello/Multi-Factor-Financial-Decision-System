"""Walk-forward embargo, calibration separation, and held-out explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model import ModelConfig
from model_evaluation import (
    apply_probability_calibrator,
    classification_reliability_metrics,
    evaluate_walk_forward,
    expected_calibration_error,
    fit_probability_calibrator,
    local_feature_influence,
    permutation_importance_frame,
    plan_walk_forward_folds,
)
from sklearn.ensemble import HistGradientBoostingClassifier


def _frame(n: int = 420, horizon: int = 3, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    close = 100 + np.cumsum(rng.normal(0.0, 0.4, size=n))
    frame = pd.DataFrame(
        {
            "Date": pd.bdate_range("2016-01-01", periods=n),
            "Open": close,
            "Close": close,
            "MA50": close,
            "F0": signal,
            "F1": rng.normal(size=n),
            "F2": rng.normal(size=n),
        }
    )
    future = pd.Series(signal).shift(-horizon)
    frame["Target"] = np.where(future.notna(), (future.to_numpy() > 0).astype(float), np.nan)
    return frame


def _cfg(**kwargs) -> ModelConfig:
    params = dict(model_type="gb", max_iter=12, horizon_bars=3, min_rows=40, random_state=7)
    params.update(kwargs)
    return ModelConfig(**params)


def test_walk_forward_folds_are_chronological_and_contiguous():
    plans = plan_walk_forward_folds(500, n_folds=4, min_train_fraction=0.5, horizon_bars=5)
    assert len(plans) >= 2
    assert plans[-1].test_end == 500
    for earlier, later in zip(plans, plans[1:]):
        assert earlier.test_start < later.test_start
        assert earlier.test_end == later.test_start
        assert earlier.fit_end <= later.fit_end


def test_horizon_embargo_on_every_planned_fold():
    horizon = 6
    plans = plan_walk_forward_folds(480, n_folds=4, min_train_fraction=0.5, horizon_bars=horizon)
    for plan in plans:
        assert plan.embargo_bars == horizon
        assert plan.test_start - plan.fit_end >= horizon
        assert plan.fit_end <= plan.test_start - horizon
        if plan.calib_start is not None:
            assert plan.fit_end + horizon <= plan.calib_start
            assert plan.calib_end is not None
            assert plan.calib_start < plan.calib_end
            assert plan.calib_end + horizon <= plan.test_start
            fit_idx = set(range(plan.fit_end))
            calib_idx = set(range(plan.calib_start, plan.calib_end))
            test_idx = set(range(plan.test_start, plan.test_end))
            assert fit_idx.isdisjoint(calib_idx)
            assert fit_idx.isdisjoint(test_idx)
            assert calib_idx.isdisjoint(test_idx)


def test_fitted_folds_keep_labels_out_of_their_test_window():
    frame = _frame()
    result = evaluate_walk_forward(
        frame,
        cfg=_cfg(),
        feature_cols=["F0", "F1", "F2"],
        n_folds=3,
        calibration="sigmoid",
        n_repeats=0,
    )
    assert len(result.folds) >= 2
    dates = pd.to_datetime(result.oos_frame["Date"])
    assert dates.is_monotonic_increasing
    for _, row in result.folds.iterrows():
        assert row.fit_last_pos + int(row.embargo_bars) < row.test_first_pos
        assert pd.Timestamp(row.fit_end_date) < pd.Timestamp(row.test_start_date)
        fit_idx = set(range(int(row.fit_end)))
        test_idx = set(range(int(row.test_start), int(row.test_end)))
        assert fit_idx.isdisjoint(test_idx)
        assert pd.notna(row.calib_start)
        assert int(row.fit_end) + int(row.embargo_bars) <= int(row.calib_start)
        assert int(row.calib_end) + int(row.embargo_bars) <= int(row.test_start)
        calib_idx = set(range(int(row.calib_start), int(row.calib_end)))
        assert calib_idx.isdisjoint(test_idx)
        assert calib_idx.isdisjoint(fit_idx)
        assert row.fit_last_pos + int(row.embargo_bars) < row.calib_first_pos
        assert row.calib_last_pos + int(row.embargo_bars) < row.test_first_pos


def test_short_history_refuses_walk_forward():
    with pytest.raises(ValueError, match="Not enough rows"):
        plan_walk_forward_folds(70, n_folds=4, min_train_fraction=0.5, horizon_bars=7)


def test_sigmoid_moves_miscalibrated_probabilities_and_ignores_one_class():
    raw = np.concatenate([np.full(80, 0.9), np.full(80, 0.1)])
    y = np.array([1] * 48 + [0] * 32 + [1] * 32 + [0] * 48)
    calibrator = fit_probability_calibrator(raw, y, method="sigmoid")
    adjusted = apply_probability_calibrator(calibrator, np.array([0.9, 0.1]))
    assert adjusted[0] < 0.9
    assert adjusted[1] > 0.1

    isotonic = fit_probability_calibrator(raw, y, method="isotonic")
    iso = apply_probability_calibrator(isotonic, np.array([0.9, 0.1]))
    assert 0.0 <= iso[0] <= 1.0 and iso[0] < 0.9
    assert 0.0 <= iso[1] <= 1.0 and iso[1] > 0.1

    assert fit_probability_calibrator(np.linspace(0.1, 0.9, 40), np.zeros(40, dtype=int), "sigmoid") is None
    raw_small = np.linspace(0.2, 0.8, 30)
    assert np.allclose(apply_probability_calibrator(None, raw_small), raw_small)


def test_uncalibrated_walk_forward_keeps_raw_probabilities():
    frame = _frame()
    result = evaluate_walk_forward(
        frame,
        cfg=_cfg(),
        feature_cols=["F0", "F1", "F2"],
        n_folds=3,
        calibration="none",
        n_repeats=0,
    )
    assert result.decision_method == "raw"
    assert np.allclose(result.oos_frame["Proba"], result.oos_frame["ProbaRaw"])
    assert result.oos_frame["ProbaCalibrated"].isna().all()
    assert result.calibration_window_metrics is None


def test_auto_calibration_uses_sigmoid_before_the_isotonic_sample_size():
    frame = _frame(n=420, horizon=3)
    result = evaluate_walk_forward(
        frame,
        cfg=_cfg(horizon_bars=3),
        feature_cols=["F0", "F1", "F2"],
        n_folds=3,
        calibration="auto",
        n_repeats=0,
    )
    assert result.decision_method == "sigmoid"
    assert result.oos_sigmoid_metrics is not None
    assert result.oos_isotonic_metrics is None
    calibrated = result.oos_frame["ProbaCalibrated"].to_numpy(dtype=float)
    assert np.isfinite(calibrated).all()
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))


def test_isotonic_is_fit_only_when_the_calibration_window_is_large_enough():
    frame = _frame(n=900, horizon=2, seed=3)
    result = evaluate_walk_forward(
        frame,
        cfg=_cfg(horizon_bars=2, max_iter=8),
        feature_cols=["F0", "F1", "F2"],
        n_folds=3,
        min_train_fraction=0.4,
        calibration="auto",
        n_repeats=0,
    )
    assert result.oos_isotonic_metrics is not None
    assert (result.folds["decision_method"] == "isotonic").any()
    assert result.oos_isotonic_metrics["n"] >= 80


def test_latest_bar_is_scored_by_a_model_that_did_not_train_on_it():
    frame = _frame()
    result = evaluate_walk_forward(
        frame,
        cfg=_cfg(),
        feature_cols=["F0", "F1", "F2"],
        n_folds=3,
        calibration="sigmoid",
        n_repeats=0,
    )
    assert len(result.tail_frame) > 0
    assert result.tail_frame["Target"].isna().all()
    assert result.explained_position == len(frame) - 1
    assert pd.Timestamp(result.explained_date) == pd.Timestamp(frame["Date"].iloc[-1])
    assert np.isfinite(result.explained_probability)
    assert 0.0 <= result.explained_probability <= 1.0
    assert result.explained_position > int(result.folds["fit_last_pos"].max())
    assert result.explained_position > int(result.folds["test_last_pos"].max())
    assert pd.Timestamp(result.explained_date) > pd.Timestamp(result.last_fit_end_date)
    assert np.isfinite(result.tail_frame["Proba"].to_numpy(dtype=float)).all()


def test_reliability_metrics_and_calibration_error():
    perfect_y = np.array([0, 0, 1, 1])
    perfect_p = np.array([0.0, 0.0, 1.0, 1.0])
    perfect = classification_reliability_metrics(perfect_y, perfect_p, min_rank_samples=4)
    assert perfect["accuracy"] == 1.0
    assert perfect["brier"] == pytest.approx(0.0, abs=1e-9)
    assert perfect["ece"] == pytest.approx(0.0, abs=1e-9)
    assert perfect["f1"] == 1.0

    y = np.array([0, 1] * 50)
    p = np.full(100, 0.9)
    ece, bins = expected_calibration_error(y, p, n_bins=10)
    assert ece == pytest.approx(0.4, abs=1e-9)
    assert len(bins) == 1
    metrics = classification_reliability_metrics(y, p)
    assert metrics["brier"] == pytest.approx(0.41, abs=1e-9)
    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert np.isfinite(metrics["log_loss"])

    one_class = classification_reliability_metrics(np.zeros(12, dtype=int), np.full(12, 0.2))
    assert np.isnan(one_class["roc_auc"])
    assert np.isnan(one_class["pr_auc"])
    assert np.isnan(one_class["balanced_accuracy"])
    assert one_class["brier"] == pytest.approx(0.04, abs=1e-9)


def test_permutation_importance_uses_held_out_rows_and_ranks_the_signal():
    rng = np.random.default_rng(4)
    n = 360
    f0 = rng.normal(size=n)
    noise = rng.normal(size=(n, 3))
    y = (3.0 * f0 + rng.normal(scale=0.25, size=n) > 0).astype(int)
    X = np.column_stack([f0, noise])
    names = ["F0", "N1", "N2", "N3"]
    split = 250
    model = HistGradientBoostingClassifier(max_iter=30, max_depth=3, learning_rate=0.08, random_state=4)
    model.fit(X[:split], y[:split])
    importance = permutation_importance_frame(model, X[split:], y[split:], names, n_repeats=6, random_state=4)
    assert list(importance["Feature"]).count("F0") == 1
    assert importance.iloc[0]["Feature"] == "F0"
    assert importance.iloc[0]["Importance"] > 0
    assert len(importance) == 4

    held = X[split:]
    medians = np.median(held, axis=0)
    extreme = int(np.argmax(np.abs(held[:, 0] - medians[0])))
    row = held[extreme]
    _base, local = local_feature_influence(model, row, medians, names)
    assert local.iloc[0]["Feature"] == "F0"
    assert np.isfinite(local["ProbabilityDelta"]).all()


def test_walk_forward_importance_is_computed_on_completed_folds():
    rng = np.random.default_rng(9)
    n = 460
    horizon = 2
    f0 = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "Date": pd.bdate_range("2017-01-01", periods=n),
            "F0": f0,
            "F1": rng.normal(size=n),
            "F2": rng.normal(size=n),
        }
    )
    target = np.full(n, np.nan)
    target[:-horizon] = (f0[:-horizon] > 0).astype(float)
    frame["Target"] = target
    frame.loc[frame.index[-1], "F0"] = 5.0
    result = evaluate_walk_forward(
        frame,
        cfg=_cfg(horizon_bars=horizon, max_iter=20),
        feature_cols=["F0", "F1", "F2"],
        n_folds=3,
        calibration="none",
        n_repeats=3,
    )
    assert len(result.importance) == 3
    assert result.importance.iloc[0]["Feature"] == "F0"
    assert set(result.importance.columns) >= {"Feature", "Importance", "FoldStd", "RepeatStd", "Folds"}
    assert result.importance["Folds"].min() >= 1
    assert len(result.local_explanation) == 3
    assert result.local_explanation.iloc[0]["Feature"] == "F0"


def _ohlcv(n: int = 520, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=n)))
    open_ = close * (1 + rng.normal(0, 0.001, size=n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, size=n))
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(1_000_000, 2_000_000, size=n),
        }
    )


def _weekly(daily: pd.DataFrame) -> pd.DataFrame:
    bars = daily.set_index("Date").sort_index()
    weekly = bars.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return weekly.dropna().reset_index()


def test_technical_pipeline_trades_walk_forward_probabilities_and_keeps_net_costs():
    from four_model_pipeline import FourModelConfig, _run_four_model_pipeline_with_data

    daily = _ohlcv()
    out = _run_four_model_pipeline_with_data(
        "TEST",
        FourModelConfig(
            initial_capital=100_000,
            model_max_iter=12,
            walk_forward_folds=3,
            permutation_repeats=2,
        ),
        {"1d": daily, "1wk": _weekly(daily)},
    )
    reliability = out["reliability"]
    assert reliability["evaluation"] == "walk_forward"
    assert out["strategy_mode"] == "technical_only"
    backtest = out["backtest"]
    assert backtest["total_return_pct"] == pytest.approx(backtest["net_return_pct"])
    assert "gross_return_pct" in backtest
    assert "total_transaction_costs" in backtest
    first_fold = pd.Timestamp(reliability["folds"]["test_start_date"].iloc[0])
    assert pd.Timestamp(out["test_df"]["Date"].iloc[0]) == first_fold
    model_dates = pd.to_datetime(out["model_df"]["Date"])
    assert pd.Timestamp(out["test_df"]["Date"].iloc[0]) < model_dates.iloc[int(len(model_dates) * 0.75)]
    assert np.isfinite(out["latest"]["proba"])
    scored = out["test_df"]
    calibrated = scored["ProbaCalibrated"].notna()
    if calibrated.any():
        assert np.allclose(
            scored.loc[calibrated, "Proba"].to_numpy(dtype=float),
            scored.loc[calibrated, "ProbaCalibrated"].to_numpy(dtype=float),
        )


def test_pipeline_falls_back_to_the_holdout_when_walk_forward_fails(monkeypatch):
    from four_model_pipeline import FourModelConfig, _run_four_model_pipeline_with_data

    def _boom(*args, **kwargs):
        raise ValueError("not enough for this test")

    monkeypatch.setattr("four_model_pipeline.evaluate_walk_forward", _boom)
    daily = _ohlcv()
    out = _run_four_model_pipeline_with_data(
        "TEST",
        FourModelConfig(initial_capital=100_000, model_max_iter=8),
        {"1d": daily, "1wk": _weekly(daily)},
    )
    assert out["reliability"]["evaluation"] == "holdout"
    assert out["reliability"]["decision_method"] == "raw"
    model_dates = pd.to_datetime(out["model_df"]["Date"])
    assert pd.Timestamp(out["test_df"]["Date"].iloc[0]) >= model_dates.iloc[int(len(model_dates) * 0.7)]


def _reliability_panel_payload() -> dict:
    metrics = {
        "n": 50,
        "accuracy": 0.55,
        "precision": 0.5,
        "recall": 0.4,
        "f1": 0.44,
        "balanced_accuracy": 0.52,
        "roc_auc": 0.6,
        "pr_auc": 0.57,
        "brier": 0.24,
        "log_loss": 0.68,
        "ece": 0.08,
    }
    payload = {
        "evaluation": "walk_forward",
        "decision_method": "sigmoid",
        "insample_accuracy": 0.7,
        "oos_metrics": metrics,
        "oos_raw_metrics": {**metrics, "brier": 0.25, "ece": 0.1},
        "oos_sigmoid_metrics": metrics,
        "oos_isotonic_metrics": None,
        "calibration_window_metrics": {"brier_raw": 0.22, "brier_calibrated": 0.2},
        "folds": pd.DataFrame(
            [
                {
                    "fold": 1,
                    "test_start_date": pd.Timestamp("2020-01-01"),
                    "test_end_date": pd.Timestamp("2020-06-01"),
                    "n_fit": 100,
                    "n_calib": 40,
                    "n_test": 50,
                    "decision_method": "sigmoid",
                    "accuracy": 0.55,
                    "brier_raw": 0.25,
                    "brier_decision": 0.24,
                    "brier_sigmoid": 0.24,
                    "brier_isotonic": np.nan,
                }
            ]
        ),
        "reliability_bins": pd.DataFrame({"confidence": [0.2, 0.8], "frequency": [0.3, 0.7]}),
        "reliability_bins_raw": pd.DataFrame({"confidence": [0.2, 0.8], "frequency": [0.4, 0.6]}),
        "confusion": pd.DataFrame(
            [[10, 2], [3, 8]],
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        ),
        "importance": pd.DataFrame(
            {
                "Feature": ["MA50", "RSI14"],
                "Importance": [0.04, 0.01],
                "FoldStd": [0.01, 0.02],
                "RepeatStd": [0.005, 0.004],
                "Folds": [3, 3],
            }
        ),
        "local_explanation": pd.DataFrame(
            {
                "Feature": ["MA50"],
                "HeldOutMedian": [1.0],
                "Probability": [0.62],
                "ReplacedProbability": [0.48],
                "ProbabilityDelta": [0.14],
            }
        ),
        "oos_proba_decision": np.array([0.2, 0.8, 0.6]),
        "explained_date": pd.Timestamp("2020-06-02"),
        "explained_probability": 0.62,
    }
    return payload


def test_reliability_panel_renders_scores_without_a_narrative():
    from streamlit.testing.v1 import AppTest
    import textwrap

    source = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(__import__('pathlib').Path(__file__).resolve().parents[1])!r})
        from tests.test_reliability import _reliability_panel_payload
        from analytics_dashboard import render_model_reliability
        render_model_reliability(_reliability_panel_payload())
        """
    )
    app = AppTest.from_string(source, default_timeout=30)
    app.run()
    assert not app.exception
    chunks = []
    for name in ("markdown", "caption", "info", "text"):
        for block in getattr(app, name, []):
            chunks.append(str(getattr(block, "value", block)))
    text = " ".join(chunks)
    assert "not a causal" in text
    assert "out-of-sample" in text.lower()
    assert len(app.dataframe) >= 3

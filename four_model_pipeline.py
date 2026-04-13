from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np
import pandas as pd

from backtest import (
    BacktestConfig,
    PortfolioBacktestConfig,
    backtest_long_cash,
    backtest_portfolio_long_cash,
    backtest_portfolio_topn_long_cash,
)
from core_config import (
    CORE_MODE,
    CORE_BASE_INTERVAL,
    CORE_BUY_THRESHOLD,
    CORE_CONTEXT_INTERVALS,
    CORE_HORIZON_BARS,
    CORE_MACRO_FUSION_WEIGHT,
    CORE_MAX_TRADES_PER_DAY,
    CORE_MIN_MINUTES_BETWEEN_TRADES,
    CORE_MODEL_TYPE,
    CORE_RISK_CONFIG,
    CORE_SELL_THRESHOLD,
    CORE_TECH_FUSION_WEIGHT,
    TECHNICAL_MIN_FORWARD_RETURN,
    TECHNICAL_MODE,
    TECHNICAL_PORTFOLIO_TOP_N,
    assert_optimizer_allowed,
)
from data_loader import download_multi_timeframe_data
from feature_engineering import MultiTimeframeDatasetConfig, build_multi_timeframe_dataset
from fusion_model import FusionConfig, core_simple_fusion, meta_fusion, weighted_fusion
from macro_model import MacroConfig, build_macro_features, infer_macro_probability
from micro_model import MicroConfig, infer_micro_probability
from model import ModelConfig, train_model, walk_forward_validate
from risk_model import RiskConfig, apply_risk_gating, compute_risk_frame
from strategy import apply_daily_top_n_buys, attach_signals, attach_technical_trend_signals, enforce_trade_frequency
from strategy_modes import (
    STRATEGY_MEAN_REVERSION,
    STRATEGY_MOMENTUM,
    STRATEGY_MULTI_FACTOR,
    attach_mean_reversion_strategy_signals,
    attach_momentum_strategy_signals,
    normalize_strategy_mode,
)


@dataclass(frozen=True)
class FourModelConfig:
    base_interval: str = "1m"
    context_intervals: tuple[str, ...] = ("5m", "15m", "60m", "1d")
    horizon_bars: int = 5
    model_type: str = "gb"
    threshold_mode: str = "quantile"
    buy_threshold: float = 0.6
    sell_threshold: float = 0.4
    buy_quantile: float = 0.7
    sell_quantile: float = 0.3
    max_trades_per_day: int = 12
    min_minutes_between_trades: int = 5
    initial_capital: float = 100000.0
    use_meta_fusion: bool = False
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    fast_mode: bool = False
    compute_walk_forward: bool = True
    # CORE_MODE only: include macro in 0.7 Tech + 0.3 Macro blend; if False, technical proba only.
    core_include_macro: bool = True
    # Strategy selection: multi_factor (fusion stack), momentum (trend + ML proba), mean_reversion (Z-score rules + optional ML display).
    strategy_mode: str = STRATEGY_MULTI_FACTOR
    # TECHNICAL_MODE portfolio: max concurrent names (capital concentrated in top probabilities).
    portfolio_top_n: int = 3


def _effective_four_model_config(cfg: FourModelConfig) -> FourModelConfig:
    """
    TECHNICAL_MODE wins first: single GB model, daily(+weekly) price features, no fusion/macro/micro.
    Else CORE_MODE applies legacy evaluation defaults.
    """
    if TECHNICAL_MODE:
        return replace(
            cfg,
            base_interval=CORE_BASE_INTERVAL,
            context_intervals=CORE_CONTEXT_INTERVALS,
            horizon_bars=CORE_HORIZON_BARS,
            model_type=CORE_MODEL_TYPE,
            use_meta_fusion=False,
            core_include_macro=False,
            threshold_mode="fixed",
            buy_threshold=CORE_BUY_THRESHOLD,
            sell_threshold=CORE_SELL_THRESHOLD,
            max_trades_per_day=CORE_MAX_TRADES_PER_DAY,
            min_minutes_between_trades=CORE_MIN_MINUTES_BETWEEN_TRADES,
            compute_walk_forward=False,
            fast_mode=True,
            risk_config=CORE_RISK_CONFIG,
            portfolio_top_n=int(TECHNICAL_PORTFOLIO_TOP_N),
        )
    if CORE_MODE:
        return replace(
            cfg,
            base_interval=CORE_BASE_INTERVAL,
            context_intervals=CORE_CONTEXT_INTERVALS,
            horizon_bars=CORE_HORIZON_BARS,
            model_type=CORE_MODEL_TYPE,
            use_meta_fusion=False,
            core_include_macro=False,
            threshold_mode="fixed",
            buy_threshold=CORE_BUY_THRESHOLD,
            sell_threshold=CORE_SELL_THRESHOLD,
            max_trades_per_day=CORE_MAX_TRADES_PER_DAY,
            min_minutes_between_trades=CORE_MIN_MINUTES_BETWEEN_TRADES,
            compute_walk_forward=False,
            fast_mode=True,
            risk_config=CORE_RISK_CONFIG,
        )
    return cfg


@dataclass(frozen=True)
class OptimizationConfig:
    max_evals: int = 40
    random_state: int = 42


def _run_four_model_pipeline_with_data(
    ticker: str,
    cfg: FourModelConfig,
    timeframe_data: dict[str, pd.DataFrame],
    *,
    compute_walk_forward: bool | None = None,
    portfolio_batch: bool = False,
) -> dict:
    cfg = _effective_four_model_config(cfg)
    if compute_walk_forward is None:
        compute_walk_forward = cfg.compute_walk_forward
    if TECHNICAL_MODE or CORE_MODE:
        compute_walk_forward = False

    strategy_key = normalize_strategy_mode(cfg.strategy_mode)
    if TECHNICAL_MODE:
        strategy_key = "technical_only"
    elif strategy_key != STRATEGY_MULTI_FACTOR:
        compute_walk_forward = False

    if cfg.base_interval not in timeframe_data:
        raise ValueError(f"Base interval not downloaded: {cfg.base_interval}")

    prices = timeframe_data[cfg.base_interval]

    if TECHNICAL_MODE:
        # Price-only feature matrix + forward horizon label (meaningful move, not 1-bar noise).
        model_df, technical_feature_cols = build_multi_timeframe_dataset(
            timeframe_data,
            cfg=MultiTimeframeDatasetConfig(
                base_interval=cfg.base_interval,
                context_intervals=cfg.context_intervals,
                horizon_bars=cfg.horizon_bars,
                core_mode=False,
                strategy_mode=STRATEGY_MULTI_FACTOR,
                technical_only=True,
                technical_min_forward_return=TECHNICAL_MIN_FORWARD_RETURN,
            ),
        )
    else:
        model_df, technical_feature_cols = build_multi_timeframe_dataset(
            timeframe_data,
            cfg=MultiTimeframeDatasetConfig(
                base_interval=cfg.base_interval,
                context_intervals=cfg.context_intervals,
                horizon_bars=cfg.horizon_bars,
                core_mode=CORE_MODE,
                strategy_mode=strategy_key,
                technical_only=False,
            ),
        )

    tech_result = train_model(
        model_df,
        cfg=ModelConfig(model_type=cfg.model_type, random_state=42),
        feature_cols=technical_feature_cols,
    )
    if compute_walk_forward:
        wf = walk_forward_validate(
            model_df,
            cfg=ModelConfig(model_type=cfg.model_type, random_state=42),
            feature_cols=technical_feature_cols,
            n_folds=5,
        )
        wf_mean = float(wf.mean_score)
        wf_folds = wf.fold_scores
    else:
        wf_mean = float("nan")
        wf_folds = []

    test_df = tech_result.test_df.reset_index(drop=True).copy()

    if TECHNICAL_MODE:
        # Single probability stream; trend confirmation on MA50 (no RSI band, no fusion).
        signals_df = attach_technical_trend_signals(
            test_df,
            tech_result.test_proba,
            buy_threshold=cfg.buy_threshold,
            sell_threshold=cfg.sell_threshold,
        )
        if portfolio_batch:
            return {
                "pre_risk_signals": signals_df,
                "prices": prices,
                "model_df": model_df,
                "technical_test_accuracy": float(tech_result.test_accuracy),
                "walk_forward_mean_accuracy": wf_mean,
                "walk_forward_fold_scores": wf_folds,
                "trained_model": tech_result.model,
                "feature_names": list(technical_feature_cols),
                "strategy_mode": strategy_key,
                "ticker": ticker,
            }
    elif strategy_key == STRATEGY_MOMENTUM:
        # Trend-following sleeve: model proba × structural trend/RSI gates (no macro/micro fusion).
        signals_df = attach_momentum_strategy_signals(
            test_df,
            tech_result.test_proba,
            buy_threshold=cfg.buy_threshold,
            sell_threshold=cfg.sell_threshold,
        )
    elif strategy_key == STRATEGY_MEAN_REVERSION:
        # Stat-arb style bounds: Z-score rules drive trades; classifier proba is diagnostic only.
        signals_df = attach_mean_reversion_strategy_signals(test_df, tech_result.test_proba)
    else:
        test_dates = test_df["Date"]

        macro_features = build_macro_features(MacroConfig(horizon_days=1))
        macro_proba_full = infer_macro_probability(macro_features, model_df["Date"])
        if CORE_MODE:
            micro_proba_full = pd.Series(0.5, index=model_df.index)
        else:
            micro_cfg = MicroConfig(max_depth=3, max_iter=120) if cfg.fast_mode else MicroConfig()
            micro_proba_full = infer_micro_probability(model_df, ticker, cfg=micro_cfg)
        tech_proba_full = pd.Series(0.0, index=model_df.index, name="TechProba")
        tech_proba_full.iloc[len(model_df) - len(tech_result.test_proba) :] = tech_result.test_proba
        tech_proba_full = tech_proba_full.ffill().bfill()

        feature_stack = pd.DataFrame(
            {
                "Date": model_df["Date"].values,
                "MacroProba": macro_proba_full.values,
                "MicroProba": micro_proba_full.values,
                "TechProba": tech_proba_full.values,
                "Target": model_df["Target"].values,
            }
        )
        feature_stack_test = feature_stack[feature_stack["Date"].isin(test_dates)].reset_index(drop=True)
        if CORE_MODE:
            alpha_test = core_simple_fusion(
                feature_stack_test["TechProba"],
                feature_stack_test["MacroProba"],
                include_macro=cfg.core_include_macro,
                tech_weight=CORE_TECH_FUSION_WEIGHT,
                macro_weight=CORE_MACRO_FUSION_WEIGHT,
            )
        elif cfg.use_meta_fusion:
            alpha_full = meta_fusion(feature_stack[["MacroProba", "MicroProba", "TechProba"]], feature_stack["Target"], FusionConfig())
            alpha_test = alpha_full[feature_stack["Date"].isin(test_dates)].reset_index(drop=True)
        else:
            alpha_test = weighted_fusion(
                feature_stack_test["MacroProba"],
                feature_stack_test["MicroProba"],
                feature_stack_test["TechProba"],
                FusionConfig(),
            )

        signals_df = attach_signals(
            test_df,
            alpha_test.values,
            threshold_mode="fixed" if CORE_MODE else cfg.threshold_mode,
            buy_threshold=cfg.buy_threshold,
            sell_threshold=cfg.sell_threshold,
            buy_quantile=cfg.buy_quantile,
            sell_quantile=cfg.sell_quantile,
        )

    risk_frame = compute_risk_frame(prices, cfg=cfg.risk_config)
    gated_df = apply_risk_gating(signals_df, risk_frame)
    gated_df = enforce_trade_frequency(
        gated_df,
        max_trades_per_day=cfg.max_trades_per_day,
        min_minutes_between_trades=cfg.min_minutes_between_trades,
    )
    backtest = backtest_long_cash(gated_df, cfg=BacktestConfig(initial_capital=cfg.initial_capital))

    latest_row = gated_df.iloc[-1]
    trades = int(gated_df["Signal"].isin(["BUY", "SELL"]).sum())
    if len(gated_df) > 0:
        date_ser = pd.to_datetime(gated_df["Date"])
        active_days = max(1, int(date_ser.dt.date.nunique()))
        trades_per_day = float(trades / active_days)
    else:
        trades_per_day = 0.0

    return {
        "prices": prices,
        "model_df": model_df,
        "test_df": gated_df,
        "backtest": backtest,
        "technical_test_accuracy": float(tech_result.test_accuracy),
        "walk_forward_mean_accuracy": wf_mean,
        "walk_forward_fold_scores": wf_folds,
        "trades": trades,
        "trades_per_day": trades_per_day,
        "strategy_mode": strategy_key,
        "trained_model": tech_result.model,
        "feature_names": list(technical_feature_cols),
        "latest": {
            "signal": str(latest_row["Signal"]),
            "proba": float(latest_row["Proba"]),
            "risk_score": float(latest_row["RiskScore"]),
            "position_size": float(latest_row["PositionSize"]),
            "trade_date": latest_row["Date"],
        },
    }


def _detect_vol_regime(prices: pd.DataFrame, base_risk_cfg: RiskConfig) -> str:
    risk_frame = compute_risk_frame(prices, cfg=base_risk_cfg)
    vol = risk_frame["RealizedVol"].dropna()
    if len(vol) < 30:
        return "normal"
    q33 = float(vol.quantile(0.33))
    q67 = float(vol.quantile(0.67))
    latest = float(vol.iloc[-1])
    if latest <= q33:
        return "low"
    if latest >= q67:
        return "high"
    return "normal"


def _score_candidate(result: dict) -> float:
    bt = result["backtest"]
    ret = float(bt.get("total_return_pct", 0.0))
    sharpe = float(bt.get("sharpe", 0.0))
    if not np.isfinite(sharpe):
        sharpe = 0.0
    max_dd = abs(float(bt.get("max_drawdown_pct", 0.0)))
    trades_per_day = float(result.get("trades_per_day", 0.0))
    accuracy = float(result.get("technical_test_accuracy", 0.0))

    # Reward return/sharpe/accuracy and active intraday execution, penalize drawdown.
    return 1.0 * ret + 4.0 * sharpe + 25.0 * accuracy + 0.8 * trades_per_day - 0.35 * max_dd


def _score_portfolio_candidate(result: dict) -> float:
    portfolio_bt = result["portfolio_backtest"]
    ret = float(portfolio_bt.get("total_return_pct", 0.0))
    sharpe = float(portfolio_bt.get("sharpe", 0.0))
    if not np.isfinite(sharpe):
        sharpe = 0.0
    max_dd = abs(float(portfolio_bt.get("max_drawdown_pct", 0.0)))
    mean_accuracy = float(result.get("mean_accuracy", 0.0))
    mean_trades_per_day = float(result.get("mean_trades_per_day", 0.0))
    success_ratio = float(result.get("success_ratio", 0.0))

    return 1.2 * ret + 5.0 * sharpe + 20.0 * mean_accuracy + 0.6 * mean_trades_per_day + 10.0 * success_ratio - 0.4 * max_dd


def _detect_portfolio_vol_regime(prices_by_ticker: dict[str, pd.DataFrame], base_risk_cfg: RiskConfig) -> str:
    vals: list[float] = []
    for prices in prices_by_ticker.values():
        try:
            risk_frame = compute_risk_frame(prices, cfg=base_risk_cfg)
            vol = risk_frame["RealizedVol"].dropna()
            if len(vol) > 0:
                vals.append(float(vol.iloc[-1]))
        except Exception:
            continue
    if len(vals) < 3:
        return "normal"
    med = float(np.median(vals))
    q33 = float(np.quantile(vals, 0.33))
    q67 = float(np.quantile(vals, 0.67))
    if med <= q33:
        return "low"
    if med >= q67:
        return "high"
    return "normal"


def optimize_four_model_pipeline(
    ticker: str,
    base_cfg: FourModelConfig | None = None,
    opt_cfg: OptimizationConfig | None = None,
) -> dict:
    """
    Regime-aware parameter search over thresholds, trade-frequency, horizon, and risk controls.
    """
    assert_optimizer_allowed()
    if base_cfg is None:
        base_cfg = FourModelConfig()
    if opt_cfg is None:
        opt_cfg = OptimizationConfig()
    if opt_cfg.max_evals <= 0:
        raise ValueError("max_evals must be >= 1")

    intervals = tuple(dict.fromkeys((base_cfg.base_interval,) + tuple(base_cfg.context_intervals)))
    timeframe_data = download_multi_timeframe_data(ticker, intervals=intervals, auto_adjust=False)
    prices = timeframe_data.get(base_cfg.base_interval)
    if prices is None or prices.empty:
        raise ValueError(f"No base timeframe data found for {base_cfg.base_interval}")

    regime = _detect_vol_regime(prices, base_cfg.risk_config)

    if regime == "high":
        buy_q = [0.72, 0.78, 0.84]
        sell_q = [0.16, 0.22, 0.28]
        max_trades = [4, 8, 12]
        min_gap = [5, 10, 15]
        hard_vol = [0.04, 0.05]
        hard_dd = [0.12, 0.15]
    elif regime == "low":
        buy_q = [0.58, 0.64, 0.70]
        sell_q = [0.30, 0.36, 0.42]
        max_trades = [10, 16, 24]
        min_gap = [1, 3, 5]
        hard_vol = [0.05, 0.06]
        hard_dd = [0.15, 0.20]
    else:
        buy_q = [0.65, 0.72, 0.78]
        sell_q = [0.22, 0.30, 0.36]
        max_trades = [8, 12, 18]
        min_gap = [3, 5, 10]
        hard_vol = [0.045, 0.05, 0.055]
        hard_dd = [0.13, 0.15, 0.18]

    horizon_opts = sorted(set([max(1, base_cfg.horizon_bars - 2), base_cfg.horizon_bars, base_cfg.horizon_bars + 3]))

    grid = list(product(horizon_opts, buy_q, sell_q, max_trades, min_gap, hard_vol, hard_dd))
    rng = np.random.default_rng(opt_cfg.random_state)
    if len(grid) > opt_cfg.max_evals:
        idx = rng.choice(len(grid), size=opt_cfg.max_evals, replace=False)
        grid = [grid[int(i)] for i in idx]

    leaderboard: list[dict] = []
    best_result: dict | None = None
    best_cfg: FourModelConfig | None = None
    best_score = -1e18

    for candidate in grid:
        horizon_bars, buy_quantile, sell_quantile, max_trades_per_day, min_minutes_between_trades, hard_vol_stop, hard_drawdown_stop = candidate
        if not (0 < sell_quantile < buy_quantile < 1):
            continue

        risk_cfg = replace(
            base_cfg.risk_config,
            hard_vol_stop=float(hard_vol_stop),
            hard_drawdown_stop=float(hard_drawdown_stop),
        )
        run_cfg = replace(
            base_cfg,
            threshold_mode="quantile",
            horizon_bars=int(horizon_bars),
            buy_quantile=float(buy_quantile),
            sell_quantile=float(sell_quantile),
            max_trades_per_day=int(max_trades_per_day),
            min_minutes_between_trades=int(min_minutes_between_trades),
            risk_config=risk_cfg,
        )

        try:
            candidate_result = _run_four_model_pipeline_with_data(
                ticker,
                run_cfg,
                timeframe_data,
                compute_walk_forward=False,
            )
            score = _score_candidate(candidate_result)
            row = {
                "score": float(score),
                "regime": regime,
                "horizon_bars": int(run_cfg.horizon_bars),
                "buy_quantile": float(run_cfg.buy_quantile),
                "sell_quantile": float(run_cfg.sell_quantile),
                "max_trades_per_day": int(run_cfg.max_trades_per_day),
                "min_minutes_between_trades": int(run_cfg.min_minutes_between_trades),
                "hard_vol_stop": float(run_cfg.risk_config.hard_vol_stop),
                "hard_drawdown_stop": float(run_cfg.risk_config.hard_drawdown_stop),
                "return_pct": float(candidate_result["backtest"]["total_return_pct"]),
                "max_drawdown_pct": float(candidate_result["backtest"]["max_drawdown_pct"]),
                "sharpe": float(candidate_result["backtest"]["sharpe"]),
                "accuracy": float(candidate_result["technical_test_accuracy"]),
                "trades_per_day": float(candidate_result["trades_per_day"]),
            }
            leaderboard.append(row)
            if score > best_score:
                best_score = score
                best_result = candidate_result
                best_cfg = run_cfg
        except Exception:
            continue

    if best_result is None or best_cfg is None:
        raise RuntimeError("Optimization failed for all candidate configurations.")

    board_df = pd.DataFrame(leaderboard).sort_values("score", ascending=False).reset_index(drop=True)
    return {
        "regime": regime,
        "best_score": float(best_score),
        "best_config": best_cfg,
        "best_result": best_result,
        "leaderboard": board_df,
        "evaluated": int(len(leaderboard)),
    }


def optimize_portfolio_four_model_pipeline(
    tickers: list[str],
    base_cfg: FourModelConfig | None = None,
    opt_cfg: OptimizationConfig | None = None,
) -> dict:
    """
    Optimize a single global four-model config for a basket by maximizing
    aggregated portfolio metrics.
    """
    assert_optimizer_allowed()
    if base_cfg is None:
        base_cfg = FourModelConfig()
    if opt_cfg is None:
        opt_cfg = OptimizationConfig()
    if not tickers:
        raise ValueError("tickers list cannot be empty")
    if opt_cfg.max_evals <= 0:
        raise ValueError("max_evals must be >= 1")

    cleaned = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not cleaned:
        raise ValueError("No valid tickers provided")

    intervals = tuple(dict.fromkeys((base_cfg.base_interval,) + tuple(base_cfg.context_intervals)))
    timeframe_data_by_ticker: dict[str, dict[str, pd.DataFrame]] = {}
    prices_by_ticker: dict[str, pd.DataFrame] = {}

    for t in cleaned:
        try:
            tf_data = download_multi_timeframe_data(t, intervals=intervals, auto_adjust=False)
            prices = tf_data.get(base_cfg.base_interval)
            if prices is None or prices.empty:
                continue
            timeframe_data_by_ticker[t] = tf_data
            prices_by_ticker[t] = prices
        except Exception:
            continue

    if not timeframe_data_by_ticker:
        raise RuntimeError("Unable to load data for any ticker in basket.")

    regime = _detect_portfolio_vol_regime(prices_by_ticker, base_cfg.risk_config)

    if regime == "high":
        buy_q = [0.72, 0.78, 0.84]
        sell_q = [0.16, 0.22, 0.28]
        max_trades = [4, 8, 12]
        min_gap = [5, 10, 15]
        hard_vol = [0.04, 0.05]
        hard_dd = [0.12, 0.15]
    elif regime == "low":
        buy_q = [0.58, 0.64, 0.70]
        sell_q = [0.30, 0.36, 0.42]
        max_trades = [10, 16, 24]
        min_gap = [1, 3, 5]
        hard_vol = [0.05, 0.06]
        hard_dd = [0.15, 0.20]
    else:
        buy_q = [0.65, 0.72, 0.78]
        sell_q = [0.22, 0.30, 0.36]
        max_trades = [8, 12, 18]
        min_gap = [3, 5, 10]
        hard_vol = [0.045, 0.05, 0.055]
        hard_dd = [0.13, 0.15, 0.18]

    horizon_opts = sorted(set([max(1, base_cfg.horizon_bars - 2), base_cfg.horizon_bars, base_cfg.horizon_bars + 3]))
    grid = list(product(horizon_opts, buy_q, sell_q, max_trades, min_gap, hard_vol, hard_dd))

    rng = np.random.default_rng(opt_cfg.random_state)
    if len(grid) > opt_cfg.max_evals:
        idx = rng.choice(len(grid), size=opt_cfg.max_evals, replace=False)
        grid = [grid[int(i)] for i in idx]

    leaderboard: list[dict] = []
    best_score = -1e18
    best_cfg: FourModelConfig | None = None
    best_result: dict | None = None

    for candidate in grid:
        horizon_bars, buy_quantile, sell_quantile, max_trades_per_day, min_minutes_between_trades, hard_vol_stop, hard_drawdown_stop = candidate
        if not (0 < sell_quantile < buy_quantile < 1):
            continue

        risk_cfg = replace(
            base_cfg.risk_config,
            hard_vol_stop=float(hard_vol_stop),
            hard_drawdown_stop=float(hard_drawdown_stop),
        )
        run_cfg = replace(
            base_cfg,
            threshold_mode="quantile",
            horizon_bars=int(horizon_bars),
            buy_quantile=float(buy_quantile),
            sell_quantile=float(sell_quantile),
            max_trades_per_day=int(max_trades_per_day),
            min_minutes_between_trades=int(min_minutes_between_trades),
            risk_config=risk_cfg,
        )

        signals_by_ticker: dict[str, pd.DataFrame] = {}
        ticker_rows: list[dict] = []
        for t, tf_data in timeframe_data_by_ticker.items():
            try:
                out = _run_four_model_pipeline_with_data(
                    t,
                    run_cfg,
                    tf_data,
                    compute_walk_forward=False,
                )
                signals_by_ticker[t] = out["test_df"]
                ticker_rows.append(
                    {
                        "Ticker": t,
                        "Latest_Signal": out["latest"]["signal"],
                        "Latest_Proba": float(out["latest"]["proba"]),
                        "Test_Accuracy": float(out["technical_test_accuracy"]),
                        "Trades_Per_Day": float(out["trades_per_day"]),
                    }
                )
            except Exception:
                continue

        if not signals_by_ticker:
            continue

        portfolio_backtest = backtest_portfolio_long_cash(
            signals_by_ticker,
            cfg=PortfolioBacktestConfig(initial_capital=float(base_cfg.initial_capital), allocation="equal"),
        )
        table_df = pd.DataFrame(ticker_rows).sort_values("Ticker").reset_index(drop=True)

        candidate_result = {
            "portfolio_backtest": portfolio_backtest,
            "table_df": table_df,
            "signals_by_ticker": signals_by_ticker,
            "success_ratio": float(len(signals_by_ticker) / max(1, len(cleaned))),
            "mean_accuracy": float(table_df["Test_Accuracy"].mean()) if len(table_df) else 0.0,
            "mean_trades_per_day": float(table_df["Trades_Per_Day"].mean()) if len(table_df) else 0.0,
        }
        score = _score_portfolio_candidate(candidate_result)

        row = {
            "score": float(score),
            "regime": regime,
            "horizon_bars": int(run_cfg.horizon_bars),
            "buy_quantile": float(run_cfg.buy_quantile),
            "sell_quantile": float(run_cfg.sell_quantile),
            "max_trades_per_day": int(run_cfg.max_trades_per_day),
            "min_minutes_between_trades": int(run_cfg.min_minutes_between_trades),
            "hard_vol_stop": float(run_cfg.risk_config.hard_vol_stop),
            "hard_drawdown_stop": float(run_cfg.risk_config.hard_drawdown_stop),
            "portfolio_return_pct": float(portfolio_backtest["total_return_pct"]),
            "portfolio_max_drawdown_pct": float(portfolio_backtest.get("max_drawdown_pct", 0.0)),
            "portfolio_sharpe": float(portfolio_backtest.get("sharpe", float("nan"))),
            "mean_accuracy": float(candidate_result["mean_accuracy"]),
            "mean_trades_per_day": float(candidate_result["mean_trades_per_day"]),
            "success_ratio": float(candidate_result["success_ratio"]),
        }
        leaderboard.append(row)

        if score > best_score:
            best_score = score
            best_cfg = run_cfg
            best_result = candidate_result

    if best_cfg is None or best_result is None or not leaderboard:
        raise RuntimeError("Portfolio optimization failed for all candidate configurations.")

    board_df = pd.DataFrame(leaderboard).sort_values("score", ascending=False).reset_index(drop=True)
    return {
        "regime": regime,
        "best_score": float(best_score),
        "best_config": best_cfg,
        "best_result": best_result,
        "leaderboard": board_df,
        "evaluated": int(len(leaderboard)),
    }


def run_portfolio_four_model_pipeline(
    tickers: list[str],
    cfg: FourModelConfig | None = None,
) -> dict:
    """
    Run a single global config across a ticker basket (no optimization).
    """
    if cfg is None:
        cfg = FourModelConfig()
    cleaned = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not cleaned:
        raise ValueError("No valid tickers provided")

    intervals = tuple(dict.fromkeys((cfg.base_interval,) + tuple(cfg.context_intervals)))
    timeframe_data_by_ticker: dict[str, dict[str, pd.DataFrame]] = {}
    for t in cleaned:
        try:
            timeframe_data_by_ticker[t] = download_multi_timeframe_data(t, intervals=intervals, auto_adjust=False)
        except Exception:
            continue

    if not timeframe_data_by_ticker:
        raise RuntimeError("Unable to load data for any ticker in basket.")

    signals_by_ticker: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    partial_by_ticker: dict[str, dict] = {}

    if TECHNICAL_MODE:
        for t, tf_data in timeframe_data_by_ticker.items():
            try:
                partial = _run_four_model_pipeline_with_data(
                    t,
                    cfg,
                    tf_data,
                    compute_walk_forward=False,
                    portfolio_batch=True,
                )
                partial_by_ticker[t] = partial
            except Exception:
                continue
        if not partial_by_ticker:
            raise RuntimeError("No ticker produced a valid pipeline output.")
        pre_signals = {t: partial_by_ticker[t]["pre_risk_signals"] for t in partial_by_ticker}
        after_topn = apply_daily_top_n_buys(pre_signals, top_n=int(cfg.portfolio_top_n))
        for t, partial in partial_by_ticker.items():
            prices = partial["prices"]
            risk_frame = compute_risk_frame(prices, cfg=cfg.risk_config)
            gated = apply_risk_gating(after_topn[t], risk_frame)
            gated = enforce_trade_frequency(
                gated,
                max_trades_per_day=cfg.max_trades_per_day,
                min_minutes_between_trades=cfg.min_minutes_between_trades,
            )
            signals_by_ticker[t] = gated
            trades = int(gated["Signal"].isin(["BUY", "SELL"]).sum())
            active_days = max(1, int(pd.to_datetime(gated["Date"]).dt.date.nunique())) if len(gated) else 1
            rows.append(
                {
                    "Ticker": t,
                    "Latest_Signal": str(gated.iloc[-1]["Signal"]),
                    "Latest_Proba": float(gated.iloc[-1]["Proba"]),
                    "Test_Accuracy": float(partial["technical_test_accuracy"]),
                    "Trades_Per_Day": float(trades / active_days),
                }
            )
        try:
            portfolio_backtest = backtest_portfolio_topn_long_cash(
                signals_by_ticker,
                top_n=int(cfg.portfolio_top_n),
                cfg=PortfolioBacktestConfig(initial_capital=float(cfg.initial_capital), allocation="equal"),
            )
        except ValueError:
            portfolio_backtest = backtest_portfolio_long_cash(
                signals_by_ticker,
                cfg=PortfolioBacktestConfig(initial_capital=float(cfg.initial_capital), allocation="equal"),
            )
    else:
        for t, tf_data in timeframe_data_by_ticker.items():
            try:
                out = _run_four_model_pipeline_with_data(t, cfg, tf_data)
                signals_by_ticker[t] = out["test_df"]
                rows.append(
                    {
                        "Ticker": t,
                        "Latest_Signal": out["latest"]["signal"],
                        "Latest_Proba": float(out["latest"]["proba"]),
                        "Test_Accuracy": float(out["technical_test_accuracy"]),
                        "Trades_Per_Day": float(out["trades_per_day"]),
                    }
                )
            except Exception:
                continue

        if not signals_by_ticker:
            raise RuntimeError("No ticker produced a valid pipeline output.")

        portfolio_backtest = backtest_portfolio_long_cash(
            signals_by_ticker,
            cfg=PortfolioBacktestConfig(initial_capital=float(cfg.initial_capital), allocation="equal"),
        )
    table_df = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
    return {
        "table_df": table_df,
        "portfolio_backtest": portfolio_backtest,
        "signals_by_ticker": signals_by_ticker,
    }


def run_two_stage_portfolio_pipeline(
    tickers: list[str],
    *,
    initial_capital: float = 100000.0,
    research_subset_size: int = 12,
    research_evals: int = 20,
    top_k: int = 3,
    model_type: str = "gb",
    use_meta_fusion: bool = False,
) -> dict:
    """
    Two-stage approach:
    1) Research stage (fast, 5m, subset, limited evals)
    2) Execution validation (full, 1m, full universe, stricter risk)
    """
    assert_optimizer_allowed()
    cleaned = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not cleaned:
        raise ValueError("No valid tickers provided")

    subset = cleaned[: max(1, int(research_subset_size))]
    research_cfg = FourModelConfig(
        base_interval="5m",
        context_intervals=("15m", "60m", "1d"),
        horizon_bars=5,
        model_type=model_type,
        threshold_mode="quantile",
        buy_quantile=0.70,
        sell_quantile=0.30,
        max_trades_per_day=12,
        min_minutes_between_trades=5,
        initial_capital=float(initial_capital),
        use_meta_fusion=use_meta_fusion,
        fast_mode=True,
        compute_walk_forward=False,
        risk_config=RiskConfig(hard_vol_stop=0.05, hard_drawdown_stop=0.15),
    )

    research = optimize_portfolio_four_model_pipeline(
        subset,
        base_cfg=research_cfg,
        opt_cfg=OptimizationConfig(max_evals=max(1, int(research_evals)), random_state=42),
    )

    board = research["leaderboard"].copy()
    if len(board) == 0:
        raise RuntimeError("Research stage returned no candidate configurations.")

    top = board.head(max(1, int(top_k))).reset_index(drop=True)
    validation_rows: list[dict] = []
    best_validation: dict | None = None
    best_score = -1e18

    for _, row in top.iterrows():
        strict_risk = RiskConfig(
            hard_vol_stop=min(float(row["hard_vol_stop"]), 0.045),
            hard_drawdown_stop=min(float(row["hard_drawdown_stop"]), 0.12),
        )
        exec_cfg = FourModelConfig(
            base_interval="1m",
            context_intervals=("5m", "15m", "60m", "1d"),
            horizon_bars=int(row["horizon_bars"]),
            model_type=model_type,
            threshold_mode="quantile",
            buy_quantile=float(row["buy_quantile"]),
            sell_quantile=float(row["sell_quantile"]),
            max_trades_per_day=int(row["max_trades_per_day"]),
            min_minutes_between_trades=int(row["min_minutes_between_trades"]),
            initial_capital=float(initial_capital),
            use_meta_fusion=use_meta_fusion,
            fast_mode=False,
            compute_walk_forward=True,
            risk_config=strict_risk,
        )

        try:
            out = run_portfolio_four_model_pipeline(cleaned, cfg=exec_cfg)
            bt = out["portfolio_backtest"]
            ret = float(bt.get("total_return_pct", 0.0))
            dd = abs(float(bt.get("max_drawdown_pct", 0.0)))
            sharpe = float(bt.get("sharpe", 0.0))
            if not np.isfinite(sharpe):
                sharpe = 0.0
            score = 1.2 * ret + 5.0 * sharpe - 0.45 * dd

            rec = {
                "score": float(score),
                "horizon_bars": int(exec_cfg.horizon_bars),
                "buy_quantile": float(exec_cfg.buy_quantile),
                "sell_quantile": float(exec_cfg.sell_quantile),
                "max_trades_per_day": int(exec_cfg.max_trades_per_day),
                "min_minutes_between_trades": int(exec_cfg.min_minutes_between_trades),
                "hard_vol_stop": float(exec_cfg.risk_config.hard_vol_stop),
                "hard_drawdown_stop": float(exec_cfg.risk_config.hard_drawdown_stop),
                "portfolio_return_pct": ret,
                "portfolio_max_drawdown_pct": float(bt.get("max_drawdown_pct", float("nan"))),
                "portfolio_sharpe": float(bt.get("sharpe", float("nan"))),
            }
            validation_rows.append(rec)
            if score > best_score:
                best_score = score
                best_validation = {
                    "config": exec_cfg,
                    "result": out,
                    "metrics": rec,
                }
        except Exception:
            continue

    if best_validation is None:
        raise RuntimeError("Execution validation stage failed for all top configs.")

    validation_df = pd.DataFrame(validation_rows).sort_values("score", ascending=False).reset_index(drop=True)
    return {
        "research": research,
        "research_top": top,
        "validation": validation_df,
        "best_validation": best_validation,
    }


def run_four_model_pipeline(ticker: str, cfg: FourModelConfig | None = None) -> dict:
    if cfg is None:
        cfg = FourModelConfig()
    # Apply CORE overrides before download so intervals match evaluation mode (e.g. 1d+1wk, not 1m defaults).
    cfg = _effective_four_model_config(cfg)
    intervals = tuple(dict.fromkeys((cfg.base_interval,) + tuple(cfg.context_intervals)))
    timeframe_data = download_multi_timeframe_data(ticker, intervals=intervals, auto_adjust=False)
    return _run_four_model_pipeline_with_data(ticker, cfg, timeframe_data)

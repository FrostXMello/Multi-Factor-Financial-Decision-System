from __future__ import annotations

from dataclasses import replace
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analytics_dashboard import render_portfolio_quant_dashboard, render_single_ticker_quant_dashboard
from backtest import (
    BacktestConfig,
    PortfolioBacktestConfig,
    backtest_buy_and_hold,
    backtest_long_cash,
    backtest_portfolio_buy_and_hold,
    backtest_portfolio_long_cash,
)
from core_config import CORE_MODE, TECHNICAL_MODE
from multi_layer_backtest import backtest_multi_layer_walk_forward
from multi_layer_config import MULTI_LAYER_MODE, MultiLayerPipelineConfig
from multi_layer_data import UniverseDownloadConfig, download_universe_daily
from multi_layer_pipeline import run_multi_layer_pipeline
from multi_layer_universe import get_multi_layer_full_universe
from sector_rotation_backtest import backtest_sector_rotation_walk_forward
from sector_rotation_config import SECTOR_ROTATION_MODE, SectorRotationConfig
from sector_rotation_pipeline import (
    download_sector_index_ohlcv,
    prepare_stock_prices_for_backtest,
    run_sector_rotation_snapshot,
)
from sector_universe import all_sector_rotation_tickers
from data_api import DataAPIConfig, load_candles_features_metadata
from data_loader import download_multi_timeframe_data
from fundamentals_store import (
    FundamentalsConfig,
    build_fundamentals_quality_report,
    collect_nifty50_fundamentals,
    get_fundamentals_for_ticker,
    subset_fundamentals,
)
from four_model_pipeline import (
    FourModelConfig,
    OptimizationConfig,
    optimize_four_model_pipeline,
    optimize_portfolio_four_model_pipeline,
    run_four_model_pipeline,
    run_portfolio_four_model_pipeline,
    run_two_stage_portfolio_pipeline,
)
from feature_engineering import MultiTimeframeDatasetConfig, build_multi_timeframe_dataset
from macro_model import MacroConfig, build_macro_features, infer_macro_probability
from model import ModelConfig, train_model, walk_forward_validate
from risk_model import RiskConfig, apply_risk_gating, compute_risk_frame
from strategy import attach_signals, enforce_trade_frequency
from strategy_modes import STRATEGY_MEAN_REVERSION, STRATEGY_MOMENTUM, STRATEGY_MULTI_FACTOR
from utils import NIFTY50_TICKERS, NIFTY_MIDCAP150_TICKERS, parse_tickers_text

STRATEGY_UI_TO_CANON = {
    "Multi-Factor": STRATEGY_MULTI_FACTOR,
    "Momentum": STRATEGY_MOMENTUM,
    "Mean Reversion": STRATEGY_MEAN_REVERSION,
}


st.set_page_config(page_title="Multi-Factor AI Financial Decision System", layout="wide")
st.title("Multi-Factor AI Financial Decision System")
if CORE_MODE:
    if TECHNICAL_MODE:
        st.caption(
            "TECHNICAL_MODE (`core_config.TECHNICAL_MODE`): daily (+ optional weekly) **price-only** features, one gradient boosting model, "
            "meaningful-move label, MA50 + probability signals, top-N portfolio concentration, simplified risk — **no macro/micro/fusion** at runtime. "
            "See `core_config.py`."
        )
    else:
        st.caption(
            "CORE evaluation mode: daily data, selectable **Multi-Factor / Momentum / Mean Reversion** strategies, gradient boosting, "
            "**technical signals only** (no macro blend in CORE; no micro, no meta-fusion, no optimizer). See `core_config.py` and README."
        )
else:
    st.caption("Run multi-timeframe analysis (down to 1-minute bars) and backtest risk-aware BUY/SELL/HOLD execution.")

if not CORE_MODE:
    with st.expander("Methodology Caveats (Important)", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- The stack is intentionally rich (Macro + Micro + Fusion + Optimizer + Intraday), but complexity can create over-engineering risk without strict out-of-sample validation.",
                    "- We use proxy data sources due to academic constraints: Yahoo intraday coverage can be limited/noisy, and fundamentals metadata is not strictly point-in-time audited.",
                    "- Meta-fusion can be optimistic when using in-sample stacking; future improvement should include out-of-fold stacking to reduce bias.",
                    "- Optimizer overfitting risk is controlled with constrained search spaces and Fast mode, but search results should still be validated in holdout periods.",
                ]
            )
        )


POPULAR_TICKER_OPTIONS = sorted(
    set(
        NIFTY50_TICKERS
        + NIFTY_MIDCAP150_TICKERS
        + [
            "^NSEI",
            "^NSEBANK",
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
            "META",
            "TSLA",
            "SPY",
            "QQQ",
            "BTC-USD",
            "ETH-USD",
            "GC=F",
            "CL=F",
        ]
    )
)

# Full pool for search + multiselect (NIFTY 50 + Midcap 150 + popular globals / indices).
FULL_TICKER_UNIVERSE: list[str] = sorted(set(POPULAR_TICKER_OPTIONS))


def _filter_ticker_pool(pool: list[str], search: str) -> list[str]:
    q = (search or "").strip().upper()
    if not q:
        return list(pool)
    return [t for t in pool if q in t.upper()]


def _render_single_ticker_chooser(*, default_ticker: str = "RELIANCE.NS") -> str:
    st.markdown("**Ticker**")
    q = st.text_input(
        "Search symbols",
        value="",
        key="single_ticker_search_q",
        placeholder="Type to filter (e.g. RELIANCE, BANK, AAPL)",
        help="Narrows the dropdown list below. The multiselect also has its own search.",
    )
    filt = _filter_ticker_pool(FULL_TICKER_UNIVERSE, q)
    if not filt:
        st.warning("No symbols match your search — clear it or broaden the filter.")
        filt = [default_ticker] if default_ticker in FULL_TICKER_UNIVERSE else [FULL_TICKER_UNIVERSE[0]]

    if "single_ticker_multiselect" not in st.session_state:
        st.session_state["single_ticker_multiselect"] = [default_ticker] if default_ticker in filt else [filt[0]]

    picked = st.multiselect(
        "Select ticker (pick one in the dropdown)",
        options=filt,
        max_selections=1,
        key="single_ticker_multiselect",
        help="Choose one symbol. You can type inside the control to jump quickly.",
    )
    ticker = (picked[0] if picked else default_ticker).strip().upper()

    with st.expander("Or type ticker manually", expanded=False):
        manual = st.text_input(
            "Exact Yahoo Finance symbol (optional override)",
            value="",
            key="single_ticker_manual",
            placeholder=ticker,
            help="If you fill this, it overrides the dropdown for this run.",
        )
    if manual and str(manual).strip():
        ticker = str(manual).strip().upper()

    return ticker


def _render_portfolio_ticker_chooser(*, default_selection: list[str]) -> str:
    st.markdown("**Tickers**")
    q = st.text_input(
        "Search symbols",
        value="",
        key="portfolio_ticker_search_q",
        placeholder="Filter list (e.g. TATA, BANK, .NS)",
        help="Only symbols containing this text (case-insensitive) stay in the list and dropdown.",
    )
    filt = _filter_ticker_pool(FULL_TICKER_UNIVERSE, q)
    if not filt:
        st.warning("No symbols match — clear search or widen the filter.")
        filt = [t for t in default_selection if t in FULL_TICKER_UNIVERSE] or NIFTY50_TICKERS[:5]

    select_all = st.checkbox(
        "Choose all matching tickers",
        value=False,
        key="portfolio_select_all_filtered",
        help="Selects every symbol in the filtered list above (respects Max tickers when you run).",
    )
    if select_all:
        st.session_state["portfolio_ticker_multiselect"] = list(filt)

    if "portfolio_ticker_multiselect" not in st.session_state:
        seed = [t for t in default_selection if t in filt]
        st.session_state["portfolio_ticker_multiselect"] = seed or filt[: min(10, len(filt))]

    st.multiselect(
        "Select tickers",
        options=filt,
        key="portfolio_ticker_multiselect",
        help="Multi-select dropdown with built-in search. Use “Choose all matching” to take the full filtered set.",
    )
    selected = list(st.session_state.get("portfolio_ticker_multiselect", []))
    selected = [s.strip().upper() for s in selected if s and str(s).strip()]

    with st.expander("Or paste comma-separated symbols", expanded=False):
        paste = st.text_area("Extra tickers (merged with selection)", value="", key="portfolio_ticker_paste", height=68)
        if paste.strip():
            selected = parse_tickers_text(",".join(selected) + "," + paste)

    return ",".join(dict.fromkeys(selected))


@st.cache_data(ttl=10 * 60)
def load_multi_timeframe_cached(ticker: str, base_interval: str, context_intervals: tuple[str, ...]):
    intervals = tuple(dict.fromkeys((base_interval,) + context_intervals + ("1d",)))
    return download_multi_timeframe_data(ticker, intervals=intervals, auto_adjust=False)


@st.cache_data(ttl=10 * 60)
def load_clean_data_api_cached(ticker: str, base_interval: str, context_intervals: tuple[str, ...], horizon_bars: int):
    return load_candles_features_metadata(
        ticker,
        cfg=DataAPIConfig(
            base_interval=base_interval,
            context_intervals=context_intervals,
            horizon_bars=horizon_bars,
            auto_adjust=False,
        ),
    )


@st.cache_data(ttl=24 * 60 * 60)
def load_nifty50_fundamentals_cached(force_refresh: bool = False):
    return collect_nifty50_fundamentals(FundamentalsConfig(), force_refresh=force_refresh)


def _build_data_health_table(timeframe_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
        date_ser = pd.to_datetime(df["Date"]) if "Date" in df.columns else pd.Series(dtype="datetime64[ns]")
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


def _run_single_benchmark_suite(
    ticker: str,
    *,
    initial_capital: float,
    max_runs: int,
    use_optimizer: bool,
    optimizer_evals: int,
    use_meta_fusion: bool,
    context_intervals: tuple[str, ...],
    fast_mode: bool,
) -> pd.DataFrame:
    model_opts = ["gb", "rf", "logreg"]
    interval_opts = ["1m", "5m", "15m", "60m", "1d"]
    horizon_opts = [3, 5, 10]
    grid = list(product(model_opts, interval_opts, horizon_opts))

    if len(grid) > max_runs:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(grid), size=max_runs, replace=False)
        grid = [grid[int(i)] for i in idx]

    rows: list[dict] = []
    for model_type, interval, horizon in grid:
        base_cfg = FourModelConfig(
            base_interval=interval,
            context_intervals=context_intervals,
            horizon_bars=horizon,
            model_type=model_type,
            threshold_mode="quantile",
            buy_quantile=0.7,
            sell_quantile=0.3,
            max_trades_per_day=12,
            min_minutes_between_trades=5,
            initial_capital=initial_capital,
            use_meta_fusion=use_meta_fusion,
            fast_mode=fast_mode,
            compute_walk_forward=(not fast_mode),
        )
        try:
            if use_optimizer:
                opt = optimize_four_model_pipeline(
                    ticker,
                    base_cfg=base_cfg,
                    opt_cfg=OptimizationConfig(max_evals=optimizer_evals, random_state=42),
                )
                out = opt["best_result"]
                opt_score = float(opt.get("best_score", float("nan")))
                regime = str(opt.get("regime", "unknown"))
            else:
                out = run_four_model_pipeline(ticker, cfg=base_cfg)
                opt_score = float("nan")
                regime = "na"

            bt = out["backtest"]
            rows.append(
                {
                    "Model": model_type,
                    "BaseInterval": interval,
                    "HorizonBars": horizon,
                    "Optimized": bool(use_optimizer),
                    "Regime": regime,
                    "ReturnPct": float(bt.get("total_return_pct", float("nan"))),
                    "Sharpe": float(bt.get("sharpe", float("nan"))),
                    "MaxDrawdownPct": float(bt.get("max_drawdown_pct", float("nan"))),
                    "Accuracy": float(out.get("technical_test_accuracy", float("nan"))),
                    "TradesPerDay": float(out.get("trades_per_day", float("nan"))),
                    "FinalValue": float(bt.get("final_portfolio_value", float("nan"))),
                    "OptimizerScore": opt_score,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "Model": model_type,
                    "BaseInterval": interval,
                    "HorizonBars": horizon,
                    "Optimized": bool(use_optimizer),
                    "Regime": "error",
                    "ReturnPct": float("nan"),
                    "Sharpe": float("nan"),
                    "MaxDrawdownPct": float("nan"),
                    "Accuracy": float("nan"),
                    "TradesPerDay": float("nan"),
                    "FinalValue": float("nan"),
                    "OptimizerScore": float("nan"),
                    "Error": str(e),
                }
            )

    board = pd.DataFrame(rows)
    if len(board) > 0 and "ReturnPct" in board.columns:
        board = board.sort_values("ReturnPct", ascending=False, na_position="last").reset_index(drop=True)
    return board


def _run_multi_layer_ui(
    *,
    max_names: int,
    pipe_cfg: MultiLayerPipelineConfig,
    period: str,
    backtest_years: float | None = 1.0,
) -> dict:
    """
    Download diversified universe OHLCV, run Layer1→2→3 snapshot, and walk-forward backtest
    (month-end rebalance; window controlled by ``backtest_years``).
    """
    full = get_multi_layer_full_universe(max_names=max_names)
    bar = st.progress(0.0, text="Downloading universe (Yahoo Finance)…")

    def _cb(done: int, total: int, sym: str) -> None:
        if total > 0:
            bar.progress(min(1.0, float(done) / float(total)), text=f"{sym} ({done}/{total})")

    prices = download_universe_daily(
        full,
        UniverseDownloadConfig(period=period),
        progress_callback=_cb,
    )
    bar.empty()
    if len(prices) < 15:
        raise RuntimeError(
            f"Too few symbols after liquidity/history pre-filter ({len(prices)}). "
            "Lower the universe cap or check connectivity."
        )
    snap = run_multi_layer_pipeline(prices, pipe_cfg)
    with st.spinner("Walk-forward backtest (month-end rebalance; Layer 1→3 each rebalance; may take several minutes)…"):
        bt = backtest_multi_layer_walk_forward(prices, pipe_cfg, backtest_years=backtest_years)
    return {
        "universe_requested": len(full),
        "universe_loaded": len(prices),
        "snapshot": snap,
        "backtest": bt,
        "prices": prices,
    }


def _run_sector_rotation_ui(
    *,
    period: str,
    sr_cfg: SectorRotationConfig,
) -> dict:
    """Download sector indices + stock buckets, snapshot, weekly walk-forward backtest."""
    bar = st.progress(0.0, text="Downloading sector indices…")
    ix = download_sector_index_ohlcv(sr_cfg, period=period)
    bar.progress(0.2, text="Downloading sector stock universe…")
    tickers = all_sector_rotation_tickers()

    def _st_cb(done: int, total: int, sym: str) -> None:
        if total > 0:
            bar.progress(0.2 + 0.75 * min(1.0, float(done) / float(total)), text=f"Stock {sym} ({done}/{total})")

    prices = prepare_stock_prices_for_backtest(tickers, period=period, progress_callback=_st_cb)
    bar.progress(1.0, text="Done downloading.")
    bar.empty()
    if len(ix) < 4:
        raise RuntimeError(f"Too few sector indices downloaded ({len(ix)}). Check Yahoo / connectivity.")
    if len(prices) < 15:
        raise RuntimeError(f"Too few stocks after download ({len(prices)}). Check connectivity.")

    snap = run_sector_rotation_snapshot(ix, prices, sr_cfg, asof=None)
    with st.spinner("Weekly walk-forward backtest (sector rank → ML pick per sector; several minutes)…"):
        bt = backtest_sector_rotation_walk_forward(ix, prices, sr_cfg)
    return {
        "indices_loaded": len(ix),
        "stocks_loaded": len(prices),
        "snapshot": snap,
        "backtest": bt,
        "index_frames": ix,
        "prices": prices,
    }


def _simple_fusion_proba(tech_proba: np.ndarray, signal_dates: pd.Series, *, tech_weight: float = 0.8) -> np.ndarray:
    macro_features = build_macro_features(MacroConfig(horizon_days=1))
    macro_proba = infer_macro_probability(macro_features, signal_dates)
    alpha = float(tech_weight) * np.asarray(tech_proba, dtype=float) + (1.0 - float(tech_weight)) * macro_proba.values
    alpha = np.where(np.isfinite(alpha), alpha, 0.5)
    return np.clip(alpha, 0.0, 1.0)


col_left, col_right = st.columns([1, 1])
with col_left:
    ml_pipe_cfg = MultiLayerPipelineConfig()
    ml_max_names = 120
    ml_bt_window = "Last 1 year (default)"
    ml_period = "5y"
    sr_top_sectors = 3
    sr_period = "5y"
    sr_model_label = "Gradient Boosting"
    sr_bt_window = "Last 1 year (default)"
    sr_max_vol = 0.045

    if CORE_MODE:
        st.success(
            "CORE_MODE is **on** (`core_config.CORE_MODE`). Advanced controls are hidden; pipeline uses fixed academic defaults."
        )
        _core_u = ["Single Ticker", "NIFTY 50 Portfolio", "NIFTY Midcap 150 Portfolio"]
        if MULTI_LAYER_MODE:
            _core_u.append("Multi-Layer (3-Stage)")
        if SECTOR_ROTATION_MODE:
            _core_u.append("Sector Rotation (Flow)")
        universe_mode = st.selectbox("Universe", options=_core_u, index=0)
        if TECHNICAL_MODE:
            strategy_choice = "Technical-Only"
            st.caption(
                "Strategy variants are disabled while TECHNICAL_MODE is on — the pipeline is fixed to the simplified technical path in `core_config.py`."
            )
        else:
            strategy_choice = st.selectbox(
                "Select Strategy",
                ["Multi-Factor", "Momentum", "Mean Reversion"],
                index=0,
                help="Multi-Factor: fusion stack. Momentum: trend + classifier gates. Mean Reversion: Z-score rules (ML proba for display).",
            )
        core_include_macro = False
        if not TECHNICAL_MODE:
            st.caption("CORE uses **technical probability only** (macro is disabled for this mode).")
        if universe_mode == "Single Ticker":
            ticker = _render_single_ticker_chooser(default_ticker="RELIANCE.NS")
        elif universe_mode == "Multi-Layer (3-Stage)":
            st.caption(
                "Large/mid/small NSE universe (see `multi_layer_universe.py`). Pre-filter drops illiquid / short-history names."
            )
        elif universe_mode == "Sector Rotation (Flow)":
            st.caption(
                "Sector indices (Bank, IT, FMCG, Auto, Pharma, Metal, Energy) → risk-adjusted momentum rank → "
                "ML probability + MA50: **one stock per top sector**. Weekly walk-forward backtest vs Nifty (spot)."
            )
        elif universe_mode == "NIFTY Midcap 150 Portfolio":
            st.caption(
                "NIFTY Midcap 150 list: `utils.NIFTY_MIDCAP150_TICKERS` (NSE composition; update after index rebalancing)."
            )
            tickers_text = _render_portfolio_ticker_chooser(default_selection=NIFTY_MIDCAP150_TICKERS[:15])
            max_tickers = st.slider("Max tickers", 1, 150, 15, step=1)
        else:
            st.caption("Smaller NIFTY subset recommended for speed in CORE mode. Use search + multiselect or “Choose all matching”.")
            tickers_text = _render_portfolio_ticker_chooser(default_selection=NIFTY50_TICKERS[:10])
            max_tickers = st.slider("Max tickers", 1, 50, 10, step=1)
        initial_capital_total = st.number_input("Initial capital", value=100000.0, min_value=1000.0, step=1000.0)
        run = st.button("Run CORE pipeline", type="primary")
        pipeline_mode = "CORE"
        advanced_mode = False
        stack_mode = "Four-Model Stack"
        ticker_input_mode = "Manual Input"
        refresh_fundamentals = False
        model_choice = "Gradient Boosting"
        run_profile = "Fast"
        base_interval = "1d"
        horizon_bars = 1
        max_trades_per_day = 9999
        min_minutes_between_trades = 0
        use_meta_fusion = False
        auto_optimize = False
        optimizer_evals = 10
        run_two_stage = False
        research_subset_size = 12
        research_evals = 20
        top_k_configs = 3
        run_benchmark_suite = False
        benchmark_max_runs = 4
        benchmark_use_optimizer = False
        simple_use_fusion = False
        simple_tech_weight = 1.0
        threshold_mode = "Fixed"
        buy_threshold = 0.6
        sell_threshold = 0.4
        buy_quantile = 0.7
        sell_quantile = 0.3
        context_intervals = ("1wk",)
        fast_mode = True
        effective_optimizer_evals = 10
    else:
        _adv_u = ["Single Ticker", "NIFTY 50 Portfolio", "NIFTY Midcap 150 Portfolio"]
        if MULTI_LAYER_MODE:
            _adv_u.append("Multi-Layer (3-Stage)")
        if SECTOR_ROTATION_MODE:
            _adv_u.append("Sector Rotation (Flow)")
        universe_mode = st.selectbox("Universe Mode", options=_adv_u, index=0)
        core_include_macro = True
        pipeline_mode = st.selectbox("Pipeline Mode", options=["Simple Production", "Advanced Research"], index=0)
        advanced_mode = pipeline_mode == "Advanced Research"
        if advanced_mode:
            stack_mode = st.selectbox("Model Stack", options=["Technical Only", "Four-Model Stack"], index=1)
        else:
            stack_mode = "Technical Only"
            st.info(
                "Simple pipeline: DATA -> FEATURES (multi-timeframe simple) -> TECHNICAL MODEL -> "
                "SIMPLE FUSION (optional) -> SIGNAL -> RISK FILTER -> BACKTEST"
            )
            simple_use_fusion = st.checkbox("Enable simple fusion (Technical + Macro)", value=False)
            simple_tech_weight = (
                st.slider("Technical weight in simple fusion", 0.50, 1.00, 0.80, step=0.01)
                if simple_use_fusion
                else 1.0
            )

        if advanced_mode and stack_mode == "Four-Model Stack":
            strategy_choice = st.selectbox(
                "Select Strategy",
                ["Multi-Factor", "Momentum", "Mean Reversion"],
                index=0,
                help="Used for Four-Model stack runs: features, signal logic, and fusion path.",
            )
        else:
            strategy_choice = "Multi-Factor"

        if universe_mode == "Single Ticker":
            ticker_input_mode = "Search + dropdown"
            ticker = _render_single_ticker_chooser(default_ticker="RELIANCE.NS")
        elif universe_mode == "Multi-Layer (3-Stage)":
            ticker_input_mode = "Multi-layer"
            st.caption("3-stage scan: fast filters → per-name ML → diversified Top-N.")
        elif universe_mode == "Sector Rotation (Flow)":
            ticker_input_mode = "Sector rotation"
            st.caption("Capital-flow path: `sector_universe.py` defines indices and per-sector stock buckets.")
        elif universe_mode == "NIFTY Midcap 150 Portfolio":
            ticker_input_mode = "NIFTY Midcap 150"
            st.write(
                "Pick symbols from the Midcap 150 list (`utils.NIFTY_MIDCAP150_TICKERS`) plus search. Optional paste for extras."
            )
            tickers_text = _render_portfolio_ticker_chooser(default_selection=list(NIFTY_MIDCAP150_TICKERS))
            max_tickers = st.slider("Max tickers to run (for speed)", 1, 150, 50, step=1)
        else:
            st.write("Pick symbols from the searchable list (NIFTY 50 + popular). Optional paste for extras.")
            tickers_text = _render_portfolio_ticker_chooser(default_selection=list(NIFTY50_TICKERS))
            max_tickers = st.slider("Max tickers to run (for speed)", 1, 50, 50, step=1)

        model_choice = st.selectbox("Technical Model", options=["Gradient Boosting", "Logistic Regression", "Random Forest"], index=0)
        run_profile = st.selectbox("Run Profile", options=["Fast", "Full"], index=0)
        base_interval = st.selectbox("Execution interval", options=["1m", "5m", "15m", "60m", "1d"], index=0)
        horizon_bars = st.slider("Prediction horizon (bars)", 1, 60, 5)
        max_trades_per_day = st.slider("Max trades per day", 1, 50, 12)
        min_minutes_between_trades = st.slider("Min minutes between trades", 0, 120, 5)
        use_meta_fusion = st.checkbox("Use trainable meta-fusion (instead of weighted fusion)", value=False)
        refresh_fundamentals = st.checkbox("Refresh NIFTY50 fundamentals cache", value=False)

        if advanced_mode:
            auto_optimize = st.checkbox("Auto-optimize four-model params (regime-aware)", value=False)
            optimizer_evals = st.slider("Optimizer evaluations", 10, 120, 40, step=5)
            run_two_stage = st.checkbox("Run 2-stage portfolio workflow (research -> execution)", value=False)
            research_subset_size = st.slider("Research subset size", 5, 25, 12, step=1)
            research_evals = st.slider("Research stage evaluations", 5, 60, 20, step=5)
            top_k_configs = st.slider("Top research configs to validate", 1, 5, 3, step=1)
            run_benchmark_suite = st.checkbox("Run comprehensive benchmark suite", value=False)
            benchmark_max_runs = st.slider("Benchmark runs", 4, 40, 12, step=2)
            benchmark_use_optimizer = st.checkbox("Optimize each benchmark run", value=False)
        else:
            auto_optimize = False
            optimizer_evals = 10
            run_two_stage = False
            research_subset_size = 12
            research_evals = 20
            top_k_configs = 3
            run_benchmark_suite = False
            benchmark_max_runs = 4
            benchmark_use_optimizer = False

        st.subheader("Thresholding (convert probability -> BUY/SELL/HOLD)")
        threshold_mode = st.selectbox("Threshold Mode", options=["Quantile", "Fixed"], index=0)

        if threshold_mode == "Fixed":
            buy_threshold = st.slider("BUY if Proba >", 0.5, 0.95, 0.6, step=0.01)
            sell_threshold = st.slider("SELL if Proba <", 0.05, 0.5, 0.4, step=0.01)
            buy_quantile = 0.7
            sell_quantile = 0.3
        else:
            buy_threshold = 0.6
            sell_threshold = 0.4
            buy_quantile = st.slider("BUY top quantile (>=)", 0.5, 0.95, 0.7, step=0.01)
            sell_quantile = st.slider("SELL bottom quantile (<=)", 0.05, 0.5, 0.3, step=0.01)

        initial_capital_total = st.number_input("Initial capital (portfolio or single)", value=100000.0, min_value=1000.0, step=1000.0)

        run = st.button("Run Model", type="primary")

    if MULTI_LAYER_MODE and universe_mode == "Multi-Layer (3-Stage)":
        st.divider()
        st.markdown("##### Technical Multi-Layer")
        st.caption("Layer 1: momentum + trend + vol + liquidity  →  Layer 2: GB classifier  →  Layer 3: proba + MA50 + sector + corr")
        ml_max_names = st.slider("Max symbols to request from universe list", 50, 300, 120, 10)
        ml_l1 = st.slider("Layer 1 output cap (after rank)", 30, 80, 55, 5)
        ml_l2k = st.slider("Layer 2 ML shortlist size", 15, 40, 25, 1)
        ml_final_n = st.slider("Layer 3 portfolio size (Top-N)", 3, 5, 4, 1)
        ml_min_p = st.slider("Layer 3 min probability", 0.50, 0.85, 0.60, 0.01)
        ml_risk_adj = st.checkbox("Risk-adjusted rank (score = P / ann_vol)", value=False)
        ml_corr = st.checkbox("Correlation prune in Layer 3", value=True)
        ml_period = st.selectbox("Download history", ["3y", "5y", "max"], index=1)
        ml_bt_window = st.selectbox(
            "Backtest window (walk-forward)",
            ["Last 1 year (default)", "Last 3 years", "Full downloaded history"],
            index=0,
            help="Uses month-end rebalance dates in range. Full history still respects max rebalance points in config.",
        )
        ml_pipe_cfg = MultiLayerPipelineConfig(
            layer1_out_max=int(ml_l1),
            layer1_out_min=40,
            layer2_top_k=int(ml_l2k),
            layer3_n=int(ml_final_n),
            min_proba=float(ml_min_p),
            use_risk_adjusted=bool(ml_risk_adj),
            max_pairwise_corr=0.88 if ml_corr else None,
        )

    if SECTOR_ROTATION_MODE and universe_mode == "Sector Rotation (Flow)":
        st.divider()
        st.markdown("##### Sector rotation (capital flow)")
        sr_top_sectors = st.slider("Top sectors to trade", 2, 3, 3, 1)
        sr_model_label = st.selectbox("Classifier", ["Gradient Boosting", "Random Forest"], index=0)
        sr_period = st.selectbox("Download history", ["3y", "5y", "max"], index=1)
        sr_bt_window = st.selectbox(
            "Backtest window (weekly rebalance)",
            ["Last 1 year (default)", "Last 3 years", "Full downloaded history"],
            index=0,
        )
        sr_max_vol = st.slider("Max 10d rolling vol (exclude names above)", 0.02, 0.08, 0.045, 0.005)

if CORE_MODE:
    pass  # fast_mode / context already set in CORE block
else:
    fast_mode = run_profile == "Fast"
    context_intervals = ("15m", "60m", "1d") if fast_mode else ("5m", "15m", "60m", "1d")
    effective_optimizer_evals = min(int(optimizer_evals), 20) if fast_mode else int(optimizer_evals)
    if advanced_mode:
        simple_use_fusion = False
        simple_tech_weight = 1.0

if TECHNICAL_MODE:
    pipeline_strategy_mode = "technical_only"
    four_model_strategy_mode = STRATEGY_MULTI_FACTOR
else:
    pipeline_strategy_mode = STRATEGY_UI_TO_CANON[strategy_choice]
    four_model_strategy_mode = pipeline_strategy_mode

with col_right:
    st.markdown("### Output")
    if universe_mode == "Multi-Layer (3-Stage)":
        st.caption("Active: **Technical Multi-Layer** (universe → L1 → L2 → L3)")
    elif universe_mode == "Sector Rotation (Flow)":
        st.caption("Active: **Sector rotation** (index momentum / vol → ML + MA50 per sector)")
    else:
        st.caption(f"Active strategy: **{strategy_choice}** (`{pipeline_strategy_mode}`)")
    latest_signal_placeholder = st.empty()
    latest_proba_placeholder = st.empty()
    accuracy_placeholder = st.empty()
    backtest_placeholder = st.empty()
    signal_counts_placeholder = st.empty()
    proba_hist_placeholder = st.empty()
    portfolio_table_placeholder = st.empty()
    optimizer_table_placeholder = st.empty()


def _run_single_pipeline(single_ticker: str):
    if not single_ticker or not single_ticker.strip():
        raise ValueError("Please enter a valid ticker.")

    if CORE_MODE:
        out = run_four_model_pipeline(
            single_ticker.strip().upper(),
            cfg=FourModelConfig(
                core_include_macro=core_include_macro,
                initial_capital=float(initial_capital_total),
                strategy_mode=four_model_strategy_mode,
            ),
        )
        counts = out["test_df"]["Signal"].value_counts().to_dict()
        return (
            out["latest"],
            float(out["technical_test_accuracy"]),
            out["backtest"],
            counts,
            out["test_df"]["Proba"].values,
            out,
        )

    model_type = "gb" if model_choice == "Gradient Boosting" else ("logreg" if model_choice == "Logistic Regression" else "rf")

    if stack_mode == "Four-Model Stack":
        base_cfg = FourModelConfig(
            base_interval=base_interval,
            context_intervals=context_intervals,
            horizon_bars=horizon_bars,
            model_type=model_type,
            threshold_mode="fixed" if threshold_mode == "Fixed" else "quantile",
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            buy_quantile=buy_quantile,
            sell_quantile=sell_quantile,
            max_trades_per_day=max_trades_per_day,
            min_minutes_between_trades=min_minutes_between_trades,
            initial_capital=float(initial_capital_total),
            use_meta_fusion=use_meta_fusion,
            fast_mode=fast_mode,
            compute_walk_forward=(not fast_mode),
            strategy_mode=four_model_strategy_mode,
        )

        if auto_optimize:
            opt = optimize_four_model_pipeline(
                single_ticker.strip().upper(),
                base_cfg=base_cfg,
                opt_cfg=OptimizationConfig(max_evals=effective_optimizer_evals, random_state=42),
            )
            out = opt["best_result"]
            counts = out["test_df"]["Signal"].value_counts().to_dict()
            extra = {
                "optimization": {
                    "enabled": True,
                    "regime": opt["regime"],
                    "score": float(opt["best_score"]),
                    "evaluated": int(opt["evaluated"]),
                    "best_config": opt["best_config"],
                    "leaderboard": opt["leaderboard"],
                }
            }
            return out["latest"], out["technical_test_accuracy"], out["backtest"], counts, out["test_df"]["Proba"].values, {
                **out,
                **extra,
            }

        out = run_four_model_pipeline(
            single_ticker.strip().upper(),
            cfg=base_cfg,
        )
        counts = out["test_df"]["Signal"].value_counts().to_dict()
        return out["latest"], out["technical_test_accuracy"], out["backtest"], counts, out["test_df"]["Proba"].values, out

    bundle = load_clean_data_api_cached(single_ticker.strip().upper(), base_interval, context_intervals, horizon_bars)
    timeframe_data = bundle["timeframe_data"]
    prices = bundle["base_prices"]
    model_df = bundle["model_df"]
    technical_cols = bundle["technical_feature_cols"]
    result = train_model(model_df, cfg=ModelConfig(model_type=model_type), feature_cols=technical_cols)
    if fast_mode:
        wf_mean = float("nan")
    else:
        wf = walk_forward_validate(model_df, cfg=ModelConfig(model_type=model_type), feature_cols=technical_cols, n_folds=5)
        wf_mean = float(wf.mean_score)
    proba_for_signal = result.test_proba
    if not advanced_mode and simple_use_fusion:
        proba_for_signal = _simple_fusion_proba(result.test_proba, result.test_df["Date"], tech_weight=simple_tech_weight)

    signals_df = attach_signals(
        result.test_df,
        proba_for_signal,
        threshold_mode="fixed" if threshold_mode == "Fixed" else "quantile",
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        buy_quantile=buy_quantile,
        sell_quantile=sell_quantile,
    )
    risk_frame = compute_risk_frame(prices, cfg=RiskConfig())
    signals_df = apply_risk_gating(signals_df, risk_frame)
    signals_df = enforce_trade_frequency(
        signals_df,
        max_trades_per_day=max_trades_per_day,
        min_minutes_between_trades=min_minutes_between_trades,
    )
    backtest = backtest_long_cash(signals_df, cfg=BacktestConfig(initial_capital=float(initial_capital_total)))
    latest_row = signals_df.iloc[-1]
    latest = {
        "signal": str(latest_row["Signal"]),
        "proba": float(latest_row["Proba"]),
        "risk_score": float(latest_row["RiskScore"]),
        "position_size": float(latest_row["PositionSize"]),
        "trade_date": latest_row["Date"],
    }
    counts = signals_df["Signal"].value_counts()
    extra = {
        "walk_forward_mean_accuracy": wf_mean,
        "test_df": signals_df,
        "prices": prices,
        "timeframe_data": timeframe_data,
        "price_quality_report": bundle.get("price_quality_report"),
        "simple_fusion_enabled": bool((not advanced_mode) and simple_use_fusion),
        "simple_fusion_tech_weight": float(simple_tech_weight),
    }

    funda = load_nifty50_fundamentals_cached(force_refresh=refresh_fundamentals)
    extra["fundamentals_ticker"] = get_fundamentals_for_ticker(funda, single_ticker)
    extra["fundamentals_quality"] = build_fundamentals_quality_report(funda)
    return latest, float(result.test_accuracy), backtest, counts.to_dict(), proba_for_signal, extra


def _run_nifty50_pipeline():
    tickers = parse_tickers_text(tickers_text)
    tickers = tickers[:max_tickers]
    if not tickers:
        raise ValueError("No tickers provided.")

    # Technical-only basket: top-N concentration + single capital pool (see `run_portfolio_four_model_pipeline`).
    if TECHNICAL_MODE:
        funda_subset = pd.DataFrame()
        funda_quality = pd.DataFrame()
        res = run_portfolio_four_model_pipeline(
            tickers,
            cfg=FourModelConfig(initial_capital=float(initial_capital_total)),
        )
        return (
            res["table_df"],
            res["portfolio_backtest"],
            {
                "optimization": {"enabled": False},
                "fundamentals_subset": funda_subset,
                "fundamentals_quality": funda_quality,
                "signals_by_ticker": res["signals_by_ticker"],
                "strategy_mode": pipeline_strategy_mode,
            },
        )

    model_type = "gb" if model_choice == "Gradient Boosting" else ("logreg" if model_choice == "Logistic Regression" else "rf")

    if CORE_MODE:
        funda_subset = pd.DataFrame()
        funda_quality = pd.DataFrame()
    else:
        funda = load_nifty50_fundamentals_cached(force_refresh=refresh_fundamentals)
        funda_subset = subset_fundamentals(funda, tickers)
        funda_quality = build_fundamentals_quality_report(funda_subset)

    if stack_mode == "Four-Model Stack" and run_two_stage:
        two_stage = run_two_stage_portfolio_pipeline(
            tickers,
            initial_capital=float(initial_capital_total),
            research_subset_size=int(research_subset_size),
            research_evals=int(research_evals if fast_mode else min(research_evals, effective_optimizer_evals)),
            top_k=int(top_k_configs),
            model_type=model_type,
            use_meta_fusion=use_meta_fusion,
        )
        best = two_stage["best_validation"]
        return (
            best["result"]["table_df"],
            best["result"]["portfolio_backtest"],
            {
                "optimization": {
                    "enabled": True,
                    "mode": "two-stage",
                    "regime": two_stage["research"].get("regime"),
                    "score": float(two_stage["research"].get("best_score", float("nan"))),
                    "evaluated": int(two_stage["research"].get("evaluated", 0)),
                    "best_config": best["config"],
                    "leaderboard": two_stage["validation"],
                    "research_top": two_stage["research_top"],
                },
                "fundamentals_subset": funda_subset,
                "fundamentals_quality": funda_quality,
            },
        )

    if stack_mode == "Four-Model Stack" and auto_optimize:
        opt = optimize_portfolio_four_model_pipeline(
            tickers,
            base_cfg=FourModelConfig(
                base_interval=base_interval,
                context_intervals=context_intervals,
                horizon_bars=horizon_bars,
                model_type=model_type,
                threshold_mode="fixed" if threshold_mode == "Fixed" else "quantile",
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                buy_quantile=buy_quantile,
                sell_quantile=sell_quantile,
                max_trades_per_day=max_trades_per_day,
                min_minutes_between_trades=min_minutes_between_trades,
                initial_capital=float(initial_capital_total),
                use_meta_fusion=use_meta_fusion,
                fast_mode=fast_mode,
                compute_walk_forward=(not fast_mode),
                strategy_mode=four_model_strategy_mode,
            ),
            opt_cfg=OptimizationConfig(max_evals=effective_optimizer_evals, random_state=42),
        )
        return opt["best_result"]["table_df"], opt["best_result"]["portfolio_backtest"], {
            "optimization": {
                "enabled": True,
                "regime": opt["regime"],
                "score": float(opt["best_score"]),
                "evaluated": int(opt["evaluated"]),
                "best_config": opt["best_config"],
                "leaderboard": opt["leaderboard"],
            },
            "fundamentals_subset": funda_subset,
            "fundamentals_quality": funda_quality,
        }

    signals_by_ticker: dict[str, pd.DataFrame] = {}
    latest_rows: list[dict] = []
    price_quality_rows: list[dict] = []

    # Note: this is intentionally sequential for clearer error reporting.
    for t in tickers:
        try:
            timeframe_data = load_multi_timeframe_cached(t, base_interval, context_intervals)
            prices = timeframe_data[base_interval]
            price_quality_rows.append(
                {
                    "Ticker": t,
                    "Rows": int(len(prices)),
                    "Start": pd.to_datetime(prices["Date"]).min() if len(prices) else None,
                    "End": pd.to_datetime(prices["Date"]).max() if len(prices) else None,
                    "MissingPct": float(prices.isna().mean().mean() * 100.0) if len(prices) else 100.0,
                }
            )
            if stack_mode == "Four-Model Stack":
                out = run_four_model_pipeline(
                    t,
                    cfg=FourModelConfig(
                        core_include_macro=core_include_macro,
                        base_interval=base_interval,
                        context_intervals=context_intervals,
                        horizon_bars=horizon_bars,
                        model_type=model_type,
                        threshold_mode="fixed" if threshold_mode == "Fixed" else "quantile",
                        buy_threshold=buy_threshold,
                        sell_threshold=sell_threshold,
                        buy_quantile=buy_quantile,
                        sell_quantile=sell_quantile,
                        max_trades_per_day=max_trades_per_day,
                        min_minutes_between_trades=min_minutes_between_trades,
                        initial_capital=float(initial_capital_total) / max(1, len(tickers)),
                        use_meta_fusion=use_meta_fusion,
                        fast_mode=fast_mode,
                        compute_walk_forward=(not fast_mode),
                        strategy_mode=four_model_strategy_mode,
                    ),
                )
                signals_df = out["test_df"]
                test_accuracy = out["technical_test_accuracy"]
                latest_proba = out["latest"]["proba"]
                latest_signal = out["latest"]["signal"]
            else:
                model_df, technical_cols = build_multi_timeframe_dataset(
                    timeframe_data,
                    cfg=MultiTimeframeDatasetConfig(
                        base_interval=base_interval,
                        context_intervals=context_intervals,
                        horizon_bars=horizon_bars,
                    ),
                )
                result = train_model(model_df, cfg=ModelConfig(model_type=model_type), feature_cols=technical_cols)
                proba_for_signal = result.test_proba
                if not advanced_mode and simple_use_fusion:
                    proba_for_signal = _simple_fusion_proba(result.test_proba, result.test_df["Date"], tech_weight=simple_tech_weight)
                signals_df = attach_signals(
                    result.test_df,
                    proba_for_signal,
                    threshold_mode="fixed" if threshold_mode == "Fixed" else "quantile",
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                    buy_quantile=buy_quantile,
                    sell_quantile=sell_quantile,
                )
                risk_frame = compute_risk_frame(prices, cfg=RiskConfig())
                signals_df = apply_risk_gating(signals_df, risk_frame)
                signals_df = enforce_trade_frequency(
                    signals_df,
                    max_trades_per_day=max_trades_per_day,
                    min_minutes_between_trades=min_minutes_between_trades,
                )
                latest_proba = float(signals_df.iloc[-1]["Proba"])
                latest_signal = str(signals_df.iloc[-1]["Signal"])
                test_accuracy = float(result.test_accuracy)
            signals_by_ticker[t] = signals_df

            latest_rows.append(
                {
                    "Ticker": t,
                    "Latest_Signal": latest_signal,
                    "Latest_Proba": latest_proba,
                    "Test_Accuracy": test_accuracy,
                }
            )
        except Exception as e:
            # Skip tickers that fail to download or cannot produce enough indicator data.
            st.warning(f"Skipping {t}: {e}")

    if not signals_by_ticker:
        raise RuntimeError("All tickers failed. Try editing the universe list or thresholds.")

    portfolio_backtest = backtest_portfolio_long_cash(
        signals_by_ticker,
        cfg=PortfolioBacktestConfig(initial_capital=float(initial_capital_total), allocation="equal"),
    )

    table_df = pd.DataFrame(latest_rows).sort_values("Ticker").reset_index(drop=True)
    return table_df, portfolio_backtest, {
        "optimization": {"enabled": False},
        "fundamentals_subset": funda_subset,
        "fundamentals_quality": funda_quality,
        "price_quality_universe": pd.DataFrame(price_quality_rows),
        "signals_by_ticker": signals_by_ticker,
        "strategy_mode": pipeline_strategy_mode,
    }


if run:
    try:
        ml_pack = None
        sector_pack = None
        with st.spinner("Fetching data, training model, predicting signals, and backtesting..."):
            if (not CORE_MODE) and fast_mode and universe_mode not in (
                "Multi-Layer (3-Stage)",
                "Sector Rotation (Flow)",
            ):
                st.info(
                    f"Fast profile enabled: using reduced context intervals {context_intervals} and optimizer evals={effective_optimizer_evals}."
                )
            if universe_mode == "Multi-Layer (3-Stage)":
                if not MULTI_LAYER_MODE:
                    st.error("Enable `MULTI_LAYER_MODE` in `multi_layer_config.py` to run this pipeline.")
                else:
                    _ml_years_map = {
                        "Last 1 year (default)": 1.0,
                        "Last 3 years": 3.0,
                        "Full downloaded history": None,
                    }
                    ml_pack = _run_multi_layer_ui(
                        max_names=int(ml_max_names),
                        pipe_cfg=ml_pipe_cfg,
                        period=str(ml_period),
                        backtest_years=_ml_years_map.get(str(ml_bt_window), 1.0),
                    )
                benchmark_df = None
                latest = None
                test_acc = float("nan")
                backtest = {}
                counts = {}
                test_proba = np.array([])
                extra = {}
                table_df = pd.DataFrame()
                portfolio_backtest = {}
            elif universe_mode == "Sector Rotation (Flow)":
                if not SECTOR_ROTATION_MODE:
                    st.error("Enable `SECTOR_ROTATION_MODE` in `sector_rotation_config.py` to run this pipeline.")
                else:
                    _sr_years_map = {
                        "Last 1 year (default)": 1.0,
                        "Last 3 years": 3.0,
                        "Full downloaded history": None,
                    }
                    _mt = "gb" if "Gradient" in str(sr_model_label) else "rf"
                    sr_cfg = replace(
                        SectorRotationConfig(),
                        top_sectors=int(sr_top_sectors),
                        min_top_sectors=2,
                        model_type=_mt,
                        backtest_years=_sr_years_map.get(str(sr_bt_window), 1.0),
                        max_rolling_vol_10=float(sr_max_vol),
                    )
                    sector_pack = _run_sector_rotation_ui(period=str(sr_period), sr_cfg=sr_cfg)
                benchmark_df = None
                latest = None
                test_acc = float("nan")
                backtest = {}
                counts = {}
                test_proba = np.array([])
                extra = {}
                table_df = pd.DataFrame()
                portfolio_backtest = {}
            elif universe_mode == "Single Ticker":
                latest, test_acc, backtest, counts, test_proba, extra = _run_single_pipeline(ticker)
                benchmark_df = None
                if run_benchmark_suite and stack_mode == "Four-Model Stack":
                    benchmark_df = _run_single_benchmark_suite(
                        ticker.strip().upper(),
                        initial_capital=float(initial_capital_total),
                        max_runs=int(benchmark_max_runs),
                        use_optimizer=bool(benchmark_use_optimizer),
                        optimizer_evals=int(effective_optimizer_evals),
                        use_meta_fusion=bool(use_meta_fusion),
                        context_intervals=context_intervals,
                        fast_mode=fast_mode,
                    )
            elif universe_mode in ("NIFTY 50 Portfolio", "NIFTY Midcap 150 Portfolio"):
                table_df, portfolio_backtest, extra = _run_nifty50_pipeline()
                benchmark_df = None

        if universe_mode == "Multi-Layer (3-Stage)" and ml_pack is not None:
            snap = ml_pack["snapshot"]
            bt = ml_pack["backtest"]
            st.success(
                f"Universe: requested **{ml_pack['universe_requested']}** symbols → **{ml_pack['universe_loaded']}** after pre-filter. "
                f"Layer 1 passed **{len(snap['layer1_tickers'])}** names."
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Layer 1 (fast filter)", f"{len(snap['layer1_tickers'])} names")
            with c2:
                st.metric("Layer 2 (ML shortlist)", f"{len(snap['layer2_table'])} rows")
            with c3:
                st.metric("Layer 3 (portfolio)", f"{len(snap['final_tickers'])} picks")

            st.markdown("### Latest snapshot — final selection (equal-weight target)")
            fin = snap.get("layer3_table")
            if fin is not None and len(fin) > 0:
                st.dataframe(fin, use_container_width=True, hide_index=True)
                alloc = 100.0 / max(1, len(fin))
                st.caption(f"Target allocation ≈ **{alloc:.1f}%** each (before frictions). Same pipeline runs at **each month-end** in the backtest below.")
            else:
                st.warning("No names passed Layer 3 gates — relax probability / Layer 1–2 sizes.")

            with st.expander("Layer 1 detail (momentum rank)", expanded=False):
                st.dataframe(snap.get("layer1_table", pd.DataFrame()), use_container_width=True, hide_index=True)
            with st.expander("Layer 2 detail (ML scores)", expanded=False):
                st.dataframe(snap.get("layer2_table", pd.DataFrame()), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("## Walk-forward portfolio analytics")
            _wr = float(bt.get("rebalance_win_rate", float("nan")))
            st.caption(
                f"Window: **{ml_bt_window}** · Month-end **rebalances** (Layer 1→3 each time): **{int(bt.get('rebalance_count', 0))}** · "
                f"Holding **segments** (trades to next rebalance): **{int(bt.get('rebalance_segments', 0))}** · "
                f"Segment win rate: **{_wr:.1%}** (fraction of segments with positive PnL)."
            )
            _pbt = bt["portfolio_backtest"]
            _bbt = bt["benchmark_backtest"]
            render_portfolio_quant_dashboard(
                portfolio_backtest=_pbt,
                benchmark_bt=_bbt,
                initial_capital=float(initial_capital_total),
                extra={
                    "signals_by_ticker": {},
                    "multi_layer_walk_forward": True,
                    "rebalance_log": bt.get("rebalance_log"),
                },
                core_mode=CORE_MODE,
            )
            _rlog = bt.get("rebalance_log") or []
            if _rlog:
                with st.expander("Rebalance log (picks at each month-end)", expanded=False):
                    _lr = []
                    for e in _rlog:
                        pk = e.get("Picks") or []
                        s = ", ".join(pk[:25]) + (" …" if len(pk) > 25 else "")
                        _lr.append({"RebalanceDate": e.get("Date"), "N": e.get("N"), "Picks": s})
                    st.dataframe(pd.DataFrame(_lr), use_container_width=True, hide_index=True)

            _tdf = _pbt.get("trades_df")
            if _tdf is not None and isinstance(_tdf, pd.DataFrame) and len(_tdf) > 0:
                with st.expander("Holding segments — PnL vs start of segment", expanded=False):
                    st.dataframe(_tdf, use_container_width=True, hide_index=True)

            latest_signal_placeholder.empty()
            latest_proba_placeholder.empty()
            accuracy_placeholder.empty()
            backtest_placeholder.empty()
            signal_counts_placeholder.empty()
            proba_hist_placeholder.empty()
            portfolio_table_placeholder.empty()

        elif universe_mode == "Sector Rotation (Flow)" and sector_pack is not None:
            snap = sector_pack["snapshot"]
            bt = sector_pack["backtest"]
            st.success(
                f"Sector indices: **{sector_pack['indices_loaded']}** downloaded · "
                f"Universe stocks: **{sector_pack['stocks_loaded']}** with history."
            )
            st.markdown("### Sector ranking")
            srt = snap.get("sector_rank_table")
            if srt is not None and isinstance(srt, pd.DataFrame) and len(srt) > 0:
                st.dataframe(srt, use_container_width=True, hide_index=True)
            else:
                st.warning("No sector scores computed.")
            st.markdown("### Active top sectors (this rebalance logic)")
            st.write(", ".join(snap.get("selected_sectors") or []) or "—")
            st.markdown("### Final portfolio (equal weight)")
            ptab = snap.get("portfolio_table")
            if ptab is not None and isinstance(ptab, pd.DataFrame) and len(ptab) > 0:
                st.dataframe(ptab, use_container_width=True, hide_index=True)
            else:
                st.warning("No stocks selected — try Random Forest, relax vol filter, or extend history.")

            for sel in snap.get("selections") or []:
                sk = str(sel.get("SectorKey", ""))
                with st.expander(f"Sector **{sk}** → pick **{sel.get('Ticker', '')}** (P={float(sel.get('Proba', float('nan'))):.3f})", expanded=False):
                    cands = sel.get("Candidates") or []
                    st.dataframe(pd.DataFrame(cands), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("## Weekly walk-forward portfolio analytics")
            _wr_s = float(bt.get("rebalance_win_rate", float("nan")))
            st.caption(
                f"Backtest window: **{sr_bt_window}** · Weekly rebalance events: **{int(bt.get('rebalance_count', 0))}** · "
                f"Holding segments: **{int(bt.get('rebalance_segments', 0))}** · Segment win rate: **{_wr_s:.1%}** · "
                f"Benchmark: **Nifty (^NSEI)** buy & hold on same calendar."
            )
            _ps = bt["portfolio_backtest"]
            _bs = bt["benchmark_backtest"]
            render_portfolio_quant_dashboard(
                portfolio_backtest=_ps,
                benchmark_bt=_bs,
                initial_capital=float(initial_capital_total),
                extra={
                    "signals_by_ticker": {},
                    "sector_rotation_walk_forward": True,
                    "rebalance_log": bt.get("rebalance_log"),
                },
                core_mode=CORE_MODE,
            )
            _srl = bt.get("rebalance_log") or []
            if _srl:
                with st.expander("Weekly rebalance log", expanded=False):
                    rows = []
                    for e in _srl:
                        pk = e.get("Picks") or []
                        rows.append(
                            {
                                "WeekAnchor": e.get("Date"),
                                "N": e.get("N"),
                                "Sectors": ", ".join(e.get("Sectors") or []),
                                "Picks": ", ".join(pk[:12]) + (" …" if len(pk) > 12 else ""),
                            }
                        )
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            latest_signal_placeholder.empty()
            latest_proba_placeholder.empty()
            accuracy_placeholder.empty()
            backtest_placeholder.empty()
            signal_counts_placeholder.empty()
            proba_hist_placeholder.empty()
            portfolio_table_placeholder.empty()

        elif universe_mode == "Single Ticker":
            latest_signal_placeholder.metric("Latest Signal (for next day)", latest["signal"])
            _proba_lbl = (
                "Model probability (P(meaningful forward move > threshold over horizon))"
                if TECHNICAL_MODE
                else "Model Probability (P(next return > 0))"
            )
            latest_proba_placeholder.metric(_proba_lbl, f"{latest['proba']:.3f}")
            accuracy_placeholder.info(f"Test Accuracy: {test_acc:.3f}")
            backtest_placeholder.success(
                f"Final Portfolio Value: {backtest['final_portfolio_value']:.2f} | Total Return: {backtest['total_return_pct']:.2f}%"
            )
            wf_msg = ""
            if not CORE_MODE:
                wf_msg = f" | Walk-forward mean accuracy: {extra.get('walk_forward_mean_accuracy', float('nan')):.3f}"
            st.write(
                f"Risk Score: {latest.get('risk_score', float('nan')):.3f} | "
                f"Position Size: {latest.get('position_size', float('nan')):.2f}"
                f"{wf_msg}"
            )
            _sm = extra.get("strategy_mode") if isinstance(extra, dict) else None
            if _sm:
                st.success(f"Pipeline strategy mode: `{_sm}` (features + signal logic for this run).")

            with st.expander("Top-Level Requirements & Data Health", expanded=False):
                st.write("Requirements status")
                st.write(f"Universe mode: {universe_mode}")
                st.write(f"Model stack: {stack_mode}")
                st.write(f"Pipeline mode: {pipeline_mode}")
                st.write(f"Execution interval: {base_interval}")
                if not CORE_MODE:
                    st.write(f"Auto optimization enabled: {auto_optimize}")
                    st.write(f"Comprehensive benchmark enabled: {run_benchmark_suite}")
                    if not advanced_mode:
                        st.write(f"Simple fusion enabled: {extra.get('simple_fusion_enabled', False)}")
                        st.write(f"Simple fusion technical weight: {extra.get('simple_fusion_tech_weight', float('nan')):.2f}")

                tf_data = extra.get("timeframe_data") if isinstance(extra, dict) else None
                price_quality = extra.get("price_quality_report") if isinstance(extra, dict) else None
                if price_quality is None or len(price_quality) == 0:
                    if isinstance(tf_data, dict):
                        price_quality = _build_data_health_table(tf_data)
                    elif ticker and ticker.strip():
                        try:
                            qb = load_clean_data_api_cached(ticker.strip().upper(), base_interval, context_intervals, horizon_bars)
                            price_quality = qb.get("price_quality_report")
                        except Exception:
                            price_quality = pd.DataFrame()

                if price_quality is not None and len(price_quality) > 0:
                    st.markdown("Price data quality by timeframe")
                    st.dataframe(price_quality, use_container_width=True, hide_index=True)

                funda_quality = extra.get("fundamentals_quality") if isinstance(extra, dict) else None
                funda_ticker = extra.get("fundamentals_ticker") if isinstance(extra, dict) else None
                if funda_quality is None or len(funda_quality) == 0:
                    try:
                        funda_df = load_nifty50_fundamentals_cached(force_refresh=refresh_fundamentals)
                        funda_quality = build_fundamentals_quality_report(funda_df)
                        funda_ticker = get_fundamentals_for_ticker(funda_df, ticker)
                    except Exception:
                        funda_quality = pd.DataFrame()
                        funda_ticker = pd.DataFrame()

                if (not CORE_MODE) and funda_quality is not None and len(funda_quality) > 0:
                    st.markdown("Fundamentals cache quality")
                    st.dataframe(funda_quality, use_container_width=True, hide_index=True)
                if (not CORE_MODE) and funda_ticker is not None and len(funda_ticker) > 0:
                    st.markdown("Selected ticker fundamentals")
                    st.dataframe(funda_ticker, use_container_width=True, hide_index=True)
            opt_pack = extra.get("optimization") if isinstance(extra, dict) else None
            if isinstance(opt_pack, dict) and opt_pack.get("enabled"):
                best_cfg = opt_pack.get("best_config")
                st.success(
                    f"Optimizer regime={opt_pack.get('regime')} | score={opt_pack.get('score', float('nan')):.2f} | "
                    f"evaluated={opt_pack.get('evaluated', 0)}"
                )
                if best_cfg is not None:
                    st.write(
                        f"Best params: horizon_bars={best_cfg.horizon_bars}, buy_q={best_cfg.buy_quantile:.2f}, "
                        f"sell_q={best_cfg.sell_quantile:.2f}, max_trades/day={best_cfg.max_trades_per_day}, "
                        f"min_gap_min={best_cfg.min_minutes_between_trades}, "
                        f"hard_vol_stop={best_cfg.risk_config.hard_vol_stop:.3f}, "
                        f"hard_dd_stop={best_cfg.risk_config.hard_drawdown_stop:.3f}"
                    )
                leaderboard = opt_pack.get("leaderboard")
                if leaderboard is not None and len(leaderboard) > 0:
                    st.markdown("### Optimizer Leaderboard (Top 10)")
                    optimizer_table_placeholder.dataframe(leaderboard.head(10), use_container_width=True, hide_index=True)
            signal_counts_placeholder.write(f"Signal counts on test: {counts}")

            if counts.get("BUY", 0) == 0 or counts.get("SELL", 0) == 0:
                model_counts = {}
                if "SignalModel" in extra.get("test_df", pd.DataFrame()).columns:
                    model_counts = extra["test_df"]["SignalModel"].value_counts().to_dict()
                elif "SignalModel" in extra and hasattr(extra["SignalModel"], "value_counts"):
                    model_counts = extra["SignalModel"].value_counts().to_dict()
                elif "SignalModel" in locals():
                    model_counts = {}

                if isinstance(extra, dict) and "test_df" in extra and "SignalModel" in extra["test_df"].columns:
                    model_counts = extra["test_df"]["SignalModel"].value_counts().to_dict()

                if model_counts and (model_counts.get("BUY", 0) > 0 and model_counts.get("SELL", 0) > 0):
                    st.warning(
                        "Model produced BUY/SELL signals, but risk gating removed one side into HOLD. "
                        "Try loosening risk limits or trade-frequency constraints."
                    )
                else:
                    st.warning(
                        "Signal distribution is still one-sided in this run. "
                        + (
                            "In CORE_MODE fixed thresholds (0.6 / 0.4) usually indicate flat classifier probabilities or risk gates blocking trades."
                            if CORE_MODE
                            else "Quantile mapping is enabled, so this often means flat probabilities or heavy risk constraints."
                        )
                    )

            proba_hist_placeholder.empty()

            st.divider()
            st.markdown("## Analytics dashboard")
            _tdf = extra.get("test_df") if isinstance(extra, dict) else None
            _bh_single = None
            if _tdf is not None and isinstance(_tdf, pd.DataFrame) and len(_tdf) > 0:
                _bh_single = backtest_buy_and_hold(_tdf, cfg=BacktestConfig(initial_capital=float(initial_capital_total)))
            render_single_ticker_quant_dashboard(
                ticker=ticker,
                latest=latest,
                test_acc=test_acc,
                backtest=backtest,
                benchmark_bt=_bh_single,
                extra=extra if isinstance(extra, dict) else {},
                test_proba=test_proba,
                counts=counts,
                initial_capital=float(initial_capital_total),
                core_mode=CORE_MODE,
            )

            if benchmark_df is not None and len(benchmark_df) > 0:
                st.markdown("### Comprehensive Benchmark Suite")
                st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
                best_row = benchmark_df.iloc[0]
                st.success(
                    f"Best benchmark: model={best_row.get('Model')}, interval={best_row.get('BaseInterval')}, "
                    f"horizon={best_row.get('HorizonBars')} | return={best_row.get('ReturnPct', float('nan')):.2f}%"
                )
                st.download_button(
                    "Download benchmark CSV",
                    data=benchmark_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"benchmark_{ticker.strip().upper()}.csv",
                    mime="text/csv",
                )
        elif universe_mode in ("NIFTY 50 Portfolio", "NIFTY Midcap 150 Portfolio"):
            _plab = "NIFTY 50" if universe_mode == "NIFTY 50 Portfolio" else "NIFTY Midcap 150"
            latest_signal_placeholder.metric(f"{_plab} runs", f"{len(table_df)} tickers")
            latest_proba_placeholder.metric("Portfolio total return", f"{portfolio_backtest['total_return_pct']:.2f}%")

            accuracy_placeholder.info("Each ticker has its own test accuracy (see table).")
            backtest_placeholder.success(
                f"Final Portfolio Value: {portfolio_backtest['final_portfolio_value']:.2f} | Total Return: {portfolio_backtest['total_return_pct']:.2f}%"
            )
            st.write(
                f"Portfolio Sharpe: {portfolio_backtest.get('sharpe', float('nan')):.3f} | "
                f"Portfolio Max Drawdown: {portfolio_backtest.get('max_drawdown_pct', float('nan')):.2f}%"
            )
            _sm_p = extra.get("strategy_mode") if isinstance(extra, dict) else None
            if _sm_p:
                st.success(f"Pipeline strategy mode: `{_sm_p}` (applied per ticker in this basket).")

            with st.expander("Top-Level Requirements & Data Health", expanded=False):
                st.write("Requirements status")
                st.write(f"Universe mode: {universe_mode}")
                st.write(f"Model stack: {stack_mode}")
                st.write(f"Pipeline mode: {pipeline_mode}")
                st.write(f"Run profile: {run_profile}")
                st.write(f"2-stage workflow enabled: {run_two_stage}")
                st.write(f"Auto optimization enabled: {auto_optimize}")

                pq = extra.get("price_quality_universe") if isinstance(extra, dict) else None
                if pq is not None and len(pq) > 0:
                    st.markdown("Price quality by ticker (base interval)")
                    st.dataframe(pq.sort_values("Ticker"), use_container_width=True, hide_index=True)

                fq = extra.get("fundamentals_quality") if isinstance(extra, dict) else None
                fs = extra.get("fundamentals_subset") if isinstance(extra, dict) else None
                if fq is not None and len(fq) > 0:
                    st.markdown("Fundamentals quality (selected universe)")
                    st.dataframe(fq, use_container_width=True, hide_index=True)
                if fs is not None and len(fs) > 0:
                    st.markdown("Fundamentals snapshot (selected universe)")
                    st.dataframe(fs.sort_values("Ticker"), use_container_width=True, hide_index=True)

            opt_pack = extra.get("optimization") if isinstance(extra, dict) else None
            if isinstance(opt_pack, dict) and opt_pack.get("enabled"):
                best_cfg = opt_pack.get("best_config")
                st.success(
                    f"Portfolio optimizer regime={opt_pack.get('regime')} | score={opt_pack.get('score', float('nan')):.2f} | "
                    f"evaluated={opt_pack.get('evaluated', 0)}"
                )
                if best_cfg is not None:
                    st.write(
                        f"Best global params: horizon_bars={best_cfg.horizon_bars}, buy_q={best_cfg.buy_quantile:.2f}, "
                        f"sell_q={best_cfg.sell_quantile:.2f}, max_trades/day={best_cfg.max_trades_per_day}, "
                        f"min_gap_min={best_cfg.min_minutes_between_trades}, "
                        f"hard_vol_stop={best_cfg.risk_config.hard_vol_stop:.3f}, "
                        f"hard_dd_stop={best_cfg.risk_config.hard_drawdown_stop:.3f}"
                    )
                leaderboard = opt_pack.get("leaderboard")
                if leaderboard is not None and len(leaderboard) > 0:
                    title = "### Portfolio 2-Stage Validation Leaderboard (Top 10)" if opt_pack.get("mode") == "two-stage" else "### Portfolio Optimizer Leaderboard (Top 10)"
                    st.markdown(title)
                    optimizer_table_placeholder.dataframe(leaderboard.head(10), use_container_width=True, hide_index=True)
                research_top = opt_pack.get("research_top")
                if research_top is not None and len(research_top) > 0:
                    st.markdown("### Research Stage Top Configs")
                    st.dataframe(research_top, use_container_width=True, hide_index=True)

            portfolio_table_placeholder.dataframe(table_df, use_container_width=True, hide_index=True)
            _csv_name = (
                "portfolio_latest_signals_nifty50.csv"
                if universe_mode == "NIFTY 50 Portfolio"
                else "portfolio_latest_signals_midcap150.csv"
            )
            st.download_button(
                "Download portfolio table CSV",
                data=table_df.to_csv(index=False).encode("utf-8"),
                file_name=_csv_name,
                mime="text/csv",
            )

            st.divider()
            st.markdown("## Portfolio analytics dashboard")
            _sbt = extra.get("signals_by_ticker") if isinstance(extra, dict) else None
            _bh_p = None
            if isinstance(_sbt, dict) and len(_sbt) > 0:
                _bh_p = backtest_portfolio_buy_and_hold(
                    _sbt,
                    cfg=PortfolioBacktestConfig(initial_capital=float(initial_capital_total), allocation="equal"),
                )
            render_portfolio_quant_dashboard(
                portfolio_backtest=portfolio_backtest,
                benchmark_bt=_bh_p,
                initial_capital=float(initial_capital_total),
                extra=extra if isinstance(extra, dict) else {},
                core_mode=CORE_MODE,
            )
    except Exception as e:
        st.error(f"Failed to run pipeline: {e}")


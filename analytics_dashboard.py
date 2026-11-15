"""
Streamlit quant-style dashboard: tabs for overview, performance, risk, time-series, strategy insights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from backtest import (
    plot_core_combined_performance,
    plot_equity_comparison,
    plot_equity_curve,
)
from backtest_analytics import (
    build_full_analytics,
    plot_drawdown_curve,
    plot_monthly_returns_heatmap,
    plot_rolling_sharpe,
)
from model import feature_importance_dataframe


def _fmt_pct(x: float, d: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{d}f}%"


def _fmt_num(x: float, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (float, np.floating)) and np.isinf(x):
        return "∞"
    return f"{x:.{d}f}"


def _metric_value(metrics: dict | None, key: str) -> float:
    if not isinstance(metrics, dict):
        return float("nan")
    try:
        return float(metrics.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def render_model_reliability(reliability: dict | None) -> None:
    """Walk-forward reliability, calibration, and held-out explanations."""
    rel = reliability if isinstance(reliability, dict) else {}
    if rel.get("evaluation") != "walk_forward":
        reason = rel.get("reason") or "Walk-forward output is not available for this run."
        st.info(reason)
        st.caption(
            "The performance tab is then the chronological holdout. "
            "That split is a supplementary diagnostic, not the primary out-of-sample score."
        )
        return

    method = str(rel.get("decision_method", "raw"))
    st.caption(
        "In-sample accuracy is the last fit window. "
        "The calibration window is where a probability map was fit, so its error is not out-of-sample. "
        "Out-of-sample rows were used for neither the classifier nor the calibrator. "
        f"Decisions use `{method}` probabilities."
    )
    oos = rel.get("oos_metrics") or {}
    raw = rel.get("oos_raw_metrics") or {}
    window = rel.get("calibration_window_metrics") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OOS accuracy", _fmt_num(_metric_value(oos, "accuracy"), 3))
    c2.metric("In-sample accuracy", _fmt_num(float(rel.get("insample_accuracy", float("nan"))), 3))
    c3.metric("OOS Brier (decision)", _fmt_num(_metric_value(oos, "brier"), 3))
    c4.metric("OOS Brier (raw)", _fmt_num(_metric_value(raw, "brier"), 3))
    if window:
        st.caption(
            "Calibration-window Brier, raw "
            f"{_fmt_num(float(window.get('brier_raw', float('nan'))), 3)} versus calibrated "
            f"{_fmt_num(float(window.get('brier_calibrated', float('nan'))), 3)}. "
            "This window is validation for the calibrator, not a test fold."
        )

    st.subheader("Walk-forward folds")
    folds = rel.get("folds")
    if isinstance(folds, pd.DataFrame) and len(folds):
        show = [
            c
            for c in (
                "fold",
                "test_start_date",
                "test_end_date",
                "n_fit",
                "n_calib",
                "n_test",
                "decision_method",
                "accuracy",
                "brier_raw",
                "brier_decision",
                "brier_sigmoid",
                "brier_isotonic",
            )
            if c in folds.columns
        ]
        st.dataframe(folds[show], use_container_width=True, hide_index=True)
    else:
        st.caption("No fold table was returned.")

    st.subheader("Raw versus calibrated probabilities")
    series = [
        ("Out-of-sample raw", raw),
        ("Out-of-sample decision", oos),
        ("Out-of-sample sigmoid", rel.get("oos_sigmoid_metrics")),
        ("Out-of-sample isotonic", rel.get("oos_isotonic_metrics")),
    ]
    metric_rows = []
    for label, metrics in series:
        if not isinstance(metrics, dict):
            continue
        metric_rows.append(
            {
                "Series": label,
                "Rows": metrics.get("n"),
                "Accuracy": _fmt_num(_metric_value(metrics, "accuracy"), 3),
                "Precision": _fmt_num(_metric_value(metrics, "precision"), 3),
                "Recall": _fmt_num(_metric_value(metrics, "recall"), 3),
                "F1": _fmt_num(_metric_value(metrics, "f1"), 3),
                "Balanced accuracy": _fmt_num(_metric_value(metrics, "balanced_accuracy"), 3),
                "ROC-AUC": _fmt_num(_metric_value(metrics, "roc_auc"), 3),
                "PR-AUC": _fmt_num(_metric_value(metrics, "pr_auc"), 3),
                "Brier": _fmt_num(_metric_value(metrics, "brier"), 3),
                "Log loss": _fmt_num(_metric_value(metrics, "log_loss"), 3),
                "Calibration error": _fmt_num(_metric_value(metrics, "ece"), 3),
            }
        )
    if metric_rows:
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
        st.caption(
            "ROC-AUC and PR-AUC are omitted when a sample has one class or fewer than 20 rows. "
            "Precision, recall, and F1 use a 0.5 probability cutoff. "
            "Trading still uses the 0.6 / 0.4 thresholds plus the MA50 rule."
        )

    plt = _pyplot()

    left, right = st.columns(2)
    with left:
        st.pyplot(_reliability_figure(rel.get("reliability_bins_raw"), "Raw out-of-sample reliability"), use_container_width=True)
    with right:
        st.pyplot(
            _reliability_figure(rel.get("reliability_bins"), f"Decision probabilities ({method})"),
            use_container_width=True,
        )
    plt.close("all")

    st.subheader("Prediction confidence")
    proba = np.asarray(rel.get("oos_proba_decision", []), dtype=float)
    if proba.size:
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.hist(proba[np.isfinite(proba)], bins=20, color="#1f4b99", alpha=0.85)
        ax.set_xlabel("Out-of-sample decision probability")
        ax.set_ylabel("Bars")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Confusion matrix")
    st.caption("Classifier cutoff is 0.5 on out-of-sample decision probabilities. This is not the trade list.")
    confusion = rel.get("confusion")
    if isinstance(confusion, pd.DataFrame) and len(confusion):
        st.dataframe(confusion, use_container_width=True)
    else:
        st.caption("A confusion matrix needs both classes in the out-of-sample labels.")

    st.subheader("Feature importance")
    st.caption(
        "Permutation importance shuffles each feature on held-out walk-forward rows and measures the drop in negative Brier score. "
        "FoldStd is the variation across folds. This is a statistical association, not a causal explanation."
    )
    importance = rel.get("importance")
    if isinstance(importance, pd.DataFrame) and len(importance):
        st.dataframe(importance, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(8, max(2.6, 0.38 * len(importance))))
        plot_df = importance.iloc[::-1]
        xerr = plot_df["FoldStd"] if "FoldStd" in plot_df.columns else None
        ax.barh(plot_df["Feature"], plot_df["Importance"], xerr=xerr, color="#1f4b99", alpha=0.9)
        ax.set_xlabel("Mean held-out importance")
        ax.grid(True, axis="x", alpha=0.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.caption("Held-out permutation importance was not computed. The out-of-sample folds were too small, or repeats were disabled.")

    st.subheader("Latest prediction")
    explained = rel.get("explained_probability", float("nan"))
    explained_date = rel.get("explained_date")
    st.write(
        f"Scored bar: {explained_date}. Decision probability: {_fmt_num(float(explained) if explained is not None else float('nan'), 3)}."
    )
    st.caption(
        "Each row replaces one live feature with the median of the last held-out fold and re-scores the same model. "
        "ProbabilityDelta is the original probability minus that replaced probability. "
        "A positive delta means the live value scored higher than the held-out median. "
        "Features are not independent, and none of these deltas is a cause."
    )
    local = rel.get("local_explanation")
    if isinstance(local, pd.DataFrame) and len(local):
        st.dataframe(local, use_container_width=True, hide_index=True)
    else:
        st.caption("No single-bar sensitivity was computed for this run.")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _reliability_figure(bins, title: str):
    plt = _pyplot()

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7f8c8d", label="Matched frequency")
    if isinstance(bins, pd.DataFrame) and len(bins):
        ax.plot(bins["confidence"], bins["frequency"], marker="o", color="#1f4b99", label="Observed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def render_single_ticker_quant_dashboard(
    *,
    ticker: str,
    latest: dict,
    test_acc: float,
    backtest: dict,
    benchmark_bt: dict | None,
    extra: dict,
    test_proba: np.ndarray,
    counts: dict,
    initial_capital: float,
    core_mode: bool,
) -> None:
    """Full analytics for one symbol after a successful single-ticker run."""
    tdf = extra.get("test_df") if isinstance(extra, dict) else None
    if tdf is None or not isinstance(tdf, pd.DataFrame) or len(tdf) == 0:
        st.warning("No test-period dataframe available for analytics.")
        return

    analytics = build_full_analytics(
        backtest,
        initial_capital=float(initial_capital),
        benchmark=benchmark_bt,
    )
    perf = analytics["performance"]
    risk = analytics["risk"]
    radj = analytics["risk_adjusted"]
    tr = analytics["trades"]
    strat = analytics["strategy"]
    bench = analytics["benchmark"]
    series = analytics["series"]
    trades_df = analytics["trades_df"]

    sym = str(ticker).strip().upper()
    last_close = float(tdf["Close"].iloc[-1]) if "Close" in tdf.columns else float("nan")

    tab_ov, tab_rel, tab_perf, tab_risk, tab_time, tab_strat = st.tabs(
        ["Overview", "Reliability", "Performance", "Risk & trades", "Time analysis", "Strategy"]
    )

    with tab_ov:
        st.subheader("Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Symbol", sym)
        c2.metric("Latest signal", str(latest.get("signal", "—")))
        c3.metric("P(up)", _fmt_num(float(latest.get("proba", float("nan"))), 3))
        c4.metric("Last close (test bar)", _fmt_num(last_close, 2))
        rel = extra.get("reliability") if isinstance(extra, dict) else None
        if isinstance(rel, dict) and rel.get("evaluation") == "walk_forward":
            split = st.columns(3)
            split[0].metric("Out-of-sample accuracy", _fmt_num(_metric_value(rel.get("oos_metrics"), "accuracy"), 3))
            split[1].metric("In-sample accuracy", _fmt_num(float(rel.get("insample_accuracy", float("nan"))), 3))
            split[2].metric("Holdout accuracy", _fmt_num(float(test_acc), 3))
            st.caption(
                "Out-of-sample is the walk-forward test. In-sample is the last fit window. "
                "Holdout accuracy is the single chronological split and is not the headline score."
            )
        else:
            st.caption("Chronological holdout accuracy is a classifier metric, not live trading performance.")
            st.metric("Holdout accuracy", _fmt_num(float(test_acc), 3))
        if core_mode and benchmark_bt is not None:
            st.markdown("#### Indexed performance vs price & benchmark")
            fig_c = plot_core_combined_performance(
                strategy_equity=backtest["equity_curve"],
                test_df=tdf,
                buy_hold_equity=benchmark_bt["equity_curve"],
                underlying_label=f"{sym} close (normalized)",
            )
            st.pyplot(fig_c, use_container_width=True)
        else:
            st.markdown("#### Equity curve")
            st.pyplot(plot_equity_curve(backtest["equity_curve"], title=f"{sym} — strategy equity"), use_container_width=True)
            if benchmark_bt is not None:
                st.pyplot(
                    plot_equity_comparison(
                        backtest["equity_curve"],
                        benchmark_bt["equity_curve"],
                        title="Strategy vs buy & hold",
                    ),
                    use_container_width=True,
                )

    with tab_rel:
        st.subheader("Model reliability")
        render_model_reliability(extra.get("reliability") if isinstance(extra, dict) else None)

    with tab_perf:
        st.subheader("Performance summary")
        r1 = st.columns(4)
        r1[0].metric("Net return", _fmt_pct(perf.get("net_return_pct", perf["total_return_pct"])))
        r1[1].metric("Gross return", _fmt_pct(perf.get("gross_return_pct", perf["total_return_pct"])))
        r1[2].metric("CAGR", _fmt_pct(perf["cagr_pct"]))
        r1[3].metric("Final value", f"{perf['final_portfolio_value']:,.0f}")
        st.caption(
            f"Transaction costs (model assumptions): {perf.get('total_transaction_costs', 0.0):,.2f}. "
            "Signals fill at the next session open. "
            + (
                "This equity curve uses walk-forward out-of-sample probabilities, not the in-sample fit."
                if isinstance(extra, dict) and (extra.get("reliability") or {}).get("evaluation") == "walk_forward"
                else "This equity curve uses the chronological holdout when walk-forward output is unavailable."
            )
        )
        r2 = st.columns(3)
        r2[0].metric("Sharpe (ann.)", _fmt_num(radj["sharpe"], 2))
        r2[1].metric("Sortino (ann.)", _fmt_num(radj["sortino"], 2))
        r2[2].metric("Calmar", _fmt_num(radj["calmar"], 2))
        if bench:
            st.markdown("#### vs buy & hold")
            bc = st.columns(4)
            bc[0].metric("BH total return", _fmt_pct(bench.get("bh_total_return_pct", float("nan"))))
            bc[1].metric("BH CAGR", _fmt_pct(bench.get("bh_cagr", float("nan"))))
            bc[2].metric("BH Sharpe", _fmt_num(bench.get("bh_sharpe", float("nan")), 2))
            bc[3].metric("BH max DD", _fmt_pct(bench.get("bh_max_dd_pct", float("nan"))))

    with tab_risk:
        st.subheader("Risk")
        rc = st.columns(3)
        rc[0].metric("Max drawdown", _fmt_pct(risk["max_drawdown_pct"]))
        rc[1].metric("Avg. drawdown (underwater)", _fmt_pct(risk["avg_drawdown_pct"]))
        rc[2].metric("Volatility (ann.)", _fmt_pct(risk["volatility_ann_pct"]))
        st.pyplot(plot_drawdown_curve(series["drawdown"]), use_container_width=True)

        st.subheader("Trade analytics")
        if tr["total_trades"] > 0:
            tc = st.columns(4)
            tc[0].metric("Total round-trips", str(tr["total_trades"]))
            tc[1].metric(
                "Win rate",
                _fmt_pct(tr["win_rate"] * 100.0) if tr["total_trades"] > 0 and np.isfinite(tr["win_rate"]) else "—",
            )
            tc[2].metric("Profit factor", _fmt_num(tr["profit_factor"], 2))
            tc[3].metric("Expectancy ($)", _fmt_num(tr["expectancy"], 2))
            tc2 = st.columns(4)
            tc2[0].metric("Avg win ($)", _fmt_num(tr["avg_win"], 2))
            tc2[1].metric("Avg loss ($)", _fmt_num(tr["avg_loss"], 2))
            tc2[2].metric("Max win streak", str(tr["max_consecutive_wins"]))
            tc2[3].metric("Max loss streak", str(tr["max_consecutive_losses"]))
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("No completed round-trip trades in the test window (all HOLD or open position at end).")
        st.metric("Exposure (% of bars long)", _fmt_pct(strat["exposure_pct"]))

    with tab_time:
        st.subheader("Time analysis")
        mp = series["monthly_returns_pivot"]
        if mp is not None and not mp.empty:
            st.pyplot(plot_monthly_returns_heatmap(mp), use_container_width=True)
        else:
            st.caption("Not enough calendar depth for monthly heatmap.")
        rs = series["rolling_sharpe"]
        if rs is not None and len(rs) > 0:
            st.pyplot(plot_rolling_sharpe(rs, title="Rolling Sharpe (60 trading days)"), use_container_width=True)
        else:
            st.caption("Not enough observations for rolling Sharpe.")

    with tab_strat:
        st.subheader("Strategy comparison")
        rows = [
            {
                "Metric": "Total return %",
                "Strategy": _fmt_pct(perf["total_return_pct"], 4),
                "Buy & hold": _fmt_pct(bench.get("bh_total_return_pct", float("nan")), 4) if bench else "—",
            },
            {
                "Metric": "CAGR %",
                "Strategy": _fmt_pct(perf["cagr_pct"], 4),
                "Buy & hold": _fmt_pct(bench.get("bh_cagr", float("nan")), 4) if bench else "—",
            },
            {
                "Metric": "Sharpe",
                "Strategy": _fmt_num(radj["sharpe"], 4),
                "Buy & hold": _fmt_num(bench.get("bh_sharpe", float("nan")), 4) if bench else "—",
            },
            {
                "Metric": "Max DD %",
                "Strategy": _fmt_pct(risk["max_drawdown_pct"], 4),
                "Buy & hold": _fmt_pct(bench.get("bh_max_dd_pct", float("nan")), 4) if bench else "—",
            },
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Signals (test period)")
        st.write(f"Execution counts: **{counts}**")
        if isinstance(tdf, pd.DataFrame) and "SignalModel" in tdf.columns:
            st.write("Pre-risk model signals:", tdf["SignalModel"].value_counts().to_dict())

        st.subheader("Feature importance (trained classifier)")
        mdl = extra.get("trained_model")
        fn = extra.get("feature_names")
        if mdl is not None and fn and isinstance(fn, list) and len(fn) > 0:
            fi = feature_importance_dataframe(mdl, fn)
            if fi is not None and len(fi) > 0:
                st.dataframe(fi, use_container_width=True, hide_index=True)
            else:
                st.caption(
                    "Native importances are unavailable for HistGradientBoosting. "
                    "Held-out permutation importance, when the walk-forward run produced it, is on the Reliability tab."
                )
        else:
            st.caption("Feature importance is available for the four-model / full pipeline runs with a fitted classifier.")

        if not core_mode and test_proba is not None and len(test_proba) > 0:
            st.subheader("Predicted probability (test)")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 2.8))
            ax.hist(np.asarray(test_proba, dtype=float), bins=30, color="#34495e", alpha=0.85)
            ax.set_title("Distribution of P(up) on test set")
            ax.grid(True, alpha=0.25)
            st.pyplot(fig, use_container_width=True)


def render_portfolio_quant_dashboard(
    *,
    portfolio_backtest: dict,
    benchmark_bt: dict | None,
    initial_capital: float,
    extra: dict,
    core_mode: bool,
) -> None:
    """Aggregated portfolio analytics (trade-level stats are limited at basket level)."""
    analytics = build_full_analytics(
        portfolio_backtest,
        initial_capital=float(initial_capital),
        benchmark=benchmark_bt,
    )
    perf = analytics["performance"]
    risk = analytics["risk"]
    radj = analytics["risk_adjusted"]
    bench = analytics["benchmark"]
    series = analytics["series"]

    st.subheader("Portfolio analytics")
    r1 = st.columns(4)
    r1[0].metric("Net return", _fmt_pct(perf.get("net_return_pct", perf["total_return_pct"])))
    r1[1].metric("Gross return", _fmt_pct(perf.get("gross_return_pct", perf["total_return_pct"])))
    r1[2].metric("CAGR", _fmt_pct(perf["cagr_pct"]))
    r1[3].metric("Sharpe", _fmt_num(radj["sharpe"], 2))
    st.caption(
        f"Transaction costs (model assumptions): {perf.get('total_transaction_costs', 0.0):,.2f}. "
        f"Final value {perf['final_portfolio_value']:,.0f}."
    )
    if isinstance(extra, dict) and extra.get("evaluation") == "walk_forward":
        st.caption(
            "Each name is scored with purged walk-forward probabilities. "
            "Test_Accuracy remains the single holdout diagnostic. OOS_Accuracy is the walk-forward score. "
            "A higher classification score is not the same thing as a higher net return."
        )
    r2 = st.columns(4)
    r2[0].metric("Max DD", _fmt_pct(risk["max_drawdown_pct"]))
    r2[1].metric("Sortino", _fmt_num(radj["sortino"], 2))
    r2[2].metric("Calmar", _fmt_num(radj["calmar"], 2))
    r2[3].metric("Vol (ann.)", _fmt_pct(risk["volatility_ann_pct"]))

    tstats = analytics["trades"]
    if int(tstats.get("total_trades", 0) or 0) > 0:
        st.markdown("#### Rebalance holding segments (trade statistics)")
        r3 = st.columns(4)
        r3[0].metric("Segments / trades", str(int(tstats["total_trades"])))
        r3[1].metric("Win rate", _fmt_pct(float(tstats.get("win_rate", float("nan"))) * 100.0))
        r3[2].metric("Avg win", f"{float(tstats.get('avg_win', float('nan'))):,.0f}")
        r3[3].metric("Avg loss", f"{float(tstats.get('avg_loss', float('nan'))):,.0f}")
        st.caption(
            "Each segment is one equal-weight basket from a month-end rebalance through the next rebalance. "
            "Metrics use segment PnL in currency (same units as initial capital)."
        )

    sig = extra.get("signals_by_ticker")
    if core_mode and isinstance(sig, dict) and len(sig) > 0 and benchmark_bt is not None:
        from backtest import plot_core_portfolio_combined_performance

        fig = plot_core_portfolio_combined_performance(
            strategy_equity=portfolio_backtest["equity_curve"],
            signals_by_ticker=sig,
            buy_hold_equity=benchmark_bt["equity_curve"],
        )
        st.pyplot(fig, use_container_width=True)
    else:
        st.pyplot(
            plot_equity_curve(portfolio_backtest["equity_curve"], title="Portfolio equity"),
            use_container_width=True,
        )
        if benchmark_bt is not None:
            st.pyplot(
                plot_equity_comparison(
                    portfolio_backtest["equity_curve"],
                    benchmark_bt["equity_curve"],
                    title="Portfolio vs buy & hold",
                ),
                use_container_width=True,
            )

    st.pyplot(plot_drawdown_curve(series["drawdown"]), use_container_width=True)
    mp = series["monthly_returns_pivot"]
    if mp is not None and not mp.empty:
        st.pyplot(plot_monthly_returns_heatmap(mp, title="Portfolio monthly returns (%)"), use_container_width=True)
    rs = series["rolling_sharpe"]
    if rs is not None and len(rs) > 0:
        st.pyplot(plot_rolling_sharpe(rs), use_container_width=True)

    if bench:
        st.markdown("#### Strategy vs buy & hold (portfolio)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Metric": "Total return %",
                        "Strategy": round(float(perf["total_return_pct"]), 4),
                        "BH": round(float(bench.get("bh_total_return_pct", float("nan"))), 4),
                    },
                    {
                        "Metric": "CAGR %",
                        "Strategy": round(float(perf["cagr_pct"]), 4),
                        "BH": round(float(bench.get("bh_cagr", float("nan"))), 4),
                    },
                    {
                        "Metric": "Sharpe",
                        "Strategy": round(float(radj["sharpe"]), 4),
                        "BH": round(float(bench.get("bh_sharpe", float("nan"))), 4),
                    },
                    {
                        "Metric": "Max DD %",
                        "Strategy": round(float(risk["max_drawdown_pct"]), 4),
                        "BH": round(float(bench.get("bh_max_dd_pct", float("nan"))), 4),
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    if extra.get("multi_layer_walk_forward"):
        st.caption(
            "Multi-layer walk-forward: strategy = month-end Layer 1→3 picks, equal-weight until next rebalance. "
            "Benchmark = equal-weight daily returns on the full downloaded universe over the same days."
        )
    elif extra.get("sector_rotation_walk_forward"):
        st.caption(
            "Sector rotation: weekly rebalance using sector index momentum/vol rank, then one ML+MA50 pick per top sector, equal-weight. "
            "Benchmark = Nifty spot (^NSEI) on the same trading days."
        )
    else:
        st.caption(
            "Basket-level backtest sums per-ticker equity; round-trip trade stats are not aggregated here. "
            "Run single-ticker mode for full trade analytics."
        )

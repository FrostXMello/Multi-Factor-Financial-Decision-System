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

    tab_ov, tab_perf, tab_risk, tab_time, tab_strat = st.tabs(
        ["Overview", "Performance", "Risk & trades", "Time analysis", "Strategy"]
    )

    with tab_ov:
        st.subheader("Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Symbol", sym)
        c2.metric("Latest signal", str(latest.get("signal", "—")))
        c3.metric("P(up)", _fmt_num(float(latest.get("proba", float("nan"))), 3))
        c4.metric("Last close (test bar)", _fmt_num(last_close, 2))
        st.caption("Test-set accuracy (directional) is a classifier metric, not live trading performance.")
        st.metric("Test accuracy", _fmt_num(float(test_acc), 3))
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

    with tab_perf:
        st.subheader("Performance summary")
        r1 = st.columns(3)
        r1[0].metric("Total return", _fmt_pct(perf["total_return_pct"]))
        r1[1].metric("CAGR", _fmt_pct(perf["cagr_pct"]))
        r1[2].metric("Final value", f"{perf['final_portfolio_value']:,.0f}")
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
                    "Native importances are unavailable for this estimator (e.g. HistGradientBoosting). "
                    "Try Random Forest in research mode for a bar-style importance table."
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
    r1[0].metric("Total return", _fmt_pct(perf["total_return_pct"]))
    r1[1].metric("CAGR", _fmt_pct(perf["cagr_pct"]))
    r1[2].metric("Final value", f"{perf['final_portfolio_value']:,.0f}")
    r1[3].metric("Sharpe", _fmt_num(radj["sharpe"], 2))
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

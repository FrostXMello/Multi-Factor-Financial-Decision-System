"""Streamlit paper-trading terminal. Never submits real broker orders."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from paper_trading.demo import DEMO_ACCOUNT_ID, demo_symbols
from paper_trading.engine import OrderRequest, PaperTradingEngine, PAPER_ACCOUNT_ID
from paper_trading.price_sources import quote_from_market_store
from paper_trading.store import open_paper_store
from ui.theme import freshness_badge


def render_paper_trading_section() -> None:
    st.subheader("Paper Trading")
    st.markdown(
        f"{freshness_badge('PAPER')} Isolated virtual account. This path cannot place a real broker order.",
        unsafe_allow_html=True,
    )
    st.caption(
        "Default mode requires confirmation. Delayed or historical marks need an explicit historical-price toggle. "
        "Demo / Replay uses a separate account and deterministic fixtures."
    )

    store = open_paper_store()
    mode = st.radio("Account mode", ["Paper (normal)", "Demo / Replay"], horizontal=True)
    is_demo = mode.startswith("Demo")
    if is_demo:
        st.markdown(
            f"{freshness_badge('DEMO')} Every fill below uses deterministic demo fixtures. "
            f"Account `{DEMO_ACCOUNT_ID}` never writes into `{PAPER_ACCOUNT_ID}`.",
            unsafe_allow_html=True,
        )
        engine = PaperTradingEngine.demo(store)
    else:
        engine = PaperTradingEngine(
            store,
            account_id=PAPER_ACCOUNT_ID,
            mode="paper",
            price_lookup=quote_from_market_store,
        )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Reset this account", type="secondary"):
            engine.reset()
            st.success("Account reset to initial virtual capital.")
            st.rerun()
    with col_b:
        if is_demo and st.button("Run deterministic demo workflow", type="primary"):
            results = engine.run_demo_workflow()
            filled = sum(1 for r in results if r.accepted)
            st.success(f"Demo workflow finished: {filled}/{len(results)} fills on the demo account.")
            st.rerun()

    snap = engine.portfolio_snapshot()
    _render_overview(snap, is_demo=is_demo)
    _render_positions(snap)
    _render_order_ticket(engine, is_demo=is_demo)
    _render_history(snap)
    _render_analytics(snap)


def _render_overview(snap: dict, *, is_demo: bool) -> None:
    st.markdown("#### Account overview")
    if is_demo:
        st.markdown(freshness_badge("DEMO / REPLAY"), unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Initial capital", f"{snap['initial_capital']:,.2f}")
    c[1].metric("Available cash", f"{snap['cash']:,.2f}")
    c[2].metric("Invested", f"{snap['invested']:,.2f}")
    c[3].metric("Portfolio value", f"{snap['equity']:,.2f}")
    c2 = st.columns(4)
    c2[0].metric("Realized P&L", f"{snap['realized_pnl']:,.2f}")
    c2[1].metric("Unrealized P&L", f"{snap['unrealized_pnl']:,.2f}")
    c2[2].metric("Total return", f"{snap['total_return'] * 100:.2f}%")
    c2[3].metric("Drawdown", f"{snap['drawdown'] * 100:.2f}%")
    st.caption(f"Account `{snap['account_id']}` · mode `{snap['mode']}`")


def _render_positions(snap: dict) -> None:
    st.markdown("#### Positions")
    rows = snap.get("positions") or []
    if not rows:
        st.info("No open positions.")
        return
    frame = pd.DataFrame(rows)
    st.dataframe(frame, width="stretch", hide_index=True)
    st.caption("Mark source and timestamp are shown per row. Missing marks fall back to average entry and are labeled unavailable.")


def _render_order_ticket(engine: PaperTradingEngine, *, is_demo: bool) -> None:
    st.markdown("#### Place paper order")
    symbols = demo_symbols() if is_demo else ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS"]
    with st.form("paper_order_form"):
        c1, c2, c3 = st.columns(3)
        symbol = c1.selectbox("Symbol", symbols)
        side = c2.selectbox("Side", ["BUY", "SELL"])
        quantity = c3.number_input("Quantity", min_value=1.0, value=1.0, step=1.0)
        allow_historical = st.checkbox(
            "Allow historical / delayed mark (clearly labelled fill)",
            value=is_demo,
            disabled=is_demo,
        )
        confirm = st.checkbox("I confirm this paper order", value=False)
        auto_from_signal = st.checkbox(
            "Treat as confirmed model signal auto-execution",
            value=False,
            help="Default research integration requires confirmation. Enable only when you intentionally auto-execute a BUY/SELL signal.",
        )
        preview = engine.estimate(symbol, side, float(quantity), allow_historical=allow_historical or is_demo)
        if preview.get("ok"):
            quote = preview["quote"]
            st.markdown(
                f"Mark {freshness_badge(quote.freshness)} `{quote.price:,.2f}` from `{quote.source}` "
                f"at `{quote.quote_time.isoformat()}` · est. fill `{preview['fill_price']:,.4f}` · "
                f"fees `{preview['fees']:,.2f}` · cash impact `{preview['cash_impact']:,.2f}` · "
                f"slippage `{preview['slippage_bps']} bps`",
                unsafe_allow_html=True,
            )
        else:
            st.warning(preview.get("reason", "Cannot estimate fill."))
        submitted = st.form_submit_button("Submit paper order", type="primary")
    if submitted:
        result = engine.place_order(
            OrderRequest(
                symbol=symbol,
                side=side,
                quantity=float(quantity),
                allow_historical=allow_historical or is_demo,
                confirm=confirm or auto_from_signal,
                signal_ref="ui_manual" if not auto_from_signal else "ui_signal_auto",
            )
        )
        if result.accepted:
            st.success(
                f"Filled {result.order['side']} {result.order['quantity']} {result.order['symbol']} "
                f"@ {result.order['fill_price']:.4f} ({result.order['freshness']} / {result.order['price_source']})"
            )
            st.rerun()
        else:
            st.error(result.rejection_reason or "Order rejected.")

    st.markdown("##### Signal → risk → paper order")
    with st.form("signal_bridge_form"):
        sc1, sc2, sc3 = st.columns(3)
        sig = sc1.selectbox("Model signal", ["HOLD", "BUY", "SELL"])
        sig_symbol = sc2.selectbox("Signal symbol", symbols, key="sig_symbol")
        sig_qty = sc3.number_input("Signal quantity", min_value=1.0, value=1.0, step=1.0, key="sig_qty")
        auto_execute = st.checkbox("Enable paper auto-execution for this signal", value=False)
        allow_hist_sig = st.checkbox("Allow historical / delayed mark for signal fill", value=is_demo, disabled=is_demo)
        sig_submit = st.form_submit_button("Apply signal")
    if sig_submit:
        out = engine.submit_signal(
            signal=sig,
            symbol=sig_symbol,
            quantity=float(sig_qty),
            auto_execute=auto_execute,
            allow_historical=allow_hist_sig or is_demo,
            signal_ref=f"dashboard:{sig}",
        )
        if out is None:
            st.info("HOLD produces no paper order.")
        elif out.accepted:
            st.success(f"Signal {sig} filled on the {engine.mode} account.")
            st.rerun()
        else:
            st.error(out.rejection_reason)


def _render_history(snap: dict) -> None:
    st.markdown("#### Order history")
    orders = snap.get("orders") or []
    if not orders:
        st.caption("No orders yet.")
    else:
        frame = pd.DataFrame(orders)
        keep = [
            c
            for c in [
                "submitted_at",
                "status",
                "symbol",
                "side",
                "quantity",
                "fill_price",
                "fees",
                "freshness",
                "price_source",
                "price_time",
                "rejection_reason",
                "signal_ref",
            ]
            if c in frame.columns
        ]
        st.dataframe(frame[keep], width="stretch", hide_index=True)

    st.markdown("#### Closed trades")
    closed = snap.get("closed_trades") or []
    if not closed:
        st.caption("No closed trades yet.")
    else:
        st.dataframe(pd.DataFrame(closed), width="stretch", hide_index=True)


def _render_analytics(snap: dict) -> None:
    st.markdown("#### Trade analytics")
    a, b, c, d = st.columns(4)
    a.metric("Net P&L", f"{snap['net_pnl']:,.2f}")
    a_win = snap.get("win_rate")
    b.metric("Win rate", "n/a" if a_win is None else f"{a_win * 100:.1f}%")
    pf = snap.get("profit_factor")
    c.metric("Profit factor", "n/a" if pf is None else f"{pf:.2f}")
    d.metric("Max drawdown", f"{snap['max_drawdown'] * 100:.2f}%")
    curve = snap.get("equity_curve") or []
    if curve:
        frame = pd.DataFrame(curve)
        frame["ts"] = pd.to_datetime(frame["ts"])
        st.line_chart(frame.set_index("ts")[["equity"]], width="stretch")
    else:
        st.caption("Equity curve appears after the first fill.")
    st.caption(f"Snapshot time {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

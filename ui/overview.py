"""Top-level overview for an academic demonstration."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from market_data.providers import BrokerLiveProvider, YFinanceHistoryProvider
from market_data.session import nse_cash_status
from market_data.store import open_store
from paper_trading.engine import PaperTradingEngine, PAPER_ACCOUNT_ID
from paper_trading.price_sources import quote_from_market_store
from paper_trading.store import open_paper_store
from ui.theme import freshness_badge


def render_overview() -> None:
    st.markdown(
        """
<div class="mf-banner">
  <h2>Multi-Factor Financial Decision System</h2>
  <p>Research terminal for Indian equities: leakage-aware labels, walk-forward evaluation, persistent market data, and isolated paper trading.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "This is a **research and paper-trading system**, not financial advice and not a live execution platform. "
        "No path in this application submits a real broker order."
    )

    now = datetime.now(timezone.utc)
    market = nse_cash_status(now)
    history = YFinanceHistoryProvider().status()
    broker = BrokerLiveProvider().status()
    mstore = open_store()
    mstats = mstore.stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NSE cash session", market.upper())
    c2.metric("Yahoo research", history.freshness.upper())
    c3.metric("Broker feed", "UNCONFIGURED" if not broker.configured else broker.freshness.upper())
    c4.metric("Stored candles", mstats.get("candles", 0))
    st.markdown(
        f"Market data mode {freshness_badge('HISTORICAL' if mstats.get('candles', 0) else 'UNCONFIGURED')} "
        f"· Broker {freshness_badge('UNCONFIGURED' if not broker.configured else broker.freshness)} "
        f"· Checked `{now.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}`",
        unsafe_allow_html=True,
    )
    if market == "closed":
        st.info(
            "The cash session is closed right now. That is expected outside 09:15–15:30 IST on weekdays. "
            "Use Demo / Replay under Paper Trading for a deterministic closed-market walkthrough."
        )

    st.markdown("#### Paper account summary")
    try:
        engine = PaperTradingEngine(
            open_paper_store(),
            account_id=PAPER_ACCOUNT_ID,
            mode="paper",
            price_lookup=quote_from_market_store,
        )
        snap = engine.portfolio_snapshot()
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Paper cash", f"{snap['cash']:,.2f}")
        p2.metric("Paper equity", f"{snap['equity']:,.2f}")
        p3.metric("Open positions", len(snap["positions"]))
        p4.metric("Paper return", f"{snap['total_return'] * 100:.2f}%")
    except Exception as exc:
        st.warning(f"Paper account unavailable: {exc}")

    st.markdown("#### Latest research output")
    research = st.session_state.get("last_research_summary")
    if research:
        st.write(research)
    else:
        st.caption(
            "No research run is cached in this session yet. Open **Research & Predictions**, run the CORE pipeline, "
            "then return here for a summary."
        )

    st.markdown("#### Where to go next")
    st.markdown(
        """
1. **Market Data** — provider status, SQLite coverage, closed-session collection behavior  
2. **Research & Predictions** — technical model, BUY/SELL/HOLD, backtest  
3. **Backtesting / Reliability** — walk-forward folds and calibration after a research run  
4. **Paper Trading** — normal paper book or Demo / Replay for today's closed market  
5. **Settings / Diagnostics** — database paths, environment checklist, verification status  
"""
    )
    st.markdown(
        '<div class="mf-disclaimer">Academic / project demonstration only. Live NSE-session quote ingestion remains pending verification.</div>',
        unsafe_allow_html=True,
    )


def render_settings_diagnostics() -> None:
    st.subheader("Settings / Diagnostics")
    st.caption("Environment and persistence diagnostics. Secrets are never displayed.")

    import os

    from market_data.store import default_db_path
    from paper_trading.store import default_paper_db_path

    st.markdown("#### Storage")
    st.code(f"MARKET_DATA_DB -> {default_db_path()}\nPAPER_TRADING_DB -> {default_paper_db_path()}")
    mstats = open_store().stats()
    pstats = open_paper_store().stats()
    st.write({"market_data": mstats, "paper_trading": pstats})

    st.markdown("#### Environment checklist")
    keys = [
        "MARKET_DATA_DB",
        "MARKET_DATA_WATCHLIST",
        "BROKER_LIVE_BASE_URL",
        "BROKER_API_KEY",
        "BROKER_ACCESS_TOKEN",
        "BROKER_LIVE_QUOTES_PATH",
        "BROKER_LIVE_WS_URL",
        "PAPER_TRADING_DB",
    ]
    rows = []
    for key in keys:
        raw = os.environ.get(key)
        secret = any(part in key for part in ("KEY", "TOKEN"))
        rows.append(
            {
                "variable": key,
                "set": bool(raw),
                "display": "«configured, value hidden»" if raw and secret else (raw if raw else "empty"),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

    st.markdown("#### Verification status")
    st.success("M3 implementation and offline validation are complete.")
    st.warning("Live NSE-session quote ingestion remains pending verification during the next market session.")
    st.success("M4 paper engine is isolated: no broker order API is called from paper_trading.")
    st.caption("Copy `.env.example` to `.env` on a licensed machine. Do not commit `.env`.")


def render_reliability_landing() -> None:
    st.subheader("Backtesting / Model Reliability")
    st.caption(
        "Walk-forward folds, calibration, and held-out explainability are produced by a research run. "
        "They appear here when the current Streamlit session has cached a reliability payload."
    )
    payload = st.session_state.get("last_reliability")
    if payload:
        from analytics_dashboard import render_model_reliability

        render_model_reliability(payload)
    else:
        st.info(
            "No reliability payload is cached yet. Open **Research & Predictions**, run the CORE technical pipeline "
            "on a single ticker, then reopen this section. The Reliability tab inside the research dashboard remains available after a run."
        )

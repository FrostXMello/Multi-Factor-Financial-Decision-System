"""Streamlit market-data status. This view does not train a model or invent quotes."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from market_data.ingest import IngestService
from market_data.providers import BrokerLiveProvider, YFinanceHistoryProvider
from market_data.quality import quote_is_stale
from market_data.session import nse_cash_status
from market_data.store import open_store
from ui.theme import freshness_badge


def render_market_data_section() -> None:
    st.subheader("Market Data")
    st.caption(
        "Yahoo Finance is the historical research provider and is delayed. "
        "A live broker feed is used only when its credentials are set. "
        "Empty credentials do not create quotes."
    )
    store = open_store()
    history = YFinanceHistoryProvider()
    broker = BrokerLiveProvider()
    now = datetime.now(timezone.utc)
    market = nse_cash_status(now)
    data_label = _data_mode_label(store, broker, market, now)

    st.markdown(
        f"**Data mode** {freshness_badge(data_label)} &nbsp; "
        f"**Last checked** `{now.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}`",
        unsafe_allow_html=True,
    )
    if market == "closed":
        st.info(
            "NSE cash session is closed (weekday 09:15–15:30 IST clock; holidays are not in this calendar). "
            "Collection will not request a quote. Showing stored and historical rows with their real timestamps."
        )

    statuses = [history.status(), broker.status()]
    st.markdown("#### Providers")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Provider": item.provider_id,
                    "Type": item.kind,
                    "Configured": item.configured,
                    "Connected": item.connected,
                    "Freshness": item.freshness,
                    "WebSocket": item.websocket,
                    "Detail": item.detail,
                }
                for item in statuses
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    if not broker.configured:
        st.caption(
            "Broker live feed is **UNCONFIGURED**. Set `BROKER_LIVE_BASE_URL`, `BROKER_API_KEY`, "
            "and `BROKER_ACCESS_TOKEN` in the environment (see `.env.example`). No request is sent until all three are present."
        )

    st.markdown("#### Session")
    c1, c2, c3 = st.columns(3)
    c1.metric("NSE cash session", market.upper())
    c2.metric("Quote indicator", _live_indicator(store, now))
    c3.metric("Schema version", str(store.stats().get("schema_version")))
    st.caption(f"Database file: `{store.stats()['path']}`")

    stats = store.stats()
    st.markdown("#### Database")
    cols = st.columns(4)
    cols[0].metric("Candles", stats["candles"])
    cols[1].metric("Quotes", stats["quotes"])
    cols[2].metric("Quality events", stats["quality_events"])
    cols[3].metric("Provider events", stats["provider_events"])

    st.markdown("#### Historical availability")
    coverage = store.coverage()
    if coverage.empty:
        st.info("No candles are stored yet. A research run writes Yahoo history into this database.")
    else:
        st.dataframe(coverage, width="stretch", hide_index=True)

    watchlist = _watchlist()
    st.markdown("#### Watchlist quotes")
    if not watchlist:
        st.caption("MARKET_DATA_WATCHLIST is empty. Add symbols in the environment to collect quotes.")
    quotes = store.latest_quotes(watchlist or None)
    if quotes.empty:
        st.caption(
            "No quotes are stored. Delayed, cached, and simulated rows are labeled when present. "
            "Nothing here is implied to be live."
        )
        if market == "closed":
            st.caption("Because the session is closed, the collector will not invent a ticker tape.")
    else:
        display = quotes.copy()
        if "quote_time" in display.columns:
            display["quote_time"] = pd.to_datetime(display["quote_time"]).astype(str)
        st.dataframe(display, width="stretch", hide_index=True)
        st.caption("Each row keeps its provider, timestamp, and freshness label from storage.")

    symbol = st.text_input("Inspect stored candles", value=watchlist[0] if watchlist else "RELIANCE.NS")
    if symbol.strip():
        candles = store.read_candles(symbol.strip().upper(), "1d")
        if candles.empty:
            st.caption(f"No daily candles stored for {symbol.strip().upper()}.")
        else:
            st.caption(
                f"Latest stored daily bars for {symbol.strip().upper()}. "
                f"Source is the database (last bar {candles['Date'].iloc[-1]}), not a live print."
            )
            st.dataframe(candles.tail(8), width="stretch", hide_index=True)

    st.markdown("#### Quality diagnostics")
    quality = store.quality_events(30)
    if quality.empty:
        st.caption("No validation failures are recorded.")
    else:
        st.dataframe(quality, width="stretch", hide_index=True)
    events = store.provider_events(20)
    if not events.empty:
        st.markdown("#### Provider messages")
        st.dataframe(events, width="stretch", hide_index=True)
    else:
        st.caption("No provider errors recorded yet.")

    st.markdown("#### Collection")
    if st.button("Collect watchlist once", type="primary"):
        service = IngestService(store, broker, watchlist)
        result = service.collect_once(now)
        if result.error:
            st.error(result.detail)
        else:
            st.info(result.detail)
    st.caption(
        "M3 implementation and offline validation are complete. "
        "Live NSE-session quote ingestion remains pending verification during the next market session."
    )


def _watchlist() -> list[str]:
    raw = os.environ.get("MARKET_DATA_WATCHLIST", "")
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def _live_indicator(store, now: datetime) -> str:
    quotes = store.latest_quotes()
    if quotes.empty:
        return "none stored"
    labels = {str(value) for value in quotes["freshness"].tolist()}
    if labels == {"live"}:
        newest = pd.to_datetime(quotes["quote_time"]).max()
        if quote_is_stale(newest, now, timedelta(seconds=30)):
            return "stale"
        return "live"
    if "simulated" in labels:
        return "simulated"
    return ", ".join(sorted(labels))


def _data_mode_label(store, broker, market: str, now: datetime) -> str:
    if market == "closed" and store.latest_quotes().empty:
        return "HISTORICAL" if store.stats().get("candles", 0) else "UNCONFIGURED" if not broker.configured else "CLOSED"
    indicator = _live_indicator(store, now)
    if indicator == "live":
        return "LIVE"
    if indicator == "none stored":
        if not broker.configured:
            return "UNCONFIGURED"
        return "DELAYED" if store.stats().get("candles", 0) else "UNCONFIGURED"
    if "simulated" in indicator:
        return "DEMO"
    if indicator == "stale":
        return "DELAYED"
    return "DELAYED"

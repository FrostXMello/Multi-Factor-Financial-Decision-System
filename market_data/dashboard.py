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


def render_market_data_section() -> None:
    st.subheader("Market data")
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

    statuses = [history.status(), broker.status()]
    st.markdown("#### Providers")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Provider": item.provider_id,
                    "Kind": item.kind,
                    "Configured": item.configured,
                    "Connected": item.connected,
                    "Freshness": item.freshness,
                    "WebSocket": item.websocket,
                    "Detail": item.detail,
                }
                for item in statuses
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("#### Session")
    c1, c2, c3 = st.columns(3)
    c1.metric("NSE cash session", market)
    c2.metric("Live data", _live_indicator(store, now))
    c3.metric("Database", str(store.stats().get("schema_version")))
    st.caption(f"Database file: {store.stats()['path']}")

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
        st.dataframe(coverage, use_container_width=True, hide_index=True)

    watchlist = _watchlist()
    st.markdown("#### Watchlist quotes")
    if not watchlist:
        st.caption("MARKET_DATA_WATCHLIST is empty. Add symbols in the environment to collect quotes.")
    quotes = store.latest_quotes(watchlist or None)
    if quotes.empty:
        st.caption("No quotes are stored. Delayed, cached, and simulated rows are labeled. Nothing here is implied to be live.")
    else:
        st.dataframe(quotes, use_container_width=True, hide_index=True)

    symbol = st.text_input("Inspect stored candles", value=watchlist[0] if watchlist else "RELIANCE.NS")
    if symbol.strip():
        candles = store.read_candles(symbol.strip().upper(), "1d")
        if candles.empty:
            st.caption(f"No daily candles stored for {symbol.strip().upper()}.")
        else:
            st.caption(f"Latest stored daily bars for {symbol.strip().upper()}. Source is the database, not a live print.")
            st.dataframe(candles.tail(8), use_container_width=True, hide_index=True)

    st.markdown("#### Quality")
    quality = store.quality_events(30)
    if quality.empty:
        st.caption("No validation failures are recorded.")
    else:
        st.dataframe(quality, use_container_width=True, hide_index=True)
    events = store.provider_events(20)
    if not events.empty:
        st.markdown("#### Provider errors")
        st.dataframe(events, use_container_width=True, hide_index=True)

    if st.button("Collect watchlist once"):
        service = IngestService(store, broker, watchlist)
        result = service.collect_once(now)
        if result.error:
            st.error(result.detail)
        else:
            st.info(result.detail)


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

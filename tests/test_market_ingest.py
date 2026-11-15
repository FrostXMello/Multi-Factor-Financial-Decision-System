"""Provider failures, reconnects, and the rule that an unconfigured feed sends nothing."""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.error import URLError

import pytest

from market_data.ingest import IngestService, ReconnectingStream, call_with_reconnect
from market_data.providers import BrokerLiveProvider, ProviderError, ProviderNotConfigured
from market_data.store import MarketStore
from market_data.types import ProviderStatus, Quote

OPEN = datetime(2026, 11, 16, 5, 0, tzinfo=timezone.utc)  # Monday 10:30 IST
CLOSED = datetime(2026, 11, 15, 5, 0, tzinfo=timezone.utc)  # Sunday


class _Conn:
    def __init__(self, messages):
        self._messages = list(messages)
        self.closed = False

    def recv(self):
        if not self._messages:
            return None
        return self._messages.pop(0)

    def close(self):
        self.closed = True


def test_reconnect_retries_then_returns_the_delayed_quote(tmp_path):
    store = MarketStore(tmp_path / "market.sqlite")

    class Flaky:
        provider_id = "broker_live"
        configured = True
        calls = 0

        def fetch_quotes(self, symbols):
            self.calls += 1
            if self.calls < 3:
                raise ProviderError("temporary outage")
            return [
                Quote(
                    symbol=symbols[0],
                    quote_time="2026-11-16T05:00:00+00:00",
                    last_price=1420,
                    provider_id=self.provider_id,
                    freshness="delayed",
                )
            ]

        def status(self):
            return ProviderStatus("broker_live", "live_broker", True, False, "unavailable", "")

    provider = Flaky()
    result = IngestService(store, provider, ["RELIANCE.NS"], attempts=3, sleep=lambda _s: None).collect_once(OPEN)
    assert result.fetched == 1
    assert result.freshness == "delayed"
    assert result.market_status == "open"
    assert provider.calls == 3
    assert len(store.provider_events()) == 2
    assert store.latest_quotes(["RELIANCE.NS"]).iloc[0]["freshness"] == "delayed"


def test_closed_market_and_missing_credentials_do_not_fetch(tmp_path):
    store = MarketStore(tmp_path / "market.sqlite")

    class Guard:
        provider_id = "broker_live"
        configured = True

        def fetch_quotes(self, symbols):
            raise AssertionError("closed session must not request quotes")

    closed = IngestService(store, Guard(), ["RELIANCE.NS"]).collect_once(CLOSED)
    assert closed.fetched == 0
    assert closed.market_status == "closed"
    assert "No quote was requested" in closed.detail

    class Unconfigured:
        provider_id = "broker_live"
        configured = False

        def status(self):
            return ProviderStatus("broker_live", "live_broker", False, False, "unavailable", "credentials missing")

        def fetch_quotes(self, symbols):
            raise AssertionError("unconfigured provider must not be called")

    idle = IngestService(store, Unconfigured(), ["RELIANCE.NS"]).collect_once(OPEN)
    assert idle.fetched == 0
    assert idle.freshness == "unavailable"
    assert store.stats()["quotes"] == 0


def test_stream_reconnects_after_a_dropped_socket():
    calls = {"n": 0}

    def connect():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("socket dropped")
        return _Conn(["simulated-tick"])

    stream = ReconnectingStream(connect, attempts=3, sleep=lambda _s: None)
    assert stream.messages(limit=1) == ["simulated-tick"]
    assert stream.reconnects >= 1
    assert stream.errors


def test_unconfigured_broker_raises_before_any_request():
    provider = BrokerLiveProvider(environ={})
    with pytest.raises(ProviderNotConfigured, match="No request was sent"):
        provider.fetch_quotes(["RELIANCE.NS"])
    assert provider.status().freshness == "unavailable"
    assert provider.status().websocket == "not_configured"


def test_broker_payload_without_freshness_stays_delayed_and_secrets_are_redacted():
    provider = BrokerLiveProvider(
        environ={
            "BROKER_LIVE_BASE_URL": "https://broker.example",
            "BROKER_API_KEY": "super-secret-key",
            "BROKER_ACCESS_TOKEN": "super-secret-token",
        }
    )
    quotes = provider.parse_quote_payload(
        '{"quotes": [{"symbol": "RELIANCE.NS", "price": 10, "time": "2026-11-16T05:00:00+00:00"}]}'
    )
    assert quotes[0].freshness == "delayed"
    assert quotes[0].live_authorized is False

    def opener(request, timeout=10):
        raise URLError(f"network down super-secret-token")

    provider._opener = opener
    with pytest.raises(ProviderError) as exc:
        provider.fetch_quotes(["RELIANCE.NS"])
    assert "super-secret-token" not in str(exc.value)
    assert "[redacted]" in str(exc.value)


def test_call_with_reconnect_stops_on_missing_configuration():
    def _boom():
        raise ProviderNotConfigured("missing")

    with pytest.raises(ProviderNotConfigured):
        call_with_reconnect(_boom, attempts=3, sleep=lambda _s: None)

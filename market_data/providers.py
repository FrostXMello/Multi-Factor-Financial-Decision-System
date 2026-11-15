"""
Historical and live provider adapters.

Yahoo Finance is the research history source. It is delayed research data, not an NSE live feed.
The broker adapter speaks HTTP only when credentials are present. It does not invent quotes.
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from market_data.types import ProviderStatus, Quote


class ProviderError(RuntimeError):
    pass


class ProviderNotConfigured(ProviderError):
    pass


def _redact(text: str, secrets: list[str]) -> str:
    cleaned = text
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "[redacted]")
    return cleaned


class YFinanceHistoryProvider:
    """Existing research downloader. Bars from this provider are not live."""

    provider_id = "yfinance"
    kind = "historical_research"
    freshness = "delayed"

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider_id=self.provider_id,
            kind=self.kind,
            configured=True,
            connected=True,
            freshness="delayed",
            detail="Yahoo Finance historical research feed. These bars are not a live NSE quote.",
            websocket="not_applicable",
        )


class BrokerLiveProvider:
    """
    Credential-gated adapter for an official broker quote endpoint.

    Expected JSON when a request is made:
    {"quotes": [{"symbol", "price", "time", "bid", "ask", "volume", "freshness"}]}

    Missing freshness is stored as delayed. This adapter never upgrades a payload to live
    unless the payload itself says live and credentials were used for that request.
    """

    provider_id = "broker_live"
    kind = "live_broker"

    def __init__(self, environ: dict[str, str] | None = None, opener=None):
        env = environ if environ is not None else os.environ
        self.base_url = str(env.get("BROKER_LIVE_BASE_URL", "")).strip().rstrip("/")
        self.api_key = str(env.get("BROKER_API_KEY", "")).strip()
        self.access_token = str(env.get("BROKER_ACCESS_TOKEN", "")).strip()
        self.quotes_path = str(env.get("BROKER_LIVE_QUOTES_PATH", "/quotes")).strip() or "/quotes"
        self.ws_url = str(env.get("BROKER_LIVE_WS_URL", "")).strip()
        self._opener = opener

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.access_token)

    def status(self) -> ProviderStatus:
        if not self.configured:
            return ProviderStatus(
                provider_id=self.provider_id,
                kind=self.kind,
                configured=False,
                connected=False,
                freshness="unavailable",
                detail=(
                    "Live broker credentials are not set. "
                    "Set BROKER_LIVE_BASE_URL, BROKER_API_KEY, and BROKER_ACCESS_TOKEN. "
                    "No quote request was sent."
                ),
                websocket="not_configured",
            )
        socket = "configured_but_not_bundled" if self.ws_url else "not_configured"
        return ProviderStatus(
            provider_id=self.provider_id,
            kind=self.kind,
            configured=True,
            connected=False,
            freshness="unavailable",
            detail="Credentials are present. No live session has been opened by this process.",
            websocket=socket,
        )

    def fetch_quotes(self, symbols: list[str]) -> list[Quote]:
        if not self.configured:
            raise ProviderNotConfigured(
                "BROKER_LIVE_BASE_URL, BROKER_API_KEY, and BROKER_ACCESS_TOKEN are required. No request was sent."
            )
        clean = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
        if not clean:
            return []
        if not self.quotes_path.startswith("/"):
            path = "/" + self.quotes_path
        else:
            path = self.quotes_path
        url = f"{self.base_url}{path}?{urlencode({'symbols': ','.join(clean)})}"
        request = Request(
            url,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "X-Api-Key": self.api_key,
                "Accept": "application/json",
            },
        )
        try:
            open_fn = self._opener or urlopen
            with open_fn(request, timeout=10) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp is not None else ""
            message = _redact(f"Broker quote request failed with HTTP {exc.code}. {body[:300]}", self._secrets())
            raise ProviderError(message) from exc
        except URLError as exc:
            raise ProviderError(_redact(f"Broker quote request failed: {exc.reason}", self._secrets())) from exc
        return self.parse_quote_payload(payload)

    def parse_quote_payload(self, payload: str) -> list[Quote]:
        try:
            body = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProviderError("Broker quote response was not JSON.") from exc
        rows = body.get("quotes", []) if isinstance(body, dict) else None
        if not isinstance(rows, list):
            raise ProviderError("Broker quote response has no quotes list.")
        quotes: list[Quote] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            declared = str(row.get("freshness", "")).strip().lower()
            if declared == "live":
                freshness = "live"
                authorized = True
            elif declared in {"delayed", "cached", "simulated"}:
                freshness = declared
                authorized = False
            else:
                freshness = "delayed"
                authorized = False
            quotes.append(
                Quote(
                    symbol=str(row.get("symbol", "")).strip().upper(),
                    quote_time=str(row.get("time", "")),
                    last_price=float(row["price"]),
                    provider_id=self.provider_id,
                    freshness=freshness,
                    live_authorized=authorized,
                    bid=_optional_float(row.get("bid")),
                    ask=_optional_float(row.get("ask")),
                    volume=_optional_float(row.get("volume")),
                )
            )
        return quotes

    def _secrets(self) -> list[str]:
        return [self.api_key, self.access_token]


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)

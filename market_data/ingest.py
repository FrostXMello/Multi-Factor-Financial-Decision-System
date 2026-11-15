"""
Watchlist collection.

Outside the NSE cash session, and whenever the live provider is not configured,
this service does not invent a quote. Reconnects are explicit and bounded.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

from market_data.providers import ProviderError, ProviderNotConfigured
from market_data.session import nse_cash_status
from market_data.store import MarketStore
from market_data.types import CollectResult

logger = logging.getLogger("market_data.ingest")


class RateLimiter:
    def __init__(self, min_interval_s: float, *, clock: Callable[[], float] | None = None, sleep: Callable[[float], None] | None = None):
        self.min_interval_s = float(min_interval_s)
        self._clock = clock or _monotonic
        self._sleep = sleep or _default_sleep
        self._last = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            delay = self.min_interval_s - (now - self._last)
            if delay > 0:
                self._sleep(delay)
                now = self._clock()
        self._last = now


def call_with_reconnect(
    fn: Callable[[], object],
    *,
    attempts: int = 3,
    sleep: Callable[[float], None] | None = None,
    on_error: Callable[[Exception, int], None] | None = None,
):
    """Call `fn`, backing off after ProviderError. The last failure is raised."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    sleeper = sleep or _default_sleep
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except ProviderNotConfigured:
            raise
        except ProviderError as exc:
            last = exc
            if on_error is not None:
                on_error(exc, attempt)
            if attempt + 1 >= attempts:
                raise
            sleeper(min(30.0, 2.0 ** attempt))
    if last is not None:
        raise last
    raise ProviderError("Reconnect ended without a result.")


class ReconnectingStream:
    """
    Read a transport until it drops, then connect again.

    `connect` returns an object with recv() and close(). recv() returns a message,
    or None when the server closes. This does not open a socket by itself.
    """

    def __init__(
        self,
        connect: Callable[[], object],
        *,
        attempts: int = 3,
        sleep: Callable[[float], None] | None = None,
    ):
        self._connect = connect
        self.attempts = int(attempts)
        self._sleep = sleep or _default_sleep
        self.errors: list[str] = []
        self.reconnects = 0

    def messages(self, limit: int) -> list:
        collected: list = []
        failures = 0
        drops = 0
        while len(collected) < limit:
            try:
                conn = self._connect()
            except Exception as exc:
                failures += 1
                self.reconnects += 1
                self.errors.append(str(exc))
                if failures >= self.attempts:
                    raise
                self._sleep(min(30.0, 2.0 ** failures))
                continue
            failures = 0
            try:
                while len(collected) < limit:
                    message = conn.recv()
                    if message is None:
                        break
                    collected.append(message)
            except Exception as exc:
                failures += 1
                self.reconnects += 1
                self.errors.append(str(exc))
                if failures >= self.attempts:
                    raise
                self._sleep(min(30.0, 2.0 ** failures))
            finally:
                if hasattr(conn, "close"):
                    conn.close()
            if len(collected) >= limit:
                break
            drops += 1
            self.reconnects += 1
            if drops >= self.attempts:
                break
        return collected


class IngestService:
    def __init__(
        self,
        store: MarketStore,
        provider,
        watchlist: list[str],
        *,
        min_interval_s: float = 1.0,
        attempts: int = 3,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
        limiter_clock: Callable[[], float] | None = None,
    ):
        self.store = store
        self.provider = provider
        self.watchlist = [str(symbol).strip().upper() for symbol in watchlist if str(symbol).strip()]
        self.attempts = int(attempts)
        self._clock = clock or _utc_now
        self._sleep = sleep or _default_sleep
        self.limiter = RateLimiter(min_interval_s, clock=limiter_clock, sleep=self._sleep)
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def collect_once(self, now: datetime | None = None) -> CollectResult:
        moment = now or self._clock()
        status = nse_cash_status(moment)
        provider_id = getattr(self.provider, "provider_id", "unknown")
        if status != "open":
            detail = "NSE cash session is closed. No quote was requested."
            logger.info(detail)
            return CollectResult(market_status=status, fetched=0, freshness="unavailable", detail=detail)
        if not getattr(self.provider, "configured", False):
            detail = self.provider.status().detail
            self.store.record_provider_event(provider_id, "info", detail)
            return CollectResult(market_status=status, fetched=0, freshness="unavailable", detail=detail)
        if self._stopped:
            return CollectResult(
                market_status=status,
                fetched=0,
                freshness="unavailable",
                detail="Collector is stopped.",
            )
        self.limiter.wait()

        def _pull():
            return self.provider.fetch_quotes(self.watchlist)

        def _on_error(exc: Exception, attempt: int) -> None:
            self.store.record_provider_event(provider_id, "error", f"attempt {attempt + 1}: {exc}")
            logger.warning("provider failure on attempt %s: %s", attempt + 1, exc)

        try:
            quotes = call_with_reconnect(_pull, attempts=self.attempts, sleep=self._sleep, on_error=_on_error)
        except ProviderNotConfigured as exc:
            self.store.record_provider_event(provider_id, "error", str(exc))
            return CollectResult(market_status=status, fetched=0, freshness="unavailable", detail=str(exc), error=str(exc))
        except ProviderError as exc:
            self.store.record_provider_event(provider_id, "error", str(exc))
            return CollectResult(market_status=status, fetched=0, freshness="unavailable", detail=str(exc), error=str(exc))
        self.store.upsert_quotes(list(quotes))
        freshness = quotes[0].freshness if quotes else "unavailable"
        return CollectResult(
            market_status=status,
            fetched=len(quotes),
            freshness=freshness,
            detail=f"Stored {len(quotes)} quote(s) labeled {freshness}.",
        )


def _utc_now() -> datetime:
    from datetime import timezone

    return datetime.now(timezone.utc)


def _monotonic() -> float:
    import time

    return time.monotonic()


def _default_sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)

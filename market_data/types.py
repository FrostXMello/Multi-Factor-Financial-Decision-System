"""Provider-neutral market data records."""

from __future__ import annotations

from dataclasses import dataclass


FRESHNESS_VALUES = ("live", "delayed", "cached", "simulated", "unavailable")


@dataclass(frozen=True)
class Instrument:
    symbol: str
    exchange: str
    name: str = ""
    asset_class: str = "equity"
    currency: str = "INR"
    lot_size: int | None = None
    source: str = ""


@dataclass(frozen=True)
class Quote:
    symbol: str
    quote_time: str
    last_price: float
    provider_id: str
    freshness: str
    live_authorized: bool = False
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None

    def __post_init__(self) -> None:
        if self.freshness not in FRESHNESS_VALUES:
            raise ValueError(f"freshness must be one of {FRESHNESS_VALUES}")
        if self.freshness == "live" and not self.live_authorized:
            raise ValueError("A quote cannot be labeled live unless the provider authorized that label.")


@dataclass(frozen=True)
class ProviderStatus:
    provider_id: str
    kind: str
    configured: bool
    connected: bool
    freshness: str
    detail: str
    websocket: str = "not_used"


@dataclass(frozen=True)
class CollectResult:
    market_status: str
    fetched: int
    freshness: str
    detail: str
    error: str = ""

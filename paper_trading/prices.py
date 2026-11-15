"""Price quotes for paper fills. Freshness labels are never upgraded to live."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

FRESHNESS_VALUES = ("live", "delayed", "cached", "historical", "demo", "unavailable")


@dataclass(frozen=True)
class PriceQuote:
    symbol: str
    price: float
    source: str
    quote_time: datetime
    freshness: str
    live_authorized: bool = False

    def __post_init__(self) -> None:
        if float(self.price) <= 0:
            raise ValueError("price must be positive")
        label = str(self.freshness).strip().lower()
        if label not in FRESHNESS_VALUES:
            raise ValueError(f"unsupported freshness: {self.freshness}")
        if label == "live" and not self.live_authorized:
            raise ValueError("live freshness requires live_authorized=True")
        object.__setattr__(self, "freshness", label)
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())

    def is_stale(self, now: datetime | None = None, *, max_age: timedelta = timedelta(minutes=15)) -> bool:
        moment = now or datetime.now(timezone.utc)
        quote_time = self.quote_time
        if quote_time.tzinfo is None:
            quote_time = quote_time.replace(tzinfo=timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return (moment - quote_time) > max_age

    def is_executable_as_market(self, *, allow_historical: bool = False, allow_demo: bool = False) -> bool:
        if self.freshness == "unavailable":
            return False
        if self.freshness == "demo":
            return bool(allow_demo)
        if self.freshness in {"historical", "delayed", "cached"}:
            return bool(allow_historical) or self.freshness in {"delayed", "cached"}
        if self.freshness == "live":
            return True
        return False

"""NSE cash-session clock. Holidays are not a full exchange calendar."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
CASH_OPEN = time(9, 15)
CASH_CLOSE = time(15, 30)


def nse_cash_status(now: datetime) -> str:
    """
    Return 'open' or 'closed' for the NSE cash session.

    Weekend days are closed. Exchange holidays are not listed here, so a holiday
    inside the clock window is reported as open and the collector may find no print.
    """
    if now.tzinfo is None:
        aware = now.replace(tzinfo=ZoneInfo("UTC"))
    else:
        aware = now
    local = aware.astimezone(IST)
    if local.weekday() >= 5:
        return "closed"
    if CASH_OPEN <= local.time() <= CASH_CLOSE:
        return "open"
    return "closed"

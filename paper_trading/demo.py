"""Deterministic demo/replay fixtures. Never labeled live."""

from __future__ import annotations

from datetime import datetime, timezone

from paper_trading.prices import PriceQuote

DEMO_ACCOUNT_ID = "demo:academic"
DEMO_INITIAL_CAPITAL = 100_000.0

# Fixed replay marks for a closed-market demonstration. Prices do not change within a session.
_DEMO_MARKS = {
    "RELIANCE.NS": 2_850.00,
    "TCS.NS": 3_920.00,
    "HDFCBANK.NS": 1_640.00,
    "INFY.NS": 1_510.00,
    "ITC.NS": 460.00,
}

DEMO_QUOTE_TIME = datetime(2024, 6, 14, 10, 0, tzinfo=timezone.utc)


def demo_symbols() -> list[str]:
    return sorted(_DEMO_MARKS)


def demo_quote(symbol: str) -> PriceQuote:
    key = str(symbol).strip().upper()
    if key not in _DEMO_MARKS:
        raise KeyError(f"No deterministic demo mark for {key}. Known: {', '.join(demo_symbols())}")
    return PriceQuote(
        symbol=key,
        price=float(_DEMO_MARKS[key]),
        source="demo_fixture",
        quote_time=DEMO_QUOTE_TIME,
        freshness="demo",
        live_authorized=False,
    )


def demo_workflow_steps() -> list[dict]:
    """Ordered steps used by the Streamlit demo button. Fully deterministic."""
    # Sizes stay under default PaperRiskConfig max_order_value (25_000).
    return [
        {"side": "BUY", "symbol": "RELIANCE.NS", "quantity": 5, "client_order_id": "demo-step-1-buy-reliance"},
        {"side": "BUY", "symbol": "TCS.NS", "quantity": 3, "client_order_id": "demo-step-2-buy-tcs"},
        {"side": "SELL", "symbol": "RELIANCE.NS", "quantity": 2, "client_order_id": "demo-step-3-sell-reliance"},
    ]

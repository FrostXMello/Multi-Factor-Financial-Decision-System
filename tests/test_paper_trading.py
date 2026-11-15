"""Offline paper-trading tests. All prices are labelled fixtures, never live."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from paper_trading.demo import DEMO_ACCOUNT_ID, DEMO_INITIAL_CAPITAL, demo_quote
from paper_trading.engine import OrderRequest, PaperTradingEngine, PAPER_ACCOUNT_ID
from paper_trading.prices import PriceQuote
from paper_trading.risk import PaperRiskConfig
from paper_trading.store import PaperStore
from transaction_costs import IndiaEquityCostConfig


FIXED_NOW = datetime(2024, 6, 14, 10, 30, tzinfo=timezone.utc)


def _quote(symbol: str, price: float, *, freshness: str = "historical") -> PriceQuote:
    return PriceQuote(
        symbol=symbol,
        price=price,
        source="test_fixture",
        quote_time=FIXED_NOW,
        freshness=freshness,
        live_authorized=False,
    )


def _paper(tmp_path, marks: dict[str, float] | None = None, **kwargs) -> PaperTradingEngine:
    store = PaperStore(tmp_path / "paper.sqlite")
    table = marks or {"RELIANCE.NS": 100.0, "TCS.NS": 200.0}

    def lookup(symbol: str) -> PriceQuote:
        return _quote(symbol, table[symbol])

    return PaperTradingEngine(
        store,
        account_id=PAPER_ACCOUNT_ID,
        mode="paper",
        initial_capital=kwargs.pop("initial_capital", 100_000.0),
        cost_config=kwargs.pop("cost_config", IndiaEquityCostConfig.equity_delivery(slippage_bps=10.0)),
        risk_config=kwargs.pop("risk_config", PaperRiskConfig()),
        clock=lambda: FIXED_NOW,
        price_lookup=lookup,
        **kwargs,
    )


def test_account_persists_across_reopen(tmp_path):
    path = tmp_path / "paper.sqlite"
    first = PaperTradingEngine(
        PaperStore(path),
        account_id=PAPER_ACCOUNT_ID,
        mode="paper",
        initial_capital=50_000.0,
        price_lookup=lambda s: _quote(s, 100.0),
        clock=lambda: FIXED_NOW,
    )
    first.place_order(
        OrderRequest(symbol="RELIANCE.NS", side="BUY", quantity=10, allow_historical=True, confirm=True, client_order_id="p1")
    )
    cash_after = first.account()["cash"]
    second = PaperTradingEngine(
        PaperStore(path),
        account_id=PAPER_ACCOUNT_ID,
        mode="paper",
        price_lookup=lambda s: _quote(s, 100.0),
        clock=lambda: FIXED_NOW,
    )
    assert second.account()["cash"] == pytest.approx(cash_after)
    assert second.store.list_positions(PAPER_ACCOUNT_ID)[0]["quantity"] == pytest.approx(10.0)
    assert second.store.stats()["schema_version"] == 1


def test_buy_sell_avg_price_fees_and_realized_pnl(tmp_path):
    engine = _paper(tmp_path, {"RELIANCE.NS": 100.0})
    buy = engine.place_order(
        OrderRequest(symbol="RELIANCE.NS", side="BUY", quantity=10, allow_historical=True, confirm=True, client_order_id="b1")
    )
    assert buy.accepted
    assert buy.order["fees"] > 0
    assert buy.order["fill_price"] > 100.0  # slippage on buy
    assert buy.order["freshness"] == "historical"
    pos = engine.store.get_position(PAPER_ACCOUNT_ID, "RELIANCE.NS")
    assert pos["quantity"] == pytest.approx(10.0)
    avg = float(pos["avg_price"])

    # Mark moves up for the sell estimate path.
    engine._price_lookup = lambda s: _quote(s, 110.0)
    sell = engine.place_order(
        OrderRequest(symbol="RELIANCE.NS", side="SELL", quantity=10, allow_historical=True, confirm=True, client_order_id="s1")
    )
    assert sell.accepted
    assert sell.order["fill_price"] < 110.0  # slippage on sell
    snap = engine.portfolio_snapshot()
    assert snap["positions"] == []
    assert snap["realized_pnl"] != 0.0
    assert abs(snap["realized_pnl"] - ((sell.order["fill_price"] - avg) * 10 - sell.order["fees"])) < 1e-6


def test_insufficient_cash_and_holdings_and_no_short(tmp_path):
    engine = _paper(tmp_path, {"RELIANCE.NS": 100.0}, initial_capital=1_000.0)
    rejected = engine.place_order(
        OrderRequest(symbol="RELIANCE.NS", side="BUY", quantity=100, allow_historical=True, confirm=True, client_order_id="big")
    )
    assert not rejected.accepted
    assert "Insufficient cash" in rejected.rejection_reason

    engine.place_order(
        OrderRequest(symbol="RELIANCE.NS", side="BUY", quantity=1, allow_historical=True, confirm=True, client_order_id="small")
    )
    short = engine.place_order(
        OrderRequest(symbol="RELIANCE.NS", side="SELL", quantity=5, allow_historical=True, confirm=True, client_order_id="short")
    )
    assert not short.accepted
    assert "Short selling is disabled" in short.rejection_reason


def test_position_exposure_and_daily_loss_limits(tmp_path):
    engine = _paper(
        tmp_path,
        {"RELIANCE.NS": 100.0},
        risk_config=PaperRiskConfig(max_order_value=500.0, max_position_value=500.0, max_daily_loss=1.0),
    )
    over = engine.place_order(
        OrderRequest(symbol="RELIANCE.NS", side="BUY", quantity=10, allow_historical=True, confirm=True, client_order_id="over")
    )
    assert not over.accepted
    assert "max order value" in over.rejection_reason.lower() or "Order value" in over.rejection_reason

    # Seed a closed trade that already breached daily loss.
    engine.store.insert_closed_trade(
        {
            "account_id": PAPER_ACCOUNT_ID,
            "symbol": "RELIANCE.NS",
            "quantity": 1,
            "entry_price": 100,
            "exit_price": 90,
            "entry_ts": FIXED_NOW.isoformat(),
            "exit_ts": FIXED_NOW.isoformat(),
            "realized_pnl": -50.0,
            "fees": 0.0,
        }
    )
    blocked = engine.place_order(
        OrderRequest(symbol="RELIANCE.NS", side="BUY", quantity=1, allow_historical=True, confirm=True, client_order_id="after-loss")
    )
    assert not blocked.accepted
    assert "Daily loss limit" in blocked.rejection_reason


def test_duplicate_and_unconfirmed_and_missing_price(tmp_path):
    engine = _paper(tmp_path, {"RELIANCE.NS": 100.0})
    first = engine.place_order(
        OrderRequest(symbol="RELIANCE.NS", side="BUY", quantity=1, allow_historical=True, confirm=True, client_order_id="dup")
    )
    again = engine.place_order(
        OrderRequest(symbol="RELIANCE.NS", side="BUY", quantity=1, allow_historical=True, confirm=True, client_order_id="dup")
    )
    assert first.accepted
    assert again.order["order_id"] == first.order["order_id"]

    unconfirmed = engine.place_order(
        OrderRequest(symbol="RELIANCE.NS", side="BUY", quantity=2, allow_historical=True, confirm=False, client_order_id="need-confirm")
    )
    assert not unconfirmed.accepted
    assert "confirmation" in unconfirmed.rejection_reason.lower()

    def missing(_s):
        raise KeyError(_s)

    engine._price_lookup = missing
    no_px = engine.place_order(
        OrderRequest(symbol="RELIANCE.NS", side="BUY", quantity=3, allow_historical=True, confirm=True, client_order_id="nopx")
    )
    assert not no_px.accepted
    assert "No price available" in no_px.rejection_reason


def test_stale_historical_requires_explicit_mode(tmp_path):
    engine = _paper(tmp_path, {"RELIANCE.NS": 100.0})
    blocked = engine.place_order(
        OrderRequest(symbol="RELIANCE.NS", side="BUY", quantity=1, allow_historical=False, confirm=True, client_order_id="hist-block")
    )
    assert not blocked.accepted
    assert "historical-price mode" in blocked.rejection_reason


def test_demo_isolation_and_deterministic_workflow(tmp_path):
    store = PaperStore(tmp_path / "paper.sqlite")
    paper = PaperTradingEngine(
        store,
        account_id=PAPER_ACCOUNT_ID,
        mode="paper",
        initial_capital=100_000.0,
        price_lookup=lambda s: _quote(s, 100.0),
        clock=lambda: FIXED_NOW,
    )
    demo = PaperTradingEngine.demo(store, clock=lambda: FIXED_NOW)
    results = demo.run_demo_workflow()
    assert all(r.accepted for r in results)
    assert all(r.order["freshness"] == "demo" for r in results)
    assert demo.account_id == DEMO_ACCOUNT_ID
    assert paper.account()["cash"] == pytest.approx(100_000.0)
    assert paper.store.list_positions(PAPER_ACCOUNT_ID) == []
    assert demo.account()["cash"] < DEMO_INITIAL_CAPITAL
    assert demo.store.list_positions(DEMO_ACCOUNT_ID)

    # Demo marks cannot fill the paper account.
    paper._price_lookup = demo_quote
    rejected = paper.place_order(
        OrderRequest(symbol="RELIANCE.NS", side="BUY", quantity=1, allow_historical=True, confirm=True, client_order_id="cross")
    )
    assert not rejected.accepted
    assert "Demo prices cannot fill" in rejected.rejection_reason


def test_signal_bridge_defaults_to_confirmation(tmp_path):
    engine = _paper(tmp_path, {"RELIANCE.NS": 100.0})
    hold = engine.submit_signal(signal="HOLD", symbol="RELIANCE.NS", quantity=1, auto_execute=True, allow_historical=True)
    assert hold is None
    manual = engine.submit_signal(signal="BUY", symbol="RELIANCE.NS", quantity=1, auto_execute=False, allow_historical=True)
    assert manual is not None and not manual.accepted
    assert "confirmation" in manual.rejection_reason.lower()
    auto = engine.submit_signal(
        signal="BUY",
        symbol="RELIANCE.NS",
        quantity=1,
        auto_execute=True,
        allow_historical=True,
        client_order_id="sig-auto",
    )
    assert auto is not None and auto.accepted
    assert auto.order["signal_ref"].startswith("signal:")


def test_live_label_without_authorization_is_rejected():
    with pytest.raises(ValueError):
        PriceQuote(
            symbol="RELIANCE.NS",
            price=10.0,
            source="x",
            quote_time=FIXED_NOW,
            freshness="live",
            live_authorized=False,
        )

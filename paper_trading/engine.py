"""
Paper execution engine.

This module never opens a broker session and never places a real order.
Fills use an explicit PriceQuote with a visible source and freshness label.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from paper_trading.demo import DEMO_ACCOUNT_ID, DEMO_INITIAL_CAPITAL, demo_quote, demo_workflow_steps
from paper_trading.prices import PriceQuote
from paper_trading.risk import PaperRiskConfig, validate_order_risk
from paper_trading.store import PaperStore, open_paper_store
from transaction_costs import IndiaEquityCostConfig, apply_slippage, buy_charges, sell_charges

PAPER_ACCOUNT_ID = "paper:default"


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str
    quantity: float
    client_order_id: str | None = None
    signal_ref: str = ""
    allow_historical: bool = False
    confirm: bool = True


@dataclass(frozen=True)
class OrderResult:
    order: dict
    accepted: bool

    @property
    def status(self) -> str:
        return str(self.order.get("status", ""))

    @property
    def rejection_reason(self) -> str:
        return str(self.order.get("rejection_reason", ""))


class PaperTradingEngine:
    def __init__(
        self,
        store: PaperStore | None = None,
        *,
        account_id: str = PAPER_ACCOUNT_ID,
        mode: str = "paper",
        initial_capital: float = 100_000.0,
        cost_config: IndiaEquityCostConfig | None = None,
        risk_config: PaperRiskConfig | None = None,
        clock: Callable[[], datetime] | None = None,
        price_lookup: Callable[[str], PriceQuote | None] | None = None,
    ):
        self.store = store or open_paper_store()
        self.mode = str(mode).strip().lower()
        if self.mode not in {"paper", "demo"}:
            raise ValueError("mode must be paper or demo")
        self.account_id = account_id
        if self.mode == "demo" and not account_id.startswith("demo:"):
            raise ValueError("demo mode requires a demo: account_id")
        if self.mode == "paper" and account_id.startswith("demo:"):
            raise ValueError("paper mode cannot use a demo: account_id")
        self.cost_config = cost_config or IndiaEquityCostConfig.equity_delivery()
        self.risk_config = risk_config or PaperRiskConfig()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._price_lookup = price_lookup
        label = "Demo / Replay" if self.mode == "demo" else "Paper"
        self.store.ensure_account(
            self.account_id,
            mode=self.mode,
            label=label,
            initial_capital=float(initial_capital),
        )

    @classmethod
    def demo(cls, store: PaperStore | None = None, **kwargs) -> "PaperTradingEngine":
        return cls(
            store,
            account_id=kwargs.pop("account_id", DEMO_ACCOUNT_ID),
            mode="demo",
            initial_capital=kwargs.pop("initial_capital", DEMO_INITIAL_CAPITAL),
            price_lookup=kwargs.pop("price_lookup", demo_quote),
            **kwargs,
        )

    def reset(self, initial_capital: float | None = None) -> dict:
        return self.store.reset_account(self.account_id, initial_capital=initial_capital)

    def account(self) -> dict:
        row = self.store.get_account(self.account_id)
        if row is None:
            raise KeyError(self.account_id)
        return row

    def resolve_price(self, symbol: str) -> PriceQuote | None:
        symbol_u = str(symbol).strip().upper()
        if self._price_lookup is not None:
            try:
                return self._price_lookup(symbol_u)
            except KeyError:
                return None
        return None

    def estimate(self, symbol: str, side: str, quantity: float, *, allow_historical: bool = False) -> dict:
        quote = self.resolve_price(symbol)
        if quote is None:
            return {"ok": False, "reason": f"No price available for {symbol}."}
        if quote.freshness == "unavailable":
            return {"ok": False, "reason": "Price freshness is unavailable.", "quote": quote}
        if quote.freshness == "demo" and self.mode != "demo":
            return {"ok": False, "reason": "Demo prices cannot fill the normal paper account.", "quote": quote}
        if self.mode == "paper" and quote.freshness in {"historical", "delayed", "cached"} and not allow_historical:
            return {
                "ok": False,
                "reason": (
                    f"Latest mark is {quote.freshness} from {quote.source} at {quote.quote_time.isoformat()}. "
                    "Enable historical-price mode to paper-fill at this mark, or wait for a fresher quote."
                ),
                "quote": quote,
            }
        if self.mode == "demo" and quote.freshness not in {"demo", "historical", "delayed", "cached"}:
            # Demo account accepts only labelled non-live marks from the fixture path.
            if quote.freshness != "live":
                return {"ok": False, "reason": f"Unsupported demo freshness: {quote.freshness}", "quote": quote}
        fill_px = apply_slippage(quote.price, side, self.cost_config)
        notional = float(quantity) * fill_px
        fees = buy_charges(notional, self.cost_config) if str(side).upper() == "BUY" else sell_charges(notional, self.cost_config)
        cash_impact = -(notional + fees) if str(side).upper() == "BUY" else (notional - fees)
        return {
            "ok": True,
            "quote": quote,
            "fill_price": fill_px,
            "fees": fees,
            "notional": notional,
            "cash_impact": cash_impact,
            "slippage_bps": self.cost_config.slippage_bps,
        }

    def place_order(self, request: OrderRequest) -> OrderResult:
        now = self._clock()
        side = str(request.side).upper().strip()
        symbol = str(request.symbol).strip().upper()
        qty = float(request.quantity)
        client_id = request.client_order_id or f"ord-{uuid.uuid4().hex[:12]}"
        existing = self.store.find_order_by_client_id(self.account_id, client_id)
        if existing is not None:
            return OrderResult(existing, accepted=existing.get("status") == "filled")

        # Duplicate protection: identical open intent inside the risk window.
        for prior in self.store.recent_orders(self.account_id, limit=20):
            if prior["status"] != "filled":
                continue
            if prior["symbol"] != symbol or prior["side"] != side:
                continue
            if abs(float(prior["quantity"]) - qty) > 1e-9:
                continue
            submitted = datetime.fromisoformat(prior["submitted_at"])
            if submitted.tzinfo is None:
                submitted = submitted.replace(tzinfo=timezone.utc)
            age = abs((now - submitted).total_seconds())
            if age <= self.risk_config.duplicate_window_seconds:
                rejected = self._reject(
                    client_id=client_id,
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    now=now,
                    reason="Duplicate order within the protection window.",
                    signal_ref=request.signal_ref,
                )
                return OrderResult(rejected, accepted=False)

        if not request.confirm:
            rejected = self._reject(
                client_id=client_id,
                symbol=symbol,
                side=side,
                quantity=qty,
                now=now,
                reason="Order requires explicit confirmation.",
                signal_ref=request.signal_ref,
            )
            return OrderResult(rejected, accepted=False)

        estimate = self.estimate(symbol, side, qty, allow_historical=request.allow_historical or self.mode == "demo")
        if not estimate.get("ok"):
            rejected = self._reject(
                client_id=client_id,
                symbol=symbol,
                side=side,
                quantity=qty,
                now=now,
                reason=str(estimate.get("reason", "Price unavailable.")),
                signal_ref=request.signal_ref,
                quote=estimate.get("quote"),
            )
            return OrderResult(rejected, accepted=False)

        quote: PriceQuote = estimate["quote"]
        fill_px = float(estimate["fill_price"])
        fees = float(estimate["fees"])
        account = self.account()
        position = self.store.get_position(self.account_id, symbol) or {"quantity": 0.0, "avg_price": 0.0}
        marks = self._position_marks()
        invested = sum(marks.values())
        equity = float(account["cash"]) + invested
        decision = validate_order_risk(
            side=side,
            quantity=qty,
            fill_price=fill_px,
            fees=fees,
            cash=float(account["cash"]),
            equity=equity,
            position_qty=float(position["quantity"]),
            position_market_value=float(position["quantity"]) * float(marks.get(symbol, fill_px)),
            invested_value=invested,
            realized_pnl_today=self.store.realized_pnl_today(self.account_id, now.date().isoformat()),
            cfg=self.risk_config,
        )
        if not decision.allowed:
            rejected = self._reject(
                client_id=client_id,
                symbol=symbol,
                side=side,
                quantity=qty,
                now=now,
                reason=decision.reason,
                signal_ref=request.signal_ref,
                quote=quote,
            )
            return OrderResult(rejected, accepted=False)

        filled = self._fill(
            client_id=client_id,
            symbol=symbol,
            side=side,
            quantity=qty,
            now=now,
            quote=quote,
            fill_px=fill_px,
            fees=fees,
            signal_ref=request.signal_ref,
            position=position,
        )
        return OrderResult(filled, accepted=True)

    def submit_signal(
        self,
        *,
        signal: str,
        symbol: str,
        quantity: float,
        client_order_id: str | None = None,
        auto_execute: bool = False,
        allow_historical: bool = False,
        signal_ref: str = "",
    ) -> OrderResult | None:
        """
        Map a model BUY/SELL/HOLD into a paper order.

        Default is manual confirmation (`auto_execute=False`), which rejects with a clear reason.
        HOLD produces no order.
        """
        sig = str(signal).upper().strip()
        if sig == "HOLD":
            return None
        if sig not in {"BUY", "SELL"}:
            raise ValueError("signal must be BUY, SELL, or HOLD")
        return self.place_order(
            OrderRequest(
                symbol=symbol,
                side=sig,
                quantity=quantity,
                client_order_id=client_order_id,
                signal_ref=signal_ref or f"signal:{sig}",
                allow_historical=allow_historical or self.mode == "demo",
                confirm=bool(auto_execute),
            )
        )

    def portfolio_snapshot(self) -> dict:
        account = self.account()
        positions = self.store.list_positions(self.account_id)
        rows = []
        invested = 0.0
        unrealized = 0.0
        for pos in positions:
            quote = self.resolve_price(pos["symbol"])
            mark = float(quote.price) if quote is not None else float(pos["avg_price"])
            source = quote.source if quote is not None else "avg_entry_fallback"
            freshness = quote.freshness if quote is not None else "unavailable"
            price_time = quote.quote_time.isoformat() if quote is not None else ""
            market_value = float(pos["quantity"]) * mark
            upnl = (mark - float(pos["avg_price"])) * float(pos["quantity"])
            invested += market_value
            unrealized += upnl
            rows.append(
                {
                    "symbol": pos["symbol"],
                    "quantity": float(pos["quantity"]),
                    "avg_entry": float(pos["avg_price"]),
                    "mark": mark,
                    "market_value": market_value,
                    "unrealized_pnl": upnl,
                    "source": source,
                    "freshness": freshness,
                    "price_time": price_time,
                }
            )
        cash = float(account["cash"])
        equity = cash + invested
        initial = float(account["initial_capital"])
        closed = self.store.closed_trades(self.account_id)
        realized = sum(float(t["realized_pnl"]) for t in closed)
        peak = initial
        max_dd = 0.0
        for point in self.store.equity_points(self.account_id):
            peak = max(peak, float(point["equity"]))
            max_dd = min(max_dd, float(point["drawdown"]))
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0 if peak > 0 else 0.0
        max_dd = min(max_dd, drawdown)
        wins = [t for t in closed if float(t["realized_pnl"]) > 0]
        losses = [t for t in closed if float(t["realized_pnl"]) < 0]
        gross_profit = sum(float(t["realized_pnl"]) for t in wins)
        gross_loss = abs(sum(float(t["realized_pnl"]) for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else None
        return {
            "account_id": self.account_id,
            "mode": self.mode,
            "initial_capital": initial,
            "cash": cash,
            "invested": invested,
            "equity": equity,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "total_return": (equity / initial - 1.0) if initial else 0.0,
            "drawdown": drawdown,
            "max_drawdown": max_dd,
            "positions": rows,
            "closed_trades": closed,
            "orders": self.store.recent_orders(self.account_id, limit=100),
            "win_rate": (len(wins) / len(closed)) if closed else None,
            "profit_factor": profit_factor,
            "net_pnl": realized + unrealized,
            "equity_curve": self.store.equity_points(self.account_id),
        }

    def run_demo_workflow(self) -> list[OrderResult]:
        if self.mode != "demo":
            raise RuntimeError("Demo workflow is only available on the demo account.")
        self.reset(DEMO_INITIAL_CAPITAL)
        results = []
        for step in demo_workflow_steps():
            results.append(
                self.place_order(
                    OrderRequest(
                        symbol=step["symbol"],
                        side=step["side"],
                        quantity=step["quantity"],
                        client_order_id=step["client_order_id"],
                        signal_ref="demo_workflow",
                        allow_historical=True,
                        confirm=True,
                    )
                )
            )
        return results

    def _position_marks(self) -> dict[str, float]:
        marks: dict[str, float] = {}
        for pos in self.store.list_positions(self.account_id):
            quote = self.resolve_price(pos["symbol"])
            marks[pos["symbol"]] = float(quote.price) if quote is not None else float(pos["avg_price"])
        return marks

    def _reject(
        self,
        *,
        client_id: str,
        symbol: str,
        side: str,
        quantity: float,
        now: datetime,
        reason: str,
        signal_ref: str = "",
        quote: PriceQuote | None = None,
    ) -> dict:
        order = {
            "order_id": f"rej-{uuid.uuid4().hex[:12]}",
            "account_id": self.account_id,
            "client_order_id": client_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "status": "rejected",
            "submitted_at": _iso(now),
            "filled_at": None,
            "requested_price": float(quote.price) if quote else None,
            "fill_price": None,
            "fees": 0.0,
            "price_source": quote.source if quote else "",
            "price_time": _iso(quote.quote_time) if quote else None,
            "freshness": quote.freshness if quote else "",
            "rejection_reason": reason,
            "signal_ref": signal_ref,
        }
        self.store.insert_order(order)
        return order

    def _fill(
        self,
        *,
        client_id: str,
        symbol: str,
        side: str,
        quantity: float,
        now: datetime,
        quote: PriceQuote,
        fill_px: float,
        fees: float,
        signal_ref: str,
        position: dict,
    ) -> dict:
        account = self.account()
        cash = float(account["cash"])
        pos_qty = float(position.get("quantity", 0.0))
        avg = float(position.get("avg_price", 0.0))
        notional = quantity * fill_px
        if side == "BUY":
            cash -= notional + fees
            new_qty = pos_qty + quantity
            new_avg = ((pos_qty * avg) + notional) / new_qty if new_qty else 0.0
            self.store.upsert_position(self.account_id, symbol, new_qty, new_avg)
            self.store.set_cash(
                self.account_id,
                cash,
                kind="buy",
                amount=-(notional + fees),
                ref=client_id,
                detail=f"BUY {quantity} {symbol} @ {fill_px:.4f}",
            )
        else:
            cash += notional - fees
            realized = (fill_px - avg) * quantity - fees
            new_qty = pos_qty - quantity
            new_avg = avg if new_qty > 1e-12 else 0.0
            self.store.upsert_position(self.account_id, symbol, new_qty, new_avg)
            self.store.set_cash(
                self.account_id,
                cash,
                kind="sell",
                amount=notional - fees,
                ref=client_id,
                detail=f"SELL {quantity} {symbol} @ {fill_px:.4f}",
            )
            self.store.insert_closed_trade(
                {
                    "account_id": self.account_id,
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": avg,
                    "exit_price": fill_px,
                    "entry_ts": position.get("updated_at") or _iso(now),
                    "exit_ts": _iso(now),
                    "realized_pnl": realized,
                    "fees": fees,
                }
            )

        order = {
            "order_id": f"fill-{uuid.uuid4().hex[:12]}",
            "account_id": self.account_id,
            "client_order_id": client_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "status": "filled",
            "submitted_at": _iso(now),
            "filled_at": _iso(now),
            "requested_price": float(quote.price),
            "fill_price": fill_px,
            "fees": fees,
            "price_source": quote.source,
            "price_time": _iso(quote.quote_time),
            "freshness": quote.freshness,
            "rejection_reason": "",
            "signal_ref": signal_ref,
        }
        self.store.insert_order(order)
        snap = self.portfolio_snapshot()
        self.store.append_equity(
            self.account_id,
            cash=snap["cash"],
            equity=snap["equity"],
            drawdown=snap["drawdown"],
        )
        return order


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.replace(microsecond=0).isoformat()

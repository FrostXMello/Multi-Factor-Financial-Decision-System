"""Paper-order risk gates. Separate from the research NoTrade gate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperRiskConfig:
    max_position_value: float = 50_000.0
    max_order_value: float = 25_000.0
    max_portfolio_exposure: float = 0.95
    max_daily_loss: float = 5_000.0
    allow_short: bool = False
    duplicate_window_seconds: int = 5


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str = ""


def validate_order_risk(
    *,
    side: str,
    quantity: float,
    fill_price: float,
    fees: float,
    cash: float,
    equity: float,
    position_qty: float,
    position_market_value: float,
    invested_value: float,
    realized_pnl_today: float,
    cfg: PaperRiskConfig | None = None,
) -> RiskDecision:
    cfg = cfg or PaperRiskConfig()
    side_u = str(side).upper().strip()
    qty = float(quantity)
    px = float(fill_price)
    if qty <= 0:
        return RiskDecision(False, "Quantity must be positive.")
    if px <= 0:
        return RiskDecision(False, "Execution price must be positive.")
    order_value = qty * px
    if order_value > cfg.max_order_value + 1e-9:
        return RiskDecision(False, f"Order value {order_value:.2f} exceeds max order value {cfg.max_order_value:.2f}.")
    if side_u == "BUY":
        cash_needed = order_value + float(fees)
        if cash_needed > cash + 1e-9:
            return RiskDecision(False, f"Insufficient cash: need {cash_needed:.2f}, available {cash:.2f}.")
        new_position_value = max(0.0, position_market_value) + order_value
        if new_position_value > cfg.max_position_value + 1e-9:
            return RiskDecision(
                False,
                f"Position value {new_position_value:.2f} would exceed max position value {cfg.max_position_value:.2f}.",
            )
        new_invested = invested_value + order_value
        if equity > 0 and (new_invested / equity) > cfg.max_portfolio_exposure + 1e-12:
            return RiskDecision(
                False,
                f"Portfolio exposure would exceed {cfg.max_portfolio_exposure:.0%} of equity.",
            )
    elif side_u == "SELL":
        if not cfg.allow_short and qty > position_qty + 1e-9:
            return RiskDecision(
                False,
                f"Insufficient holdings: have {position_qty:.4f}, tried to sell {qty:.4f}. Short selling is disabled.",
            )
    else:
        return RiskDecision(False, "Side must be BUY or SELL.")
    if realized_pnl_today <= -abs(cfg.max_daily_loss) + 1e-9:
        return RiskDecision(False, f"Daily loss limit reached ({cfg.max_daily_loss:.2f}).")
    return RiskDecision(True, "")

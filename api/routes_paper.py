"""Paper-trading HTTP routes. Never submits real broker orders."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.deps import paper_engine
from api.serialize import jsonable
from paper_trading.engine import OrderRequest

router = APIRouter(prefix="/api/paper", tags=["paper"])


class OrderBody(BaseModel):
    symbol: str
    side: str
    quantity: float = Field(gt=0)
    mode: str = "paper"
    allow_historical: bool = False
    confirm: bool = True
    client_order_id: Optional[str] = None
    signal_ref: str = ""
    auto_execute_signal: bool = False
    signal: Optional[str] = None


class ModeBody(BaseModel):
    mode: str = "paper"


def _mode(mode: str) -> str:
    mode_u = str(mode).strip().lower()
    if mode_u not in {"paper", "demo"}:
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'demo'")
    return mode_u


@router.get("/account")
def paper_account(mode: str = "paper") -> dict:
    engine = paper_engine(mode=_mode(mode))
    snap = engine.portfolio_snapshot()
    return {
        "mode": engine.mode,
        "account_id": engine.account_id,
        "demo_labelled": engine.mode == "demo",
        "snapshot": jsonable(snap),
        "disclaimer": "Paper/demo ledger only. No real broker order is submitted.",
    }


@router.post("/orders")
def place_paper_order(body: OrderBody) -> dict:
    engine = paper_engine(mode=_mode(body.mode))
    allow_historical = body.allow_historical or engine.mode == "demo"
    if body.signal:
        result = engine.submit_signal(
            signal=body.signal,
            symbol=body.symbol,
            quantity=body.quantity,
            client_order_id=body.client_order_id,
            auto_execute=body.auto_execute_signal or body.confirm,
            allow_historical=allow_historical,
            signal_ref=body.signal_ref or f"api:{body.signal}",
        )
        if result is None:
            return {"accepted": False, "status": "hold", "detail": "HOLD produces no paper order.", "order": None}
    else:
        result = engine.place_order(
            OrderRequest(
                symbol=body.symbol,
                side=body.side,
                quantity=body.quantity,
                client_order_id=body.client_order_id,
                signal_ref=body.signal_ref or "api_manual",
                allow_historical=allow_historical,
                confirm=body.confirm,
            )
        )
    payload = {
        "accepted": result.accepted,
        "status": result.status,
        "rejection_reason": result.rejection_reason,
        "order": jsonable(result.order),
        "mode": engine.mode,
        "demo_labelled": engine.mode == "demo",
    }
    if not result.accepted:
        raise HTTPException(status_code=400, detail=payload)
    return payload


@router.post("/demo/run")
def run_demo_workflow() -> dict:
    engine = paper_engine(mode="demo")
    results = engine.run_demo_workflow()
    return {
        "mode": "demo",
        "demo_labelled": True,
        "account_id": engine.account_id,
        "results": [
            {
                "accepted": item.accepted,
                "status": item.status,
                "rejection_reason": item.rejection_reason,
                "order": jsonable(item.order),
            }
            for item in results
        ],
        "snapshot": jsonable(engine.portfolio_snapshot()),
        "disclaimer": "Deterministic DEMO / REPLAY workflow. Isolated from paper:default.",
    }


@router.post("/reset")
def reset_account(body: ModeBody) -> dict:
    engine = paper_engine(mode=_mode(body.mode))
    account = engine.reset()
    return {
        "mode": engine.mode,
        "account_id": engine.account_id,
        "account": jsonable(account),
        "demo_labelled": engine.mode == "demo",
    }

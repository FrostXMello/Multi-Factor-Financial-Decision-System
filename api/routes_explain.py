"""AI explainability routes. Never expose provider API keys."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.routes_research import _LAST_RESULT
from api.serialize import jsonable
from reasoning.service import ExplainRequest, explain_decision, explain_risk_block, provider_status

router = APIRouter(prefix="/api/explain", tags=["explain"])

_LAST_EXPLANATION: dict[str, Any] = {}


class ExplainBody(BaseModel):
    ticker: Optional[str] = None
    use_llm: bool = True
    use_last_research: bool = True
    latest: Optional[dict] = None
    reliability: Optional[dict] = None
    metrics: Optional[dict] = None
    strategy_mode: Optional[str] = None
    buy_threshold: Optional[float] = None
    sell_threshold: Optional[float] = None
    sector_context: Optional[dict] = None


class RiskBlockBody(BaseModel):
    ticker: str = Field(min_length=1)
    side: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    quantity: Optional[float] = None
    source: str = "paper_risk"
    extras: Optional[dict] = None


@router.get("/status")
def explain_status() -> dict:
    status = provider_status()
    # Never return secrets; status already omits keys.
    return status


@router.post("/decision")
def explain_decision_route(body: ExplainBody) -> dict:
    summary = (_LAST_RESULT.get("summary") or {}) if body.use_last_research else {}
    latest = body.latest if body.latest is not None else (summary.get("latest") or {})
    reliability = body.reliability if body.reliability is not None else (summary.get("reliability") or {})
    metrics = body.metrics if body.metrics is not None else (summary.get("metrics") or {})
    thresholds = summary.get("thresholds") or {}
    ticker = (body.ticker or summary.get("ticker") or latest.get("symbol") or "").strip().upper()
    if not latest and not body.latest:
        raise HTTPException(
            status_code=404,
            detail="No decision context available. Run research first or supply latest fields.",
        )
    request = ExplainRequest(
        ticker=ticker,
        latest=dict(latest or {}),
        reliability=dict(reliability or {}),
        metrics=dict(metrics or {}),
        strategy_mode=body.strategy_mode or summary.get("strategy_mode"),
        buy_threshold=body.buy_threshold
        if body.buy_threshold is not None
        else thresholds.get("buy_threshold", 0.6),
        sell_threshold=body.sell_threshold
        if body.sell_threshold is not None
        else thresholds.get("sell_threshold", 0.4),
        sector_context=body.sector_context,
        use_llm=body.use_llm,
    )
    explanation = explain_decision(request)
    _LAST_EXPLANATION["payload"] = explanation
    return jsonable(explanation)


@router.get("/last")
def explain_last() -> dict:
    payload = _LAST_EXPLANATION.get("payload")
    if not payload:
        return {
            "available": False,
            "detail": "No explanation is cached in this API process yet.",
        }
    return {"available": True, "explanation": jsonable(payload)}


@router.post("/risk-block")
def explain_risk_block_route(body: RiskBlockBody) -> dict:
    explanation = explain_risk_block(
        ticker=body.ticker,
        side=body.side,
        reason=body.reason,
        quantity=body.quantity,
        source=body.source,
        extras=body.extras,
    )
    _LAST_EXPLANATION["payload"] = explanation
    return jsonable(explanation)


def clear_explain_cache() -> None:
    _LAST_EXPLANATION.clear()

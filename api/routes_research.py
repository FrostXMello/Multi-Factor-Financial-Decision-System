"""Research / prediction / reliability routes wrapping the existing pipeline."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.serialize import jsonable, summarize_research
from four_model_pipeline import FourModelConfig, run_four_model_pipeline

router = APIRouter(prefix="/api/research", tags=["research"])

_LAST_RESULT: dict[str, Any] = {}


class ResearchRunBody(BaseModel):
    ticker: str = Field(default="RELIANCE.NS", min_length=1)
    initial_capital: float = Field(default=100_000.0, gt=0)


@router.post("/run")
def research_run(body: ResearchRunBody) -> dict:
    ticker = body.ticker.strip().upper()
    try:
        result = run_four_model_pipeline(
            ticker,
            FourModelConfig(initial_capital=body.initial_capital),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Research pipeline failed: {exc}") from exc
    result = dict(result)
    result["ticker"] = ticker
    summary = summarize_research(result)
    _LAST_RESULT["summary"] = summary
    _LAST_RESULT["ticker"] = ticker
    # Keep raw pipeline objects in-process for analytics (not returned on the wire).
    _LAST_RESULT["raw"] = {
        "backtest": result.get("backtest"),
        "reliability": result.get("reliability"),
        "walk_forward_fold_scores": result.get("walk_forward_fold_scores"),
        "prices": result.get("prices"),
        "test_df": result.get("test_df"),
    }
    try:
        from api.routes_explain import _LAST_EXPLANATION
        from reasoning.service import ExplainRequest, explain_decision

        explanation = explain_decision(
            ExplainRequest(
                ticker=ticker,
                latest=dict(summary.get("latest") or {}),
                reliability=dict(summary.get("reliability") or {}),
                metrics=dict(summary.get("metrics") or {}),
                strategy_mode=summary.get("strategy_mode"),
                buy_threshold=(summary.get("thresholds") or {}).get("buy_threshold", 0.6),
                sell_threshold=(summary.get("thresholds") or {}).get("sell_threshold", 0.4),
                use_llm=True,
            )
        )
        _LAST_EXPLANATION["payload"] = explanation
        summary = dict(summary)
        summary["explanation"] = explanation
    except Exception as exc:  # noqa: BLE001 - research must still return if explain fails
        summary = dict(summary)
        summary["explanation_error"] = str(exc)
    return summary


@router.get("/last")
def research_last() -> dict:
    if not _LAST_RESULT.get("summary"):
        return {
            "available": False,
            "detail": "No research run is cached in this API process yet. POST /api/research/run first.",
        }
    return {"available": True, "summary": _LAST_RESULT["summary"]}


@router.get("/reliability")
def research_reliability() -> dict:
    summary = _LAST_RESULT.get("summary") or {}
    reliability = summary.get("reliability") if isinstance(summary, dict) else None
    if not reliability:
        return {
            "available": False,
            "detail": "No reliability payload is cached. Run research first.",
        }
    return {
        "available": True,
        "ticker": _LAST_RESULT.get("ticker"),
        "reliability": jsonable(reliability),
        "metrics": summary.get("metrics"),
        "disclaimer": "Walk-forward / holdout diagnostics from the existing evaluation modules.",
    }


def clear_research_cache() -> None:
    _LAST_RESULT.clear()

"""Public explainability service. Never mutates trading signals or risk outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from reasoning.context import build_decision_context
from reasoning.deterministic import explain_from_context, explain_risk_block_from_inputs
from reasoning.llm import LLMProviderError, enrich_with_llm, provider_status as llm_provider_status
from reasoning.schema import empty_explanation, validate_explanation


@dataclass
class ExplainRequest:
    ticker: str = ""
    latest: dict = field(default_factory=dict)
    reliability: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    strategy_mode: Optional[str] = None
    buy_threshold: Optional[float] = 0.6
    sell_threshold: Optional[float] = 0.4
    sector_context: Optional[dict] = None
    paper_rejection: Optional[dict] = None
    use_llm: bool = True


def provider_status(environ: Optional[dict] = None) -> dict:
    status = llm_provider_status(environ)
    status["can_override_signals"] = False
    status["exposes_api_keys"] = False
    return status


def explain_decision(
    request: ExplainRequest,
    *,
    environ: Optional[dict] = None,
    opener: Optional[Callable[..., Any]] = None,
) -> dict:
    if not request.latest and not request.paper_rejection:
        return empty_explanation(ticker=request.ticker)

    ctx = build_decision_context(
        ticker=request.ticker,
        latest=request.latest,
        reliability=request.reliability,
        metrics=request.metrics,
        strategy_mode=request.strategy_mode,
        buy_threshold=request.buy_threshold,
        sell_threshold=request.sell_threshold,
        sector_context=request.sector_context,
        paper_rejection=request.paper_rejection,
    )
    base = explain_from_context(ctx)
    ok, reason = validate_explanation(base, locked_signal=str(ctx["signal"]))
    if not ok:
        out = empty_explanation(ticker=request.ticker, reason=f"Deterministic explanation invalid: {reason}")
        out["decision"]["signal"] = ctx["signal"]
        return out

    if not request.use_llm:
        return base

    try:
        enriched = enrich_with_llm(base, ctx, environ=environ, opener=opener)
    except LLMProviderError as exc:
        fallback = dict(base)
        fallback["mode"] = "deterministic_fallback"
        fallback["provider_error"] = str(exc)
        fallback["limitations"] = list(fallback.get("limitations") or []) + [
            "LLM provider failed or returned an invalid payload; deterministic explanation was kept."
        ]
        return fallback
    return enriched


def explain_risk_block(
    *,
    ticker: str,
    side: str,
    reason: str,
    quantity: Optional[float] = None,
    source: str = "paper_risk",
    extras: Optional[dict] = None,
) -> dict:
    payload = explain_risk_block_from_inputs(
        ticker=ticker,
        side=side,
        quantity=quantity,
        reason=reason,
        source=source,
        extras=extras,
    )
    locked = str(payload["decision"]["signal"])
    ok, message = validate_explanation(payload, locked_signal=locked)
    if not ok:
        return empty_explanation(ticker=ticker, reason=message)
    return payload

"""Structured explanation contract. Signals are inputs, never LLM outputs."""

from __future__ import annotations

from typing import Any, Optional

REQUIRED_TOP_LEVEL = (
    "schema_version",
    "mode",
    "provider",
    "ticker",
    "decision",
    "summary",
    "confidence_interpretation",
    "probability_of_profit_note",
    "supporting_factors",
    "conflicting_signals",
    "risk_warnings",
    "market_regime",
    "sector_context",
    "supporting_data",
    "unavailable",
    "limitations",
    "signal_lock",
)


def empty_explanation(*, ticker: str = "", reason: str = "No structured decision context was provided.") -> dict:
    return {
        "schema_version": 1,
        "mode": "deterministic",
        "provider": "none",
        "ticker": ticker or "unavailable",
        "decision": {
            "signal": "unavailable",
            "signal_model": "unavailable",
            "risk_blocked": False,
            "block_reason": None,
        },
        "summary": reason,
        "confidence_interpretation": "unavailable",
        "probability_of_profit_note": (
            "Model probability is a classifier score for the research label. "
            "It is not the probability that a trade is profitable after costs."
        ),
        "supporting_factors": [],
        "conflicting_signals": [],
        "risk_warnings": [],
        "market_regime": {"available": False, "label": "unavailable", "evidence": []},
        "sector_context": {"available": False, "detail": "Sector context was not supplied."},
        "supporting_data": {},
        "unavailable": ["decision_inputs"],
        "limitations": [
            "Explanations only restate supplied structured fields.",
            "Missing values are labelled unavailable and are not invented.",
        ],
        "signal_lock": True,
    }


def validate_explanation(payload: Any, *, locked_signal: str) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "Explanation payload must be an object."
    for key in REQUIRED_TOP_LEVEL:
        if key not in payload:
            return False, f"Missing field: {key}"
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        return False, "decision must be an object."
    signal = str(decision.get("signal", "")).upper().strip()
    locked = str(locked_signal).upper().strip()
    if signal != locked:
        return False, f"signal_lock violation: explanation signal {signal!r} != locked {locked!r}"
    if payload.get("signal_lock") is not True:
        return False, "signal_lock must be true."
    for list_key in ("supporting_factors", "conflicting_signals", "risk_warnings", "unavailable", "limitations"):
        if not isinstance(payload.get(list_key), list):
            return False, f"{list_key} must be a list."
    return True, ""


def redact_secrets(text: str, secrets: Optional[list[str]] = None) -> str:
    out = str(text)
    for secret in secrets or []:
        if secret:
            out = out.replace(secret, "[redacted]")
    return out

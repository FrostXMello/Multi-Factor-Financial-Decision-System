"""
Optional LLM narrative layer.

The model may only rewrite narrative fields from supplied facts.
It must not invent numbers or change the locked signal / risk decision.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from reasoning.schema import redact_secrets, validate_explanation


class LLMProviderError(RuntimeError):
    pass


def provider_status(environ: Optional[dict] = None) -> dict:
    env = environ if environ is not None else os.environ
    configured = bool(str(env.get("AI_REASONING_API_KEY", "")).strip()) and bool(
        str(env.get("AI_REASONING_BASE_URL", "")).strip()
    )
    return {
        "configured": configured,
        "provider": str(env.get("AI_REASONING_PROVIDER", "openai_compatible") or "openai_compatible"),
        "model": str(env.get("AI_REASONING_MODEL", "") or "unavailable") if configured else "unavailable",
        "detail": (
            "LLM assist is configured. Narratives are validated against the locked signal."
            if configured
            else "No AI_REASONING_API_KEY/BASE_URL. Deterministic explanations are used."
        ),
    }


def enrich_with_llm(
    deterministic: dict,
    context: dict,
    *,
    environ: Optional[dict] = None,
    opener: Optional[Callable[..., Any]] = None,
) -> dict:
    env = environ if environ is not None else os.environ
    status = provider_status(env)
    if not status["configured"]:
        return deterministic

    api_key = str(env.get("AI_REASONING_API_KEY", "")).strip()
    base = str(env.get("AI_REASONING_BASE_URL", "")).strip().rstrip("/")
    model = str(env.get("AI_REASONING_MODEL", "gpt-4o-mini")).strip() or "gpt-4o-mini"
    path = str(env.get("AI_REASONING_CHAT_PATH", "/chat/completions")).strip() or "/chat/completions"
    url = f"{base}{path if path.startswith('/') else '/' + path}"

    locked_signal = str(deterministic["decision"]["signal"])
    prompt = {
        "instructions": [
            "You explain a trading-system decision using ONLY the provided structured facts.",
            "Do not invent prices, indicators, accuracies, or any missing values.",
            "Do not change decision.signal, decision.signal_model, risk_blocked, or block_reason.",
            "Keep signal_lock true.",
            "Return a single JSON object with the same keys as the template.",
            "You may rewrite summary, confidence_interpretation, and the interpretation strings inside factor lists.",
        ],
        "locked_signal": locked_signal,
        "facts": context,
        "template": deterministic,
    }
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You are an explainability assistant for a quantitative research system. Never invent numbers.",
            },
            {"role": "user", "content": json.dumps(prompt)},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=float(env.get("AI_REASONING_TIMEOUT_S", "20"))) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - provider failures must fall back
        raise LLMProviderError(redact_secrets(str(exc), [api_key])) from exc

    try:
        payload = json.loads(raw)
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        parsed = json.loads(content)
    except Exception as exc:  # noqa: BLE001
        raise LLMProviderError(redact_secrets(f"Malformed LLM response: {exc}", [api_key])) from exc

    # Force locked decision fields from the deterministic base.
    parsed = dict(parsed)
    parsed["decision"] = dict(deterministic["decision"])
    parsed["signal_lock"] = True
    parsed["supporting_data"] = deterministic.get("supporting_data", {})
    parsed["unavailable"] = deterministic.get("unavailable", [])
    parsed["probability_of_profit_note"] = deterministic.get("probability_of_profit_note")
    parsed["schema_version"] = 1
    parsed["mode"] = "llm_assisted"
    parsed["provider"] = status["provider"]
    parsed["ticker"] = deterministic.get("ticker")

    ok, reason = validate_explanation(parsed, locked_signal=locked_signal)
    if not ok:
        raise LLMProviderError(f"LLM explanation failed validation: {reason}")
    return parsed

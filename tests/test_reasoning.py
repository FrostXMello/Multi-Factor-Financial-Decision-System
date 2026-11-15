"""Offline explainability tests. No live LLM calls; fixtures are labelled structured inputs."""

from __future__ import annotations

import json

import pytest

from reasoning.llm import LLMProviderError, enrich_with_llm
from reasoning.schema import validate_explanation
from reasoning.service import ExplainRequest, explain_decision, explain_risk_block, provider_status


def _latest(**overrides):
    base = {
        "date": "2024-06-14",
        "signal": "BUY",
        "signal_model": "BUY",
        "proba": 0.72,
        "close": 100.0,
        "ma50": 95.0,
        "risk_score": 0.2,
        "position_size": 0.8,
        "no_trade": False,
        "realized_vol": 0.012,
        "rolling_drawdown": -0.03,
    }
    base.update(overrides)
    return base


def test_deterministic_buy_explanation_and_confidence_distinction():
    out = explain_decision(
        ExplainRequest(
            ticker="RELIANCE.NS",
            latest=_latest(),
            reliability={"decision_method": "sigmoid", "evaluation": "walk_forward", "oos_metrics": {"accuracy": 0.58}},
            buy_threshold=0.6,
            sell_threshold=0.4,
            use_llm=False,
        )
    )
    assert out["mode"] == "deterministic"
    assert out["decision"]["signal"] == "BUY"
    assert out["signal_lock"] is True
    assert "not the probability" in out["probability_of_profit_note"].lower() or "not" in out["probability_of_profit_note"].lower()
    assert any(item["factor"] == "model_probability" for item in out["supporting_factors"])
    assert out["supporting_data"]["proba"] == pytest.approx(0.72)
    ok, reason = validate_explanation(out, locked_signal="BUY")
    assert ok, reason


def test_missing_inputs_are_labelled_unavailable_not_invented():
    out = explain_decision(
        ExplainRequest(
            ticker="TCS.NS",
            latest={"signal": "HOLD"},
            use_llm=False,
        )
    )
    assert out["decision"]["signal"] == "HOLD"
    assert "proba" in out["unavailable"]
    assert out["supporting_data"]["proba"] is None
    assert out["market_regime"]["available"] is False
    assert out["sector_context"]["available"] is False


def test_risk_blocked_buy_is_explained():
    out = explain_decision(
        ExplainRequest(
            ticker="INFY.NS",
            latest=_latest(signal="HOLD", signal_model="BUY", no_trade=True, risk_score=0.9, realized_vol=0.06),
            use_llm=False,
        )
    )
    assert out["decision"]["risk_blocked"] is True
    assert out["decision"]["signal"] == "HOLD"
    assert out["decision"]["signal_model"] == "BUY"
    assert out["risk_warnings"]
    assert "blocked" in (out["decision"]["block_reason"] or "").lower()


def test_paper_risk_block_endpoint_payload():
    out = explain_risk_block(
        ticker="RELIANCE.NS",
        side="BUY",
        reason="Insufficient cash: need 50000.00, available 1000.00.",
        quantity=10,
        source="paper_risk",
    )
    assert out["decision"]["risk_blocked"] is True
    assert out["decision"]["signal"] == "BUY"
    assert "Insufficient cash" in out["summary"]


def test_provider_failure_falls_back_without_changing_signal(monkeypatch):
    def boom(*args, **kwargs):
        raise LLMProviderError("simulated provider outage")

    monkeypatch.setattr("reasoning.service.enrich_with_llm", boom)
    out = explain_decision(
        ExplainRequest(ticker="RELIANCE.NS", latest=_latest(), use_llm=True),
        environ={"AI_REASONING_API_KEY": "secret", "AI_REASONING_BASE_URL": "https://example.invalid"},
    )
    assert out["decision"]["signal"] == "BUY"
    assert out["mode"] == "deterministic_fallback"
    assert "provider_error" in out
    assert "secret" not in json.dumps(out)


def test_invalid_llm_payload_is_rejected_and_signal_cannot_be_overridden():
    deterministic = explain_decision(
        ExplainRequest(ticker="RELIANCE.NS", latest=_latest(), use_llm=False)
    )

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            # Malformed: supporting_factors must be a list. Even if the model tries SELL,
            # the service overwrites decision from the locked deterministic payload first.
            bad = {
                "schema_version": 1,
                "mode": "llm_assisted",
                "provider": "x",
                "ticker": "RELIANCE.NS",
                "decision": {"signal": "SELL", "signal_model": "SELL", "risk_blocked": False, "block_reason": None},
                "summary": "attempted override",
                "confidence_interpretation": "x",
                "probability_of_profit_note": "x",
                "supporting_factors": "not-a-list",
                "conflicting_signals": [],
                "risk_warnings": [],
                "market_regime": {},
                "sector_context": {},
                "supporting_data": {},
                "unavailable": [],
                "limitations": [],
                "signal_lock": True,
            }
            return json.dumps({"choices": [{"message": {"content": json.dumps(bad)}}]}).encode("utf-8")

    def opener(request, timeout=20):
        return _Resp()

    with pytest.raises(LLMProviderError, match="validation|supporting_factors"):
        enrich_with_llm(
            deterministic,
            {"signal": "BUY"},
            environ={
                "AI_REASONING_API_KEY": "secret-key",
                "AI_REASONING_BASE_URL": "https://example.invalid",
                "AI_REASONING_MODEL": "demo-model",
            },
            opener=opener,
        )

    ok, _ = validate_explanation(
        {**deterministic, "decision": {"signal": "SELL", "signal_model": "BUY", "risk_blocked": False, "block_reason": None}},
        locked_signal="BUY",
    )
    assert ok is False


def test_provider_status_hides_secrets():
    status = provider_status(
        {
            "AI_REASONING_API_KEY": "super-secret",
            "AI_REASONING_BASE_URL": "https://example.com",
            "AI_REASONING_MODEL": "demo",
        }
    )
    blob = json.dumps(status)
    assert status["configured"] is True
    assert "super-secret" not in blob
    assert status["can_override_signals"] is False
    assert status["exposes_api_keys"] is False


def test_empty_request_does_not_fabricate_a_trade():
    out = explain_decision(ExplainRequest(ticker="", use_llm=False))
    assert out["decision"]["signal"] == "unavailable"
    assert out["supporting_factors"] == []

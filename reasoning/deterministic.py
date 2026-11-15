"""Deterministic explanations built only from structured context."""

from __future__ import annotations

from typing import Any, Optional


def _fmt(value: Optional[float], *, digits: int = 4) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.{digits}f}"


def explain_from_context(ctx: dict) -> dict:
    signal = str(ctx.get("signal", "unavailable")).upper()
    signal_model = str(ctx.get("signal_model", "unavailable")).upper()
    proba = ctx.get("proba")
    close = ctx.get("close")
    ma50 = ctx.get("ma50")
    buy_th = ctx.get("buy_threshold")
    sell_th = ctx.get("sell_threshold")
    risk_score = ctx.get("risk_score")
    realized_vol = ctx.get("realized_vol")
    drawdown = ctx.get("rolling_drawdown")
    risk_blocked = bool(ctx.get("risk_blocked"))
    block_reason = ctx.get("block_reason")
    unavailable = list(ctx.get("unavailable") or [])

    supporting: list[dict[str, Any]] = []
    conflicting: list[dict[str, Any]] = []
    warnings: list[str] = []

    if proba is not None:
        supporting.append(
            {
                "factor": "model_probability",
                "value": proba,
                "interpretation": (
                    f"Classifier score for the research label is {_fmt(proba)}. "
                    "This is model confidence in the labelled event, not probability of profit."
                ),
            }
        )
    if close is not None and ma50 is not None:
        relation = "above" if close > ma50 else ("below" if close < ma50 else "equal to")
        item = {
            "factor": "trend_vs_ma50",
            "value": {"close": close, "ma50": ma50, "relation": relation},
            "interpretation": f"Close {_fmt(close, digits=2)} is {relation} MA50 {_fmt(ma50, digits=2)}.",
        }
        if signal == "BUY" and close > ma50:
            supporting.append(item)
        elif signal == "SELL" and close < ma50:
            supporting.append(item)
        elif signal in {"BUY", "SELL"}:
            conflicting.append(item)
        else:
            supporting.append(item)
    if buy_th is not None and proba is not None and signal in {"BUY", "HOLD", "SELL"}:
        if proba > buy_th:
            supporting.append(
                {
                    "factor": "buy_threshold",
                    "value": {"proba": proba, "buy_threshold": buy_th},
                    "interpretation": f"Probability {_fmt(proba)} is above the buy threshold {_fmt(buy_th)}.",
                }
            )
        elif signal == "BUY":
            conflicting.append(
                {
                    "factor": "buy_threshold",
                    "value": {"proba": proba, "buy_threshold": buy_th},
                    "interpretation": f"Signal is BUY while probability {_fmt(proba)} is not above buy threshold {_fmt(buy_th)}.",
                }
            )
    if sell_th is not None and proba is not None and signal == "SELL":
        if proba < sell_th:
            supporting.append(
                {
                    "factor": "sell_threshold",
                    "value": {"proba": proba, "sell_threshold": sell_th},
                    "interpretation": f"Probability {_fmt(proba)} is below the sell threshold {_fmt(sell_th)}.",
                }
            )

    if signal_model != "unavailable" and signal_model != signal:
        conflicting.append(
            {
                "factor": "pre_risk_vs_final_signal",
                "value": {"signal_model": signal_model, "signal": signal},
                "interpretation": (
                    f"Pre-risk model signal was {signal_model}; final signal after risk/rules is {signal}."
                ),
            }
        )

    if risk_blocked:
        warnings.append(str(block_reason or "A risk control blocked the actionable trade intent."))
    if risk_score is not None and risk_score >= 0.7:
        warnings.append(f"RiskScore is elevated at {_fmt(risk_score)}.")
    if realized_vol is not None:
        supporting.append(
            {
                "factor": "realized_volatility",
                "value": realized_vol,
                "interpretation": f"Realized volatility reading is {_fmt(realized_vol)}.",
            }
        )
    if drawdown is not None:
        supporting.append(
            {
                "factor": "rolling_drawdown",
                "value": drawdown,
                "interpretation": f"Rolling drawdown reading is {_fmt(drawdown)}.",
            }
        )

    regime = _regime(realized_vol, drawdown, risk_score)
    sector = ctx.get("sector_context") if isinstance(ctx.get("sector_context"), dict) else {
        "available": False,
        "detail": "Sector context was not supplied.",
    }
    if not sector.get("available"):
        if "sector_context" not in unavailable:
            unavailable.append("sector_context")

    summary = _summary(ctx, signal, signal_model, risk_blocked)
    confidence = _confidence_text(proba, ctx.get("decision_method"), ctx.get("oos_accuracy"))

    return {
        "schema_version": 1,
        "mode": "deterministic",
        "provider": "none",
        "ticker": ctx.get("ticker") or "unavailable",
        "decision": {
            "signal": signal,
            "signal_model": signal_model,
            "risk_blocked": risk_blocked,
            "block_reason": block_reason,
        },
        "summary": summary,
        "confidence_interpretation": confidence,
        "probability_of_profit_note": (
            "Model probability estimates confidence in the research classification label "
            "(for example a meaningful forward move). It is not the probability that executing "
            "this signal is profitable after slippage and transaction costs."
        ),
        "supporting_factors": supporting,
        "conflicting_signals": conflicting,
        "risk_warnings": warnings,
        "market_regime": regime,
        "sector_context": sector,
        "supporting_data": {
            "date": ctx.get("date"),
            "strategy_mode": ctx.get("strategy_mode"),
            "proba": proba,
            "close": close,
            "ma50": ma50,
            "ma20": ctx.get("ma20"),
            "rsi14": ctx.get("rsi14"),
            "risk_score": risk_score,
            "position_size": ctx.get("position_size"),
            "realized_vol": realized_vol,
            "rolling_drawdown": drawdown,
            "no_trade": ctx.get("no_trade"),
            "buy_threshold": buy_th,
            "sell_threshold": sell_th,
            "decision_method": ctx.get("decision_method"),
            "evaluation": ctx.get("evaluation"),
            "oos_accuracy": ctx.get("oos_accuracy"),
            "metrics": ctx.get("metrics") or {},
        },
        "unavailable": sorted(set(unavailable)),
        "limitations": [
            "Deterministic template uses only supplied structured fields.",
            "Unavailable fields are listed and not filled with invented values.",
            "This explanation cannot change BUY/SELL/HOLD, risk gates, or orders.",
        ],
        "signal_lock": True,
    }


def explain_risk_block_from_inputs(
    *,
    ticker: str,
    side: str,
    quantity: Optional[float],
    reason: str,
    source: str = "paper_risk",
    extras: Optional[dict] = None,
) -> dict:
    reason_text = str(reason or "").strip() or "unavailable"
    extras = dict(extras or {})
    supporting_data = {
        "ticker": ticker.upper() if ticker else "unavailable",
        "side": str(side).upper() if side else "unavailable",
        "quantity": quantity,
        "source": source,
        **{key: extras[key] for key in extras if extras[key] is not None},
    }
    unavailable = [key for key, value in supporting_data.items() if value in (None, "", "unavailable")]
    return {
        "schema_version": 1,
        "mode": "deterministic",
        "provider": "none",
        "ticker": supporting_data["ticker"],
        "decision": {
            "signal": supporting_data["side"] if supporting_data["side"] in {"BUY", "SELL", "HOLD"} else "unavailable",
            "signal_model": "unavailable",
            "risk_blocked": True,
            "block_reason": reason_text,
        },
        "summary": (
            f"The {source.replace('_', ' ')} blocked {supporting_data['side']} "
            f"on {supporting_data['ticker']}: {reason_text}"
        ),
        "confidence_interpretation": (
            "A risk rejection is a hard control outcome. It is not a model-confidence score "
            "and not a probability of profit."
        ),
        "probability_of_profit_note": (
            "No profit probability is implied by a risk block. The order was not filled."
        ),
        "supporting_factors": [
            {
                "factor": "risk_rejection_reason",
                "value": reason_text,
                "interpretation": "Human-readable rejection text from the risk engine.",
            }
        ],
        "conflicting_signals": [],
        "risk_warnings": [reason_text],
        "market_regime": {"available": False, "label": "unavailable", "evidence": []},
        "sector_context": {"available": False, "detail": "Sector context was not supplied."},
        "supporting_data": supporting_data,
        "unavailable": unavailable,
        "limitations": [
            "Risk-block explanations restate the rejection reason and supplied order fields only.",
            "This path cannot approve or override the rejection.",
        ],
        "signal_lock": True,
    }


def _regime(
    realized_vol: Optional[float],
    drawdown: Optional[float],
    risk_score: Optional[float],
) -> dict:
    evidence = []
    if realized_vol is not None:
        evidence.append({"metric": "realized_vol", "value": realized_vol})
    if drawdown is not None:
        evidence.append({"metric": "rolling_drawdown", "value": drawdown})
    if risk_score is not None:
        evidence.append({"metric": "risk_score", "value": risk_score})
    if not evidence:
        return {"available": False, "label": "unavailable", "evidence": []}
    label = "mixed"
    if risk_score is not None and risk_score >= 0.7:
        label = "elevated_risk"
    elif realized_vol is not None and realized_vol >= 0.03:
        label = "high_volatility"
    elif drawdown is not None and drawdown <= -0.08:
        label = "drawdown_pressure"
    elif risk_score is not None and risk_score <= 0.35:
        label = "calm"
    return {"available": True, "label": label, "evidence": evidence}


def _summary(ctx: dict, signal: str, signal_model: str, risk_blocked: bool) -> str:
    ticker = ctx.get("ticker") or "unavailable"
    parts = [f"{ticker}: final decision is {signal}."]
    if signal_model != "unavailable" and signal_model != signal:
        parts.append(f"Pre-risk model signal was {signal_model}.")
    if risk_blocked:
        parts.append(str(ctx.get("block_reason") or "A risk control blocked the trade intent."))
    elif signal == "HOLD":
        parts.append("No new entry/exit rule fired on the supplied thresholds and trend checks.")
    elif signal in {"BUY", "SELL"}:
        parts.append("The final signal follows the existing strategy and risk modules.")
    return " ".join(parts)


def _confidence_text(
    proba: Optional[float],
    decision_method: Optional[str],
    oos_accuracy: Optional[float],
) -> str:
    bits = []
    if proba is None:
        bits.append("Model probability is unavailable, so confidence cannot be scored from the classifier output.")
    else:
        if proba >= 0.7:
            strength = "high relative to a 0.5 baseline"
        elif proba <= 0.3:
            strength = "low relative to a 0.5 baseline"
        else:
            strength = "moderate / near decision boundaries"
        bits.append(f"Classifier score {_fmt(proba)} indicates {strength}.")
    method = decision_method or "unavailable"
    if method != "unavailable":
        bits.append(f"Probability source method: {method}.")
    if oos_accuracy is not None:
        bits.append(
            f"Reported out-of-sample classification accuracy is {_fmt(oos_accuracy)}. "
            "Accuracy is not expected return."
        )
    bits.append("Do not read model confidence as probability of profit.")
    return " ".join(bits)

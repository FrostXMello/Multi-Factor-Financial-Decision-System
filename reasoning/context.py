"""Build explainability context from structured research/risk inputs only."""

from __future__ import annotations

from typing import Any, Optional


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_decision_context(
    *,
    ticker: str,
    latest: Optional[dict] = None,
    reliability: Optional[dict] = None,
    metrics: Optional[dict] = None,
    strategy_mode: Optional[str] = None,
    buy_threshold: Optional[float] = None,
    sell_threshold: Optional[float] = None,
    sector_context: Optional[dict] = None,
    paper_rejection: Optional[dict] = None,
) -> dict:
    """
    Normalize caller-supplied fields. Missing values stay unavailable.
    This function never downloads market data and never invents numbers.
    """
    latest = dict(latest or {})
    reliability = dict(reliability or {})
    metrics = dict(metrics or {})
    unavailable: list[str] = []

    signal = _str_or_none(latest.get("signal"))
    if signal:
        signal = signal.upper()
    else:
        signal = "unavailable"
        unavailable.append("signal")

    signal_model = _str_or_none(latest.get("signal_model"))
    if signal_model:
        signal_model = signal_model.upper()
    else:
        signal_model = "unavailable"
        unavailable.append("signal_model")

    proba = _num(latest.get("proba"))
    if proba is None:
        unavailable.append("proba")
    close = _num(latest.get("close"))
    if close is None:
        unavailable.append("close")
    ma50 = _num(latest.get("ma50"))
    if ma50 is None:
        unavailable.append("ma50")
    ma20 = _num(latest.get("ma20"))
    if ma20 is None:
        unavailable.append("ma20")
    rsi = _num(latest.get("rsi14"))
    if rsi is None:
        unavailable.append("rsi14")
    risk_score = _num(latest.get("risk_score"))
    if risk_score is None:
        unavailable.append("risk_score")
    position_size = _num(latest.get("position_size"))
    realized_vol = _num(latest.get("realized_vol"))
    if realized_vol is None:
        unavailable.append("realized_vol")
    drawdown = _num(latest.get("rolling_drawdown"))
    if drawdown is None:
        unavailable.append("rolling_drawdown")

    no_trade = latest.get("no_trade")
    if no_trade is None:
        unavailable.append("no_trade")
        risk_blocked = False
        block_reason = None
    else:
        risk_blocked = bool(no_trade) and signal_model == "BUY" and signal == "HOLD"
        block_reason = (
            "Research risk gate blocked a new BUY (NoTrade). Exits remain allowed."
            if risk_blocked
            else None
        )

    if paper_rejection:
        risk_blocked = True
        block_reason = _str_or_none(paper_rejection.get("reason")) or "Paper risk control rejected the order."
        if _str_or_none(paper_rejection.get("signal")):
            # Paper rejection explains a blocked order; decision signal stays the supplied trading intent.
            pass

    oos_accuracy = None
    oos_metrics = reliability.get("oos_metrics")
    if isinstance(oos_metrics, dict):
        oos_accuracy = _num(oos_metrics.get("accuracy"))
    if oos_accuracy is None:
        oos_accuracy = _num(reliability.get("insample_accuracy"))
        if oos_accuracy is not None:
            unavailable.append("oos_accuracy")
        else:
            unavailable.append("model_accuracy")

    return {
        "ticker": (ticker or "unavailable").upper(),
        "strategy_mode": _str_or_none(strategy_mode) or "unavailable",
        "signal": signal,
        "signal_model": signal_model,
        "proba": proba,
        "close": close,
        "ma50": ma50,
        "ma20": ma20,
        "rsi14": rsi,
        "risk_score": risk_score,
        "position_size": position_size,
        "realized_vol": realized_vol,
        "rolling_drawdown": drawdown,
        "no_trade": no_trade if no_trade is None else bool(no_trade),
        "risk_blocked": bool(risk_blocked),
        "block_reason": block_reason,
        "buy_threshold": _num(buy_threshold),
        "sell_threshold": _num(sell_threshold),
        "decision_method": _str_or_none(reliability.get("decision_method")) or "unavailable",
        "evaluation": _str_or_none(reliability.get("evaluation")) or "unavailable",
        "oos_accuracy": oos_accuracy,
        "metrics": {key: metrics[key] for key in metrics if metrics.get(key) is not None},
        "sector_context": sector_context
        if isinstance(sector_context, dict)
        else {"available": False, "detail": "Sector context was not supplied."},
        "paper_rejection": paper_rejection or None,
        "date": _str_or_none(latest.get("date")),
        "unavailable": sorted(set(unavailable)),
    }

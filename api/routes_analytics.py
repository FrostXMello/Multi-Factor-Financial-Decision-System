"""Strategy research and portfolio analytics API (M7)."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_market_store, paper_engine
from api.routes_research import _LAST_RESULT
from api.serialize import jsonable
from strategy_research.portfolio import portfolio_analytics
from strategy_research.regimes import classify_regimes
from strategy_research.report import build_research_analytics_report
from strategy_research.robustness import walk_forward_robustness
from strategy_research.strategy_compare import compare_strategies

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

_LAST_ANALYTICS: dict[str, Any] = {}


class RegimeBody(BaseModel):
    symbol: str = Field(default="RELIANCE.NS", min_length=1)
    timeframe: str = "1d"
    lookback: int = Field(default=60, ge=2, le=252)
    bull_threshold: float = 0.08
    bear_threshold: float = -0.08


class PortfolioBody(BaseModel):
    mode: str = "paper"
    positions: Optional[list[dict]] = None
    cash: Optional[float] = None


class CompareBody(BaseModel):
    strategies: dict[str, dict]


class RobustnessBody(BaseModel):
    fold_scores: Optional[list[float]] = None
    use_last_research: bool = True


@router.get("/status")
def analytics_status() -> dict:
    return {
        "module": "strategy_research",
        "external_repos_integrated": [],
        "note": (
            "Previously reviewed market-data repos (fenix, nsepython, Bharat-SM-Data, india-stocks-api) "
            "are not used here. Analytics build on the existing backtesting engine only."
        ),
        "cached_report": bool(_LAST_ANALYTICS.get("report")),
    }


@router.post("/regimes")
def analytics_regimes(body: RegimeBody) -> dict:
    store = get_market_store()
    candles = store.read_candles(body.symbol.strip().upper(), body.timeframe)
    if candles.empty:
        raise HTTPException(
            status_code=404,
            detail={
                "available": False,
                "detail": "insufficient_data",
                "message": f"No stored {body.timeframe} candles for {body.symbol}. Run research/backfill first.",
            },
        )
    classified = classify_regimes(
        candles,
        lookback=body.lookback,
        bull_threshold=body.bull_threshold,
        bear_threshold=body.bear_threshold,
    )
    counts = classified["regime_end_of_day"].value_counts().to_dict()
    payload = {
        "symbol": body.symbol.strip().upper(),
        "lookback": body.lookback,
        "lookahead_safe": True,
        "counts": counts,
        "latest": {
            "date": jsonable(classified["Date"].iloc[-1]),
            "regime_end_of_day": classified["regime_end_of_day"].iloc[-1],
            "regime_for_return": classified["regime_for_return"].iloc[-1],
            "trailing_return": jsonable(classified["trailing_return"].iloc[-1]),
        },
        "series_tail": jsonable(classified.tail(30)),
        "disclaimer": "Regime labels use trailing history only; return evaluation should use regime_for_return.",
    }
    _LAST_ANALYTICS["regimes"] = payload
    return payload


@router.post("/portfolio")
def analytics_portfolio(body: PortfolioBody) -> dict:
    if body.positions is not None:
        result = portfolio_analytics(body.positions, cash=body.cash)
    else:
        engine = paper_engine(mode=body.mode)
        snap = engine.portfolio_snapshot()
        result = portfolio_analytics(
            list(snap.get("positions") or []),
            cash=float(snap.get("cash") or 0.0),
        )
        result["account_id"] = snap.get("account_id")
        result["mode"] = snap.get("mode")
    _LAST_ANALYTICS["portfolio"] = result
    return jsonable(result)


@router.post("/compare")
def analytics_compare(body: CompareBody) -> dict:
    # Accept either full backtest-shaped dicts or compact metric dicts with equity_curve omitted.
    prepared: dict[str, dict] = {}
    for name, payload in (body.strategies or {}).items():
        item = dict(payload or {})
        if "equity_curve" in item and isinstance(item["equity_curve"], list):
            item["equity_curve"] = pd.DataFrame(item["equity_curve"])
        prepared[name] = item
    result = compare_strategies(prepared)
    _LAST_ANALYTICS["compare"] = result
    return jsonable(result)


@router.post("/robustness")
def analytics_robustness(body: RobustnessBody) -> dict:
    scores = body.fold_scores
    folds_frame = None
    if body.use_last_research and scores is None:
        summary = _LAST_RESULT.get("summary") or {}
        metrics = summary.get("metrics") or {}
        scores = metrics.get("walk_forward_fold_scores")
        reliability = summary.get("reliability") or {}
        folds = reliability.get("folds")
        if isinstance(folds, list) and folds:
            folds_frame = pd.DataFrame(folds)
        raw = _LAST_RESULT.get("raw") or {}
        if scores is None and isinstance(raw.get("walk_forward_fold_scores"), list):
            scores = raw.get("walk_forward_fold_scores")
    result = walk_forward_robustness(fold_scores=scores, folds_frame=folds_frame)
    _LAST_ANALYTICS["robustness"] = result
    return jsonable(result)


@router.get("/report")
def analytics_report(symbol: Optional[str] = None, lookback: int = 60) -> dict:
    summary = _LAST_RESULT.get("summary") or {}
    raw = _LAST_RESULT.get("raw") or {}
    ticker = (symbol or summary.get("ticker") or _LAST_RESULT.get("ticker") or "").strip().upper()
    if not ticker and not raw:
        return {
            "available": False,
            "detail": "No research result is cached. POST /api/research/run first, or supply stored candles via /api/analytics/regimes.",
        }

    price_history = None
    if ticker:
        candles = get_market_store().read_candles(ticker, "1d")
        if not candles.empty:
            price_history = candles
    backtest = raw.get("backtest")
    reliability = summary.get("reliability") or raw.get("reliability")
    fold_scores = (summary.get("metrics") or {}).get("walk_forward_fold_scores")

    paper = paper_engine(mode="paper").portfolio_snapshot()
    report = build_research_analytics_report(
        ticker=ticker or "unavailable",
        price_history=price_history,
        backtest=backtest if isinstance(backtest, dict) else None,
        comparison_backtests={ticker or "strategy": backtest} if isinstance(backtest, dict) else None,
        reliability={
            **(reliability if isinstance(reliability, dict) else {}),
            "walk_forward_fold_scores": fold_scores,
            "folds": (reliability or {}).get("folds") if isinstance(reliability, dict) else None,
        },
        positions=list(paper.get("positions") or []),
        cash=float(paper.get("cash") or 0.0),
        lookback=lookback,
    )
    # If reliability folds are records, robustness may need rebuilding.
    if not (report.get("robustness") or {}).get("available") and fold_scores:
        report["robustness"] = walk_forward_robustness(fold_scores=fold_scores)
    _LAST_ANALYTICS["report"] = report
    return {"available": True, "report": jsonable(report)}


@router.get("/last")
def analytics_last() -> dict:
    if not _LAST_ANALYTICS:
        return {"available": False, "detail": "No analytics payload cached yet."}
    return {"available": True, "payloads": jsonable(_LAST_ANALYTICS)}


def clear_analytics_cache() -> None:
    _LAST_ANALYTICS.clear()

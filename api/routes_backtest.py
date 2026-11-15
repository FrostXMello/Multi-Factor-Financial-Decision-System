"""Configurable backtesting API wrapping the existing research pipeline."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_market_store, get_scan_store
from api.serialize import jsonable, summarize_research
from core_config import CORE_BUY_THRESHOLD, CORE_SELL_THRESHOLD, CORE_RISK_CONFIG
from four_model_pipeline import FourModelConfig, run_four_model_pipeline
from risk_model import RiskConfig
from transaction_costs import IndiaEquityCostConfig, cost_config_for_segment
from utils import NIFTY50_TICKERS

router = APIRouter(prefix="/api/backtest", tags=["backtesting"])

# Curated sleeves shown in the Backtesting panel. Not an optimizer search.
STRATEGY_PRESETS: list[dict[str, Any]] = [
    {
        "id": "technical_trend",
        "label": "Technical Trend",
        "mode": "technical_only",
        "buy_threshold": 0.60,
        "sell_threshold": 0.40,
        "blurb": "Walk-forward P(up) with Close vs MA50 confirmation.",
    },
    {
        "id": "technical_conservative",
        "label": "Technical Conservative",
        "mode": "technical_only",
        "buy_threshold": 0.70,
        "sell_threshold": 0.30,
        "blurb": "Stricter probability gates — fewer, higher-conviction trades.",
    },
    {
        "id": "momentum",
        "label": "Momentum",
        "mode": "momentum",
        "buy_threshold": 0.60,
        "sell_threshold": 0.40,
        "blurb": "Trend-following: bullish model + MA structure + RSI band.",
    },
    {
        "id": "mean_reversion",
        "label": "Mean Reversion",
        "mode": "mean_reversion",
        "buy_threshold": 0.60,
        "sell_threshold": 0.40,
        "blurb": "Snap-back sleeve when price stretches vs a short mean (Z-score).",
    },
    {
        "id": "multi_factor",
        "label": "Multi-Factor",
        "mode": "multi_factor",
        "buy_threshold": 0.60,
        "sell_threshold": 0.40,
        "blurb": "Technical + macro blend from the existing fusion path.",
    },
]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return number


def _rank_score(metrics: dict[str, Any]) -> float:
    """Predeclared ranking blend — not tuned on the test fold."""
    sharpe = _finite(metrics.get("sharpe"))
    net = _finite(metrics.get("net_return_pct")) / 100.0
    dd = abs(_finite(metrics.get("max_drawdown_pct"), _finite(metrics.get("max_drawdown")))) / 100.0
    win = _finite(metrics.get("win_rate"), 0.5)
    return float(0.40 * sharpe + 0.30 * net + 0.20 * (1.0 - min(dd, 1.0)) + 0.10 * win)


class BacktestRunBody(BaseModel):
    ticker: str = Field(default="RELIANCE.NS", min_length=1)
    label: str = ""
    start: Optional[str] = None
    end: Optional[str] = None
    timeframe: str = Field(default="1d")
    strategy_mode: str = Field(default="technical_only")
    initial_capital: float = Field(default=100_000.0, gt=0, le=100_000_000)
    buy_threshold: float = Field(default=CORE_BUY_THRESHOLD, ge=0.5, le=0.95)
    sell_threshold: float = Field(default=CORE_SELL_THRESHOLD, ge=0.05, le=0.5)
    base_position_size: float = Field(default=CORE_RISK_CONFIG.base_position_size, gt=0, le=1)
    min_position_size: float = Field(default=CORE_RISK_CONFIG.min_position_size, ge=0, le=1)
    hard_vol_stop: float = Field(default=CORE_RISK_CONFIG.hard_vol_stop, gt=0, le=1)
    hard_drawdown_stop: float = Field(default=CORE_RISK_CONFIG.hard_drawdown_stop, gt=0, le=1)
    cost_segment: str = Field(default="equity_delivery")
    brokerage_rate: Optional[float] = Field(default=None, ge=0, le=0.01)
    stt_buy_rate: Optional[float] = Field(default=None, ge=0, le=0.01)
    stt_sell_rate: Optional[float] = Field(default=None, ge=0, le=0.01)
    slippage_bps: Optional[float] = Field(default=None, ge=0, le=100)
    save: bool = True


def _cost_from_body(body: BacktestRunBody) -> IndiaEquityCostConfig:
    overrides: dict[str, float] = {}
    if body.brokerage_rate is not None:
        overrides["brokerage_rate"] = float(body.brokerage_rate)
    if body.stt_buy_rate is not None:
        overrides["stt_buy_rate"] = float(body.stt_buy_rate)
    if body.stt_sell_rate is not None:
        overrides["stt_sell_rate"] = float(body.stt_sell_rate)
    if body.slippage_bps is not None:
        overrides["slippage_bps"] = float(body.slippage_bps)
    return cost_config_for_segment(body.cost_segment, **overrides)


@router.get("/defaults")
def backtest_defaults() -> dict:
    delivery = IndiaEquityCostConfig.equity_delivery()
    return {
        "buy_threshold": CORE_BUY_THRESHOLD,
        "sell_threshold": CORE_SELL_THRESHOLD,
        "initial_capital": 100_000.0,
        "timeframe": "1d",
        "strategy_modes": ["technical_only", "multi_factor", "momentum", "mean_reversion"],
        "strategy_presets": STRATEGY_PRESETS,
        "cost_segments": ["equity_delivery", "equity_intraday", "zero"],
        "costs": {
            "brokerage_rate": delivery.brokerage_rate,
            "stt_buy_rate": delivery.stt_buy_rate,
            "stt_sell_rate": delivery.stt_sell_rate,
            "slippage_bps": delivery.slippage_bps,
            "exchange_rate": delivery.exchange_rate,
            "gst_rate": delivery.gst_rate,
        },
        "risk": {
            "base_position_size": CORE_RISK_CONFIG.base_position_size,
            "min_position_size": CORE_RISK_CONFIG.min_position_size,
            "hard_vol_stop": CORE_RISK_CONFIG.hard_vol_stop,
            "hard_drawdown_stop": CORE_RISK_CONFIG.hard_drawdown_stop,
            "vol_window": CORE_RISK_CONFIG.vol_window,
            "drawdown_window": CORE_RISK_CONFIG.drawdown_window,
        },
        "bounds": {
            "buy_threshold": [0.5, 0.95],
            "sell_threshold": [0.05, 0.5],
            "note": "Manual what-if only. Optimizer loops against test folds remain disabled.",
        },
        "freshness": "historical",
        "lookahead_note": (
            "Purged walk-forward / next-open fills are preserved by the existing engine. "
            "Date filters truncate available history before the run; they do not peek at future bars."
        ),
    }


@router.get("/symbols")
def backtest_symbols() -> dict:
    store = get_market_store()
    coverage = []
    try:
        # Prefer stored candle coverage when available.
        from market_data.store import MarketStore

        assert isinstance(store, MarketStore)
        rows = store._conn.execute(
            """
            SELECT symbol, timeframe, COUNT(*) AS bars, MIN(bar_time) AS start, MAX(bar_time) AS end
            FROM candles
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
            """
        ).fetchall()
        coverage = [
            {
                "symbol": r["symbol"],
                "timeframe": r["timeframe"],
                "bars": r["bars"],
                "start": r["start"],
                "end": r["end"],
            }
            for r in rows
        ]
    except Exception as exc:  # noqa: BLE001
        return {"symbols": [], "coverage": [], "detail": str(exc)}
    stored = {row["symbol"] for row in coverage}
    universe = sorted(set(NIFTY50_TICKERS) | stored)
    return {
        "symbols": universe,
        "stored_symbols": sorted(stored),
        "coverage": coverage,
        "freshness": "historical",
        "detail": (
            "NIFTY50 selection list plus any locally stored symbols. "
            "Missing history is downloaded from Yahoo (delayed) on first run."
        ),
    }


@router.post("/run")
def backtest_run(body: BacktestRunBody) -> dict:
    if body.buy_threshold <= body.sell_threshold:
        raise HTTPException(status_code=400, detail="buy_threshold must be greater than sell_threshold")
    ticker = body.ticker.strip().upper()
    cost_cfg = _cost_from_body(body)
    risk = RiskConfig(
        vol_window=CORE_RISK_CONFIG.vol_window,
        drawdown_window=CORE_RISK_CONFIG.drawdown_window,
        max_vol_for_full_risk=CORE_RISK_CONFIG.max_vol_for_full_risk,
        hard_vol_stop=float(body.hard_vol_stop),
        hard_drawdown_stop=float(body.hard_drawdown_stop),
        base_position_size=float(body.base_position_size),
        min_position_size=float(body.min_position_size),
    )
    cfg = FourModelConfig(
        initial_capital=float(body.initial_capital),
        buy_threshold=float(body.buy_threshold),
        sell_threshold=float(body.sell_threshold),
        strategy_mode=body.strategy_mode,
        risk_config=risk,
        cost_config=cost_cfg,
        manual_overrides=True,
    )
    try:
        result = run_four_model_pipeline(ticker, cfg, start=body.start, end=body.end)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Backtest failed: {exc}") from exc

    summary = summarize_research({**result, "ticker": ticker})
    summary = dict(summary)
    summary["params"] = {
        "ticker": ticker,
        "start": body.start,
        "end": body.end,
        "timeframe": body.timeframe,
        "strategy_mode": body.strategy_mode,
        "initial_capital": body.initial_capital,
        "buy_threshold": body.buy_threshold,
        "sell_threshold": body.sell_threshold,
        "risk": {
            "base_position_size": body.base_position_size,
            "min_position_size": body.min_position_size,
            "hard_vol_stop": body.hard_vol_stop,
            "hard_drawdown_stop": body.hard_drawdown_stop,
        },
        "costs": {
            "segment": cost_cfg.segment,
            "brokerage_rate": cost_cfg.brokerage_rate,
            "stt_buy_rate": cost_cfg.stt_buy_rate,
            "stt_sell_rate": cost_cfg.stt_sell_rate,
            "slippage_bps": cost_cfg.slippage_bps,
        },
    }
    summary["freshness"] = "historical"
    summary["data_mode"] = "HISTORICAL"
    summary["broker_execution"] = False

    # Enrich equity / drawdown series for charts when present.
    backtest = result.get("backtest") or {}
    equity = backtest.get("equity_curve")
    if equity is not None:
        try:
            import pandas as pd

            if isinstance(equity, pd.Series) and not equity.empty:
                summary["equity_curve"] = [
                    {"date": jsonable(idx), "equity": jsonable(val)} for idx, val in equity.items()
                ]
                peak = equity.cummax()
                dd = (equity / peak) - 1.0
                summary["drawdown_curve"] = [
                    {"date": jsonable(idx), "drawdown": jsonable(val)} for idx, val in dd.items()
                ]
        except Exception:
            pass

    run_id = None
    if body.save:
        run_id = get_scan_store().save_backtest(
            ticker=ticker,
            label=body.label or f"{ticker} backtest",
            params=summary["params"],
            summary=summary,
            freshness="historical",
        )
        summary["saved_run_id"] = run_id

    # Keep research cache warm for AI explain / reliability panels.
    try:
        from api.routes_research import _LAST_RESULT

        _LAST_RESULT["summary"] = summary
        _LAST_RESULT["ticker"] = ticker
        _LAST_RESULT["raw"] = {
            "backtest": result.get("backtest"),
            "reliability": result.get("reliability"),
            "walk_forward_fold_scores": result.get("walk_forward_fold_scores"),
            "prices": result.get("prices"),
            "test_df": result.get("test_df"),
        }
    except Exception:
        pass

    return summary


class SimulateStrategiesBody(BaseModel):
    ticker: str = Field(default="RELIANCE.NS", min_length=1)
    start: Optional[str] = None
    end: Optional[str] = None
    initial_capital: float = Field(default=100_000.0, gt=0, le=100_000_000)
    cost_segment: str = Field(default="equity_delivery")
    save: bool = True


@router.post("/simulate-strategies")
def backtest_simulate_strategies(body: SimulateStrategiesBody) -> dict:
    """
    Run the five curated strategy sleeves on one ticker and explain the best setup.

    Ranking uses a predeclared score. This is not parameter search on test folds.
    """
    ticker = body.ticker.strip().upper()
    cost_cfg = cost_config_for_segment(body.cost_segment)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for preset in STRATEGY_PRESETS:
        cfg = FourModelConfig(
            initial_capital=float(body.initial_capital),
            buy_threshold=float(preset["buy_threshold"]),
            sell_threshold=float(preset["sell_threshold"]),
            strategy_mode=str(preset["mode"]),
            risk_config=CORE_RISK_CONFIG,
            cost_config=cost_cfg,
            manual_overrides=True,
        )
        try:
            raw = run_four_model_pipeline(ticker, cfg, start=body.start, end=body.end)
            summary = dict(summarize_research({**raw, "ticker": ticker}))
            metrics = dict(summary.get("metrics") or {})
            latest = dict(summary.get("latest") or {})
            score = _rank_score(metrics)
            entry = {
                "strategy_id": preset["id"],
                "label": preset["label"],
                "blurb": preset["blurb"],
                "mode": preset["mode"],
                "buy_threshold": preset["buy_threshold"],
                "sell_threshold": preset["sell_threshold"],
                "rank_score": score,
                "signal": latest.get("signal"),
                "proba": latest.get("proba"),
                "no_trade": latest.get("no_trade"),
                "metrics": metrics,
                "latest": latest,
                "reliability": summary.get("reliability"),
                "trades_tail": summary.get("trades_tail"),
                "equity_tail": summary.get("equity_tail"),
                "disclaimer": summary.get("disclaimer"),
            }
            if body.save:
                entry["saved_run_id"] = get_scan_store().save_backtest(
                    ticker=ticker,
                    label=f"{ticker} · {preset['label']}",
                    params={
                        "ticker": ticker,
                        "strategy_id": preset["id"],
                        "mode": preset["mode"],
                        "buy_threshold": preset["buy_threshold"],
                        "sell_threshold": preset["sell_threshold"],
                        "initial_capital": body.initial_capital,
                        "start": body.start,
                        "end": body.end,
                    },
                    summary={**summary, "strategy_id": preset["id"], "rank_score": score},
                    freshness="historical",
                )
            results.append(entry)
        except Exception as exc:  # noqa: BLE001
            errors.append({"strategy_id": str(preset["id"]), "label": str(preset["label"]), "error": str(exc)})

    results.sort(key=lambda row: float(row.get("rank_score") or -1e9), reverse=True)
    for idx, row in enumerate(results, start=1):
        row["rank"] = idx

    best = results[0] if results else None
    buy_analysis: dict[str, Any] = {
        "available": False,
        "detail": "No successful strategy runs to explain.",
    }
    if best:
        latest = dict(best.get("latest") or {})
        signal = str(latest.get("signal") or "HOLD").upper()
        why_buy: list[str] = []
        why_not: list[str] = []
        if signal == "BUY":
            why_buy.append(
                f"{best['label']} currently issues BUY with model probability {latest.get('proba')}."
            )
        else:
            why_not.append(
                f"Best historical sleeve is {best['label']}, but the latest signal is {signal} — not an active BUY."
            )
        if latest.get("no_trade"):
            why_not.append("Risk engine NoTrade gate is active — new BUYs are blocked.")
        if latest.get("ma50") is not None and latest.get("close") is not None:
            try:
                if float(latest["close"]) >= float(latest["ma50"]):
                    why_buy.append("Close is at/above MA50 (trend confirmation for technical sleeves).")
                else:
                    why_not.append("Close is below MA50 — trend confirmation is weak.")
            except (TypeError, ValueError):
                pass
        metrics = dict(best.get("metrics") or {})
        if metrics.get("net_return_pct") is not None:
            why_buy.append(
                f"On historical simulation, net return was {metrics.get('net_return_pct')}% "
                f"with Sharpe {metrics.get('sharpe')} (delayed/historical data)."
            )
        if metrics.get("max_drawdown_pct") is not None or metrics.get("max_drawdown") is not None:
            why_not.append(
                f"Observe drawdown risk: max DD {metrics.get('max_drawdown_pct') or metrics.get('max_drawdown')}."
            )

        explanation = None
        try:
            from reasoning.service import ExplainRequest, explain_decision

            explanation = explain_decision(
                ExplainRequest(
                    ticker=ticker,
                    latest=latest,
                    reliability=dict(best.get("reliability") or {}),
                    metrics=metrics,
                    strategy_mode=str(best.get("mode")),
                    buy_threshold=float(best.get("buy_threshold") or 0.6),
                    sell_threshold=float(best.get("sell_threshold") or 0.4),
                    use_llm=True,
                )
            )
        except Exception as exc:  # noqa: BLE001
            explanation = {"summary": f"Explanation fallback unavailable: {exc}", "mode": "error"}

        buy_analysis = {
            "available": True,
            "ticker": ticker,
            "best_strategy": best.get("label"),
            "best_strategy_id": best.get("strategy_id"),
            "latest_signal": signal,
            "recommendation": (
                "BUY candidate under best sleeve"
                if signal == "BUY" and not latest.get("no_trade")
                else "Do not treat as an active BUY right now"
            ),
            "why_buy": why_buy,
            "why_caution": why_not,
            "explanation": explanation,
            "note": (
                "AI explains structured model/risk outputs only. It cannot override signals. "
                "History is delayed/Yahoo research data — not verified live."
            ),
        }

        # Warm research cache for the contextual AI panel.
        try:
            from api.routes_research import _LAST_RESULT

            _LAST_RESULT["summary"] = {
                "ticker": ticker,
                "latest": latest,
                "metrics": metrics,
                "reliability": best.get("reliability"),
                "strategy_mode": best.get("mode"),
                "thresholds": {
                    "buy_threshold": best.get("buy_threshold"),
                    "sell_threshold": best.get("sell_threshold"),
                },
            }
            _LAST_RESULT["ticker"] = ticker
        except Exception:
            pass

    return {
        "ticker": ticker,
        "freshness": "historical",
        "data_mode": "HISTORICAL",
        "broker_execution": False,
        "ranking_note": (
            "Strategies ranked by predeclared score: 0.4·Sharpe + 0.3·net return "
            "+ 0.2·(1−|drawdown|) + 0.1·win rate. Not optimized on the test fold."
        ),
        "strategies": results,
        "errors": errors,
        "best": best,
        "buy_analysis": buy_analysis,
    }


@router.get("/runs")
def backtest_runs(limit: int = 30) -> dict:
    return {"runs": get_scan_store().list_backtests(limit=limit), "freshness": "historical"}


@router.get("/runs/{run_id}")
def backtest_run_get(run_id: int) -> dict:
    row = get_scan_store().get_backtest(run_id)
    if not row:
        raise HTTPException(status_code=404, detail="Backtest run not found")
    return row


class CompareBody(BaseModel):
    run_ids: list[int] = Field(default_factory=list)


@router.post("/compare")
def backtest_compare(body: CompareBody) -> dict:
    store = get_scan_store()
    rows: list[dict[str, Any]] = []
    for rid in (body.run_ids or [])[:6]:
        item = store.get_backtest(int(rid))
        if item:
            metrics = (item.get("summary") or {}).get("metrics") or {}
            rows.append(
                {
                    "id": item["id"],
                    "ticker": item["ticker"],
                    "label": item["label"],
                    "created_at": item["created_at"],
                    "net_return_pct": metrics.get("net_return_pct"),
                    "sharpe": metrics.get("sharpe"),
                    "max_drawdown_pct": metrics.get("max_drawdown_pct") or metrics.get("max_drawdown"),
                    "cagr_pct": metrics.get("cagr_pct"),
                    "win_rate": metrics.get("win_rate"),
                    "params": item.get("params"),
                }
            )
    return {
        "comparisons": rows,
        "note": "Comparison of saved research runs only. This endpoint does not search parameters on test folds.",
        "freshness": "historical",
    }

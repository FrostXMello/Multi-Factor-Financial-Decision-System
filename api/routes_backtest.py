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

router = APIRouter(prefix="/api/backtest", tags=["backtesting"])


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
    symbols = sorted({row["symbol"] for row in coverage})
    return {
        "symbols": symbols,
        "coverage": coverage,
        "freshness": "historical",
        "detail": "Symbols with stored OHLCV. Research runs can also download missing Yahoo history.",
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

"""Convert research/paper payloads into JSON-safe dictionaries."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        for col in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[col]):
                frame[col] = pd.to_datetime(frame[col]).map(
                    lambda x: None if pd.isna(x) else pd.Timestamp(x).isoformat()
                )
        records = frame.where(pd.notnull(frame), None).to_dict(orient="records")
        return [{key: jsonable(item) for key, item in row.items()} for row in records]
    if isinstance(value, pd.Series):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    return str(value)


def summarize_research(result: dict) -> dict:
    """Keep the API payload bounded: KPIs, latest signal, reliability, trade sample."""
    test_df = result.get("test_df")
    latest = {}
    signals_tail = []
    if isinstance(test_df, pd.DataFrame) and not test_df.empty:
        row = test_df.iloc[-1]
        latest = {
            "date": jsonable(row.get("Date")),
            "signal": str(row.get("Signal", "HOLD")),
            "proba": jsonable(row.get("Proba")),
            "close": jsonable(row.get("Close")),
            "risk_score": jsonable(row.get("RiskScore")),
            "position_size": jsonable(row.get("PositionSize")),
        }
        cols = [c for c in ("Date", "Close", "Signal", "Proba", "RiskScore") if c in test_df.columns]
        signals_tail = jsonable(test_df[cols].tail(30))

    backtest = result.get("backtest") or {}
    equity = backtest.get("equity_curve")
    equity_tail = []
    if isinstance(equity, pd.Series) and not equity.empty:
        equity_tail = [
            {"date": jsonable(idx), "equity": jsonable(val)}
            for idx, val in equity.tail(60).items()
        ]
    elif isinstance(equity, pd.DataFrame) and not equity.empty:
        equity_tail = jsonable(equity.tail(60))

    trades_df = backtest.get("trades_df")
    trades = jsonable(trades_df.tail(20)) if isinstance(trades_df, pd.DataFrame) else []

    reliability = result.get("reliability") or {}
    reliability_public = {
        key: jsonable(reliability.get(key))
        for key in (
            "evaluation",
            "decision_method",
            "insample_accuracy",
            "oos_metrics",
            "holdout_accuracy",
            "folds",
            "permutation_importance",
            "note",
            "message",
        )
        if key in reliability
    }

    return {
        "ticker": result.get("ticker"),
        "strategy_mode": result.get("strategy_mode"),
        "latest": latest,
        "metrics": {
            "technical_test_accuracy": jsonable(result.get("technical_test_accuracy")),
            "walk_forward_mean_accuracy": jsonable(result.get("walk_forward_mean_accuracy")),
            "walk_forward_fold_scores": jsonable(result.get("walk_forward_fold_scores")),
            "trades": jsonable(result.get("trades")),
            "trades_per_day": jsonable(result.get("trades_per_day")),
            "final_portfolio_value": jsonable(backtest.get("final_portfolio_value")),
            "net_return_pct": jsonable(backtest.get("total_return_pct")),
            "gross_return_pct": jsonable(backtest.get("gross_return_pct")),
            "total_transaction_costs": jsonable(backtest.get("total_transaction_costs")),
            "max_drawdown": jsonable(backtest.get("max_drawdown")),
            "sharpe": jsonable(backtest.get("sharpe")),
        },
        "signals_tail": signals_tail,
        "equity_tail": equity_tail,
        "trades_tail": trades,
        "reliability": reliability_public,
        "disclaimer": (
            "Research output from the existing Python pipeline. "
            "Yahoo history is delayed. This is not live execution advice."
        ),
    }

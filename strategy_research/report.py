"""Assemble a research analytics report from existing pipeline outputs."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from strategy_research.portfolio import portfolio_analytics
from strategy_research.regimes import attach_causal_regimes_to_equity, classify_regimes
from strategy_research.robustness import walk_forward_robustness
from strategy_research.strategy_compare import compare_strategies


def build_research_analytics_report(
    *,
    ticker: str,
    price_history: Optional[pd.DataFrame] = None,
    backtest: Optional[dict[str, Any]] = None,
    comparison_backtests: Optional[dict[str, dict[str, Any]]] = None,
    reliability: Optional[dict[str, Any]] = None,
    positions: Optional[list[dict[str, Any]]] = None,
    cash: Optional[float] = None,
    lookback: int = 60,
) -> dict[str, Any]:
    unavailable: list[str] = []
    regimes_table = None
    regime_eval = None
    if price_history is None or not isinstance(price_history, pd.DataFrame) or price_history.empty:
        unavailable.append("price_history")
    else:
        regimes_table = classify_regimes(price_history, lookback=lookback)
        if backtest and isinstance(backtest.get("equity_curve"), pd.DataFrame):
            _merged, regime_eval = attach_causal_regimes_to_equity(
                backtest["equity_curve"],
                price_history,
                lookback=lookback,
            )
        else:
            unavailable.append("strategy_equity_for_regime_eval")

    compare = None
    if comparison_backtests:
        compare = compare_strategies(comparison_backtests)
    elif backtest:
        compare = compare_strategies({ticker or "strategy": backtest})
    else:
        unavailable.append("strategy_backtests")

    folds = None
    fold_scores = None
    if isinstance(reliability, dict):
        folds = reliability.get("folds")
        fold_scores = reliability.get("walk_forward_fold_scores")
        if fold_scores is None and isinstance(reliability.get("oos_metrics"), dict):
            # Single aggregate is not enough for fold robustness.
            pass
    robustness = walk_forward_robustness(fold_scores=fold_scores, folds_frame=folds if isinstance(folds, pd.DataFrame) else None)

    portfolio = None
    if positions is not None:
        equity = backtest.get("equity_curve") if isinstance(backtest, dict) else None
        portfolio = portfolio_analytics(
            positions,
            cash=cash,
            equity_curve=equity if isinstance(equity, pd.DataFrame) else None,
            initial_capital=backtest.get("initial_capital") if isinstance(backtest, dict) else None,
        )
    else:
        unavailable.append("portfolio_positions")

    weaknesses = []
    if regime_eval:
        weaknesses.extend(regime_eval.get("weaknesses") or [])
    if compare:
        weaknesses.extend(compare.get("weaknesses") or [])
    if robustness:
        weaknesses.extend(robustness.get("weaknesses") or [])

    latest_regime = None
    if regimes_table is not None and not regimes_table.empty:
        latest_regime = {
            "date": str(regimes_table["Date"].iloc[-1]),
            "regime_end_of_day": regimes_table["regime_end_of_day"].iloc[-1],
            "trailing_return": float(regimes_table["trailing_return"].iloc[-1])
            if pd.notna(regimes_table["trailing_return"].iloc[-1])
            else None,
        }

    return {
        "ticker": (ticker or "unavailable").upper(),
        "regimes": {
            "available": regimes_table is not None,
            "latest": latest_regime,
            "counts": regimes_table["regime_end_of_day"].value_counts().to_dict() if regimes_table is not None else {},
            "lookahead_safe": True,
            "lookback": lookback,
        },
        "regime_performance": regime_eval,
        "strategy_comparison": compare,
        "portfolio": portfolio,
        "robustness": robustness,
        "weaknesses": weaknesses,
        "unavailable": unavailable,
        "disclaimer": (
            "Analytics use historical/stored inputs only. Insufficient samples are labelled. "
            "No fabricated metrics. Walk-forward folds are not used to re-optimize parameters."
        ),
    }

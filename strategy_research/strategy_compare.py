"""Compare strategies on net returns after costs. Does not re-optimize parameters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backtest_analytics import annualized_volatility, equity_daily_returns, sharpe_ratio, sortino_ratio


def _from_backtest(name: str, backtest: dict[str, Any]) -> dict[str, Any]:
    eq = backtest.get("equity_curve")
    if not isinstance(eq, pd.DataFrame) or eq.empty or "PortfolioValue" not in eq.columns:
        return {
            "name": name,
            "available": False,
            "detail": "insufficient_data",
            "net_return_pct": None,
            "gross_return_pct": None,
            "total_transaction_costs": None,
        }
    rets = equity_daily_returns(eq)
    net = backtest.get("net_return_pct", backtest.get("total_return_pct"))
    gross = backtest.get("gross_return_pct")
    costs = backtest.get("total_transaction_costs")
    max_dd = backtest.get("max_drawdown_pct")
    available = len(rets) >= 2 and net is not None and np.isfinite(float(net))
    return {
        "name": name,
        "available": bool(available),
        "detail": "ok" if available else "insufficient_data",
        "net_return_pct": float(net) if net is not None and np.isfinite(float(net)) else None,
        "gross_return_pct": float(gross) if gross is not None and np.isfinite(float(gross)) else None,
        "total_transaction_costs": float(costs) if costs is not None and np.isfinite(float(costs)) else None,
        "max_drawdown_pct": float(max_dd) if max_dd is not None and np.isfinite(float(max_dd)) else None,
        "sharpe": sharpe_ratio(rets) if len(rets) >= 2 else None,
        "sortino": sortino_ratio(rets) if len(rets) >= 2 else None,
        "volatility_ann": annualized_volatility(rets) if len(rets) >= 2 else None,
        "observations": int(len(rets)),
        "final_portfolio_value": float(backtest.get("final_portfolio_value"))
        if backtest.get("final_portfolio_value") is not None
        else None,
    }


def compare_strategies(strategies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    `strategies` maps display name -> backtest result dict from the existing engine.

    Ranking uses net return after transaction costs when available.
    """
    rows = [_from_backtest(name, payload or {}) for name, payload in (strategies or {}).items()]
    usable = [row for row in rows if row.get("available") and row.get("net_return_pct") is not None]
    ranked = sorted(usable, key=lambda r: float(r["net_return_pct"]), reverse=True)
    weaknesses = []
    for row in rows:
        if not row.get("available"):
            weaknesses.append(f"{row['name']}: insufficient data for comparison.")
            continue
        if row.get("net_return_pct") is not None and float(row["net_return_pct"]) < 0:
            weaknesses.append(f"{row['name']}: negative net return after costs.")
        if row.get("gross_return_pct") is not None and row.get("net_return_pct") is not None:
            gap = float(row["gross_return_pct"]) - float(row["net_return_pct"])
            if gap > 5.0:
                weaknesses.append(
                    f"{row['name']}: costs reduced return by {gap:.2f} percentage points (gross vs net)."
                )
    return {
        "strategies": rows,
        "ranking_by_net_return": [r["name"] for r in ranked],
        "leader": ranked[0]["name"] if ranked else None,
        "weaknesses": weaknesses,
        "note": (
            "Comparison uses stored/backtest net returns after the existing cost model. "
            "This report does not search parameters on the evaluation sample."
        ),
    }

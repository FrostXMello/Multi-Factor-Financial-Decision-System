"""Portfolio-level analytics from explicit holdings. No fabricated exposures."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from backtest_analytics import (
    annualized_volatility,
    compute_drawdown_series,
    equity_daily_returns,
    sharpe_ratio,
    sortino_ratio,
)
from sector_universe import STOCKS_BY_SECTOR


def _sector_lookup() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for sector, tickers in STOCKS_BY_SECTOR.items():
        for ticker in tickers:
            mapping[str(ticker).upper()] = sector
    return mapping


def portfolio_analytics(
    positions: list[dict[str, Any]],
    *,
    cash: Optional[float] = None,
    equity_curve: Optional[pd.DataFrame] = None,
    initial_capital: Optional[float] = None,
) -> dict[str, Any]:
    """
    Allocation, sector exposure, concentration and optional risk-adjusted stats.

    Each position should include `symbol` and either `market_value` or
    (`quantity` and `mark`/`price`). Unknown sectors are labelled UNAVAILABLE.
    """
    sector_map = _sector_lookup()
    rows = []
    for raw in positions or []:
        symbol = str(raw.get("symbol") or raw.get("ticker") or "").strip().upper()
        if not symbol:
            continue
        market_value = raw.get("market_value")
        if market_value is None:
            qty = raw.get("quantity")
            mark = raw.get("mark", raw.get("price", raw.get("avg_entry")))
            try:
                market_value = float(qty) * float(mark) if qty is not None and mark is not None else None
            except (TypeError, ValueError):
                market_value = None
        try:
            mv = float(market_value) if market_value is not None else None
        except (TypeError, ValueError):
            mv = None
        if mv is None or not np.isfinite(mv):
            rows.append(
                {
                    "symbol": symbol,
                    "market_value": None,
                    "weight": None,
                    "sector": sector_map.get(symbol, "UNAVAILABLE"),
                    "available": False,
                }
            )
            continue
        rows.append(
            {
                "symbol": symbol,
                "market_value": float(mv),
                "weight": None,
                "sector": sector_map.get(symbol, "UNAVAILABLE"),
                "available": True,
            }
        )

    invested = float(sum(r["market_value"] for r in rows if r["available"]))
    cash_value = float(cash) if cash is not None and np.isfinite(float(cash)) else None
    nav = invested + cash_value if cash_value is not None else invested
    if nav <= 0:
        return {
            "available": False,
            "detail": "insufficient_data",
            "positions": rows,
            "allocation": [],
            "sector_exposure": [],
            "concentration": {"hhi": None, "top_weight": None, "n_positions": 0},
            "risk_adjusted": {"available": False, "detail": "insufficient_data"},
        }

    for row in rows:
        if row["available"]:
            row["weight"] = float(row["market_value"]) / nav

    allocation = [
        {
            "symbol": r["symbol"],
            "market_value": r["market_value"],
            "weight": r["weight"],
            "sector": r["sector"],
        }
        for r in rows
        if r["available"]
    ]
    if cash_value is not None and cash_value != 0:
        allocation.append(
            {
                "symbol": "CASH",
                "market_value": cash_value,
                "weight": cash_value / nav,
                "sector": "CASH",
            }
        )

    sector_totals: dict[str, float] = {}
    for item in allocation:
        sector_totals[item["sector"]] = sector_totals.get(item["sector"], 0.0) + float(item["weight"])
    sector_exposure = [
        {"sector": sector, "weight": weight, "available": sector != "UNAVAILABLE"}
        for sector, weight in sorted(sector_totals.items(), key=lambda x: -x[1])
    ]

    weights = np.array([float(r["weight"]) for r in allocation if r["symbol"] != "CASH"], dtype=float)
    hhi = float(np.sum(weights ** 2)) if len(weights) else None
    top_weight = float(weights.max()) if len(weights) else None

    risk_adjusted = _equity_risk_block(equity_curve, initial_capital=initial_capital)

    return {
        "available": True,
        "nav": nav,
        "invested": invested,
        "cash": cash_value,
        "positions": rows,
        "allocation": allocation,
        "sector_exposure": sector_exposure,
        "concentration": {
            "hhi": hhi,
            "top_weight": top_weight,
            "n_positions": int(len(weights)),
            "interpretation": (
                "HHI near 1.0 means concentrated; near 1/n means equal-weight among holdings."
                if hhi is not None
                else "unavailable"
            ),
        },
        "risk_adjusted": risk_adjusted,
        "unavailable_sectors": [r["symbol"] for r in rows if r["sector"] == "UNAVAILABLE"],
    }


def _equity_risk_block(equity_curve: Optional[pd.DataFrame], *, initial_capital: Optional[float]) -> dict[str, Any]:
    if equity_curve is None or not isinstance(equity_curve, pd.DataFrame) or equity_curve.empty:
        return {"available": False, "detail": "insufficient_data"}
    if "Date" not in equity_curve.columns or "PortfolioValue" not in equity_curve.columns:
        return {"available": False, "detail": "equity_curve missing Date/PortfolioValue"}
    eq = equity_curve.copy()
    eq["Date"] = pd.to_datetime(eq["Date"])
    eq = eq.sort_values("Date")
    rets = equity_daily_returns(eq)
    if len(rets) < 2:
        return {"available": False, "detail": "insufficient_data"}
    dd = compute_drawdown_series(eq)
    max_dd = float(dd["DrawdownPct"].min()) if len(dd) else None
    final = float(eq["PortfolioValue"].iloc[-1])
    initial = float(initial_capital) if initial_capital not in (None, 0) else float(eq["PortfolioValue"].iloc[0])
    total_return = final / initial - 1.0 if initial else None
    return {
        "available": True,
        "volatility_ann": annualized_volatility(rets),
        "sharpe": sharpe_ratio(rets),
        "sortino": sortino_ratio(rets),
        "max_drawdown_pct": max_dd,
        "total_return": total_return,
        "observations": int(len(rets)),
    }

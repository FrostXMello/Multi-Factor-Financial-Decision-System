"""
Professional backtest analytics: performance, risk, risk-adjusted ratios, trade stats, time-series views.
Designed for fast vectorized computation on equity curves produced by `backtest.py`.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _years_between(d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    days = (d1 - d0).days
    return float(days) / 365.25 if days > 0 else 0.0


def compute_cagr(initial_capital: float, final_value: float, years: float) -> float:
    if years <= 0 or initial_capital <= 0 or final_value <= 0:
        return float("nan")
    return float((final_value / initial_capital) ** (1.0 / years) - 1.0)


def equity_daily_returns(equity_curve: pd.DataFrame) -> pd.Series:
    df = equity_curve[["Date", "PortfolioValue"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df["PortfolioValue"].astype(float).pct_change().dropna()


def compute_drawdown_series(equity_curve: pd.DataFrame) -> pd.DataFrame:
    eq = equity_curve.copy()
    eq["Date"] = pd.to_datetime(eq["Date"])
    v = eq["PortfolioValue"].astype(float)
    peak = v.cummax()
    dd = v / peak - 1.0
    out = pd.DataFrame({"Date": eq["Date"], "DrawdownPct": dd * 100.0, "Equity": v})
    return out


def annualized_volatility(daily_returns: pd.Series) -> float:
    dr = daily_returns.dropna()
    if len(dr) < 2:
        return float("nan")
    return float(dr.std() * np.sqrt(252))


def sharpe_ratio(daily_returns: pd.Series, rf_daily: float = 0.0) -> float:
    dr = daily_returns.dropna() - rf_daily
    if len(dr) < 2 or dr.std() <= 0:
        return float("nan")
    return float(np.sqrt(252) * dr.mean() / dr.std())


def sortino_ratio(daily_returns: pd.Series, rf_daily: float = 0.0) -> float:
    dr = daily_returns.dropna() - rf_daily
    if len(dr) < 2:
        return float("nan")
    downside = dr[dr < 0]
    if len(downside) < 2:
        return float("nan")
    ds = float(downside.std())
    if ds <= 0 or not np.isfinite(ds):
        return float("nan")
    return float(np.sqrt(252) * dr.mean() / ds)


def average_drawdown_pct(dd_decimal: pd.Series) -> float:
    """Mean depth of underwater observations (reported as positive %)."""
    s = dd_decimal.dropna()
    under = s[s < 0]
    if len(under) == 0:
        return 0.0
    return float(-under.mean() * 100.0)


def consecutive_streaks(wins: np.ndarray) -> tuple[int, int]:
    if wins.size == 0:
        return 0, 0
    best_w, best_l = 0, 0
    cw, cl = 0, 0
    for w in wins:
        if w:
            cw += 1
            cl = 0
            best_w = max(best_w, cw)
        else:
            cl += 1
            cw = 0
            best_l = max(best_l, cl)
    return int(best_w), int(best_l)


def trade_statistics(trades_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "total_trades": 0,
        "win_rate": float("nan"),
        "avg_win": float("nan"),
        "avg_loss": float("nan"),
        "profit_factor": float("nan"),
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "expectancy": float("nan"),
        "gross_profit": 0.0,
        "gross_loss": 0.0,
    }
    if trades_df is None or len(trades_df) == 0:
        return out

    pnl = trades_df["PnL"].astype(float)
    if "Win" in trades_df.columns:
        wins = trades_df["Win"].values.astype(bool)
    else:
        wins = (pnl > 0).values
    out["total_trades"] = int(len(trades_df))
    n_w = int(wins.sum())
    out["win_rate"] = float(n_w / len(trades_df)) if len(trades_df) else float("nan")

    win_pn = pnl[wins]
    loss_pn = pnl[~wins]
    out["avg_win"] = float(win_pn.mean()) if len(win_pn) else float("nan")
    out["avg_loss"] = float(loss_pn.mean()) if len(loss_pn) else float("nan")
    out["gross_profit"] = float(win_pn.sum()) if len(win_pn) else 0.0
    out["gross_loss"] = float(loss_pn.sum()) if len(loss_pn) else 0.0
    gl = out["gross_loss"]
    out["profit_factor"] = float(out["gross_profit"] / abs(gl)) if gl < 0 and np.isfinite(gl) else float("inf")
    if gl >= 0 and out["gross_profit"] > 0:
        out["profit_factor"] = float("inf")
    if out["gross_profit"] == 0 and gl >= 0:
        out["profit_factor"] = float("nan")

    out["expectancy"] = float(pnl.mean()) if len(pnl) else float("nan")
    mw, ml = consecutive_streaks(wins)
    out["max_consecutive_wins"] = mw
    out["max_consecutive_losses"] = ml
    return out


def monthly_returns_matrix(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Year × month percentage returns for heatmap."""
    df = equity_curve[["Date", "PortfolioValue"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    month_end = df["PortfolioValue"].resample("M").last()
    mret = month_end.pct_change().dropna() * 100.0
    if mret.empty:
        return pd.DataFrame()

    s = mret.copy()
    s.index = pd.to_datetime(s.index)
    s = s.to_frame("ret")
    s["Year"] = s.index.year
    s["Month"] = s.index.month
    pivot = s.pivot_table(index="Year", columns="Month", values="ret", aggfunc="first")
    pivot.columns = [pd.Timestamp(2000, int(c), 1).strftime("%b") for c in pivot.columns]
    return pivot


def rolling_sharpe_series(daily_returns: pd.Series, window: int = 60) -> pd.Series:
    if len(daily_returns) < window + 1:
        return pd.Series(dtype=float)
    mu = daily_returns.rolling(window, min_periods=window).mean()
    sig = daily_returns.rolling(window, min_periods=window).std()
    rs = np.sqrt(252) * mu / sig.replace(0, np.nan)
    return rs.dropna()


def build_full_analytics(
    backtest: dict[str, Any],
    *,
    initial_capital: float,
    benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Aggregate all metrics from a `backtest_long_cash` result dict (+ optional benchmark dict).
    """
    eq = backtest["equity_curve"].copy()
    eq["Date"] = pd.to_datetime(eq["Date"])
    eq = eq.sort_values("Date").reset_index(drop=True)
    d0, d1 = eq["Date"].iloc[0], eq["Date"].iloc[-1]
    years = _years_between(pd.Timestamp(d0), pd.Timestamp(d1))
    fv = float(backtest["final_portfolio_value"])
    cagr_dec = compute_cagr(initial_capital, fv, years)
    dd_ser = compute_drawdown_series(eq)
    dd_dec = dd_ser["DrawdownPct"] / 100.0
    max_dd_dec = float(dd_dec.min()) if len(dd_dec) else float("nan")
    daily_ret = equity_daily_returns(eq)
    vol_ann = annualized_volatility(daily_ret)
    sharpe = sharpe_ratio(daily_ret)
    sortino = sortino_ratio(daily_ret)
    calmar = float(cagr_dec / abs(max_dd_dec)) if np.isfinite(max_dd_dec) and max_dd_dec < 0 else float("nan")

    trades_df = backtest.get("trades_df")
    if trades_df is None:
        trades_df = pd.DataFrame()
    tstats = trade_statistics(trades_df if isinstance(trades_df, pd.DataFrame) else pd.DataFrame())

    bench_row: dict[str, float] = {}
    if benchmark is not None and benchmark.get("equity_curve") is not None:
        beq = benchmark["equity_curve"]
        beq = beq.copy()
        beq["Date"] = pd.to_datetime(beq["Date"])
        b0, b1 = beq["Date"].iloc[0], beq["Date"].iloc[-1]
        yb = _years_between(pd.Timestamp(b0), pd.Timestamp(b1))
        bfv = float(benchmark["final_portfolio_value"])
        bcagr = compute_cagr(initial_capital, bfv, yb)
        bdret = equity_daily_returns(beq)
        bench_row = {
            "bh_total_return_pct": float(benchmark.get("total_return_pct", float("nan"))),
            "bh_cagr": float(bcagr * 100.0) if np.isfinite(bcagr) else float("nan"),
            "bh_sharpe": sharpe_ratio(bdret),
            "bh_max_dd_pct": float(benchmark.get("max_drawdown_pct", float("nan"))),
            "bh_final_value": bfv,
        }

    monthly = monthly_returns_matrix(eq)
    roll_sh = rolling_sharpe_series(daily_ret, window=min(60, max(20, len(daily_ret) // 3)))

    return {
        "performance": {
            "total_return_pct": float(backtest.get("total_return_pct", float("nan"))),
            "cagr_pct": float(cagr_dec * 100.0) if np.isfinite(cagr_dec) else float("nan"),
            "final_portfolio_value": fv,
            "years": years,
        },
        "risk": {
            "max_drawdown_pct": float(backtest.get("max_drawdown_pct", float("nan"))),
            "avg_drawdown_pct": average_drawdown_pct(dd_dec),
            "volatility_ann_pct": float(vol_ann * 100.0) if np.isfinite(vol_ann) else float("nan"),
        },
        "risk_adjusted": {
            "sharpe": sharpe,
            "sortino": sortino if np.isfinite(sortino) else float("nan"),
            "calmar": calmar if np.isfinite(calmar) else float("nan"),
        },
        "trades": tstats,
        "strategy": {
            "exposure_pct": float(backtest.get("exposure_pct", float("nan"))),
            "expectancy": tstats.get("expectancy", float("nan")),
        },
        "benchmark": bench_row,
        "series": {
            "drawdown": dd_ser,
            "daily_returns": daily_ret,
            "rolling_sharpe": roll_sh,
            "monthly_returns_pivot": monthly,
        },
        "trades_df": trades_df,
    }


def plot_drawdown_curve(drawdown_df: pd.DataFrame, *, title: str = "Drawdown (%)") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    x = pd.to_datetime(drawdown_df["Date"])
    y = drawdown_df["DrawdownPct"].astype(float)
    ax.fill_between(x, y, 0.0, alpha=0.35, color="#c0392b")
    ax.plot(x, y, color="#922b21", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_monthly_returns_heatmap(monthly_pivot: pd.DataFrame, *, title: str = "Monthly returns (%)") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, max(3, len(monthly_pivot) * 0.35)))
    if monthly_pivot.empty:
        ax.text(0.5, 0.5, "Insufficient data for monthly heatmap", ha="center")
        return fig
    arr = monthly_pivot.values.astype(float)
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=-15, vmax=15)
    ax.set_yticks(range(len(monthly_pivot.index)))
    ax.set_yticklabels([str(y) for y in monthly_pivot.index])
    ax.set_xticks(range(len(monthly_pivot.columns)))
    ax.set_xticklabels(list(monthly_pivot.columns), rotation=45, ha="right")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(rolling: pd.Series, *, title: str = "Rolling Sharpe (60d)") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    if rolling.empty:
        ax.text(0.5, 0.5, "Not enough data for rolling Sharpe", ha="center")
        return fig
    ax.plot(rolling.index, rolling.values, color="#2980b9", linewidth=1.5)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100000.0


def backtest_long_cash(signals_df: pd.DataFrame, cfg: BacktestConfig | None = None) -> dict:
    """
    Simple backtest:
    - Start with cash only.
    - If signal==BUY -> invest all cash into shares at `Close` of the trade day.
    - If signal==SELL -> sell all shares at `Close` of the trade day (go back to cash).
    - HOLD -> keep current position.

    Equity is marked-to-market daily using `Next_Close` (end of the predicted next day).
    """
    if cfg is None:
        cfg = BacktestConfig()

    required = ["Date", "Close", "Next_Date", "Next_Close", "Signal"]
    missing = [c for c in required if c not in signals_df.columns]
    if missing:
        raise ValueError(f"signals_df missing columns: {missing}")

    data = signals_df.sort_values("Date").reset_index(drop=True).copy()

    cash = float(cfg.initial_capital)
    shares = 0.0
    entry_date: pd.Timestamp | None = None
    entry_cost = 0.0
    entry_px: float | None = None
    trades_log: list[dict] = []
    in_position_bars = 0
    total_bars = len(data)

    equity_rows = []
    for _, row in data.iterrows():
        trade_price = float(row["Close"])
        next_price = float(row["Next_Close"])
        signal = str(row["Signal"]).upper().strip()
        next_date = row["Next_Date"]
        position_size = float(row["PositionSize"]) if "PositionSize" in row.index and pd.notna(row["PositionSize"]) else 1.0
        position_size = float(np.clip(position_size, 0.0, 1.0))

        # Execute at trade-date close; equity shown at next day's close.
        if signal == "BUY" and shares == 0.0:
            invest_cash = cash * position_size
            if trade_price > 0 and invest_cash > 0:
                shares = invest_cash / trade_price
                cash = cash - invest_cash
                entry_date = pd.Timestamp(row["Date"])
                entry_cost = float(invest_cash)
                entry_px = float(trade_price)
        elif signal == "SELL" and shares != 0.0:
            proceeds = float(shares * trade_price)
            if entry_cost > 0:
                pnl = proceeds - entry_cost
                pnl_pct = (proceeds / entry_cost - 1.0) * 100.0
                trades_log.append(
                    {
                        "EntryDate": entry_date,
                        "ExitDate": pd.Timestamp(row["Date"]),
                        "EntryPx": float(entry_px) if entry_px is not None else float("nan"),
                        "ExitPx": float(trade_price),
                        "EntryCost": float(entry_cost),
                        "Proceeds": float(proceeds),
                        "PnL": float(pnl),
                        "PnLPct": float(pnl_pct),
                        "Win": bool(pnl > 0),
                    }
                )
            cash = cash + proceeds
            shares = 0.0
            entry_date = None
            entry_cost = 0.0
            entry_px = None

        portfolio_value_next = cash + shares * next_price
        equity_rows.append({"Date": next_date, "PortfolioValue": portfolio_value_next})

        if shares > 0:
            in_position_bars += 1

    equity_curve = pd.DataFrame(equity_rows)
    final_value = float(equity_curve["PortfolioValue"].iloc[-1])
    total_return_pct = (final_value / cfg.initial_capital - 1.0) * 100.0
    eq = equity_curve["PortfolioValue"].astype(float)
    rolling_max = eq.cummax()
    drawdown = eq / rolling_max - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0)
    daily_ret = eq.pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily_ret.mean() / daily_ret.std()) if len(daily_ret) > 1 and daily_ret.std() > 0 else float("nan")

    exposure_pct = float(in_position_bars / total_bars * 100.0) if total_bars else 0.0
    trades_df = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()

    return {
        "equity_curve": equity_curve,
        "final_portfolio_value": final_value,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "trades_df": trades_df,
        "exposure_pct": exposure_pct,
        "total_signal_bars": int(total_bars),
    }


def backtest_buy_and_hold(test_rows: pd.DataFrame, cfg: BacktestConfig | None = None) -> dict:
    """
    Buy & hold benchmark on the same test-period rows as the strategy: invest full capital
    at the first bar's Close; mark to market using Next_Close / Next_Date each step.
    Deterministic; no model signal.
    """
    if cfg is None:
        cfg = BacktestConfig()
    required = ["Date", "Close", "Next_Date", "Next_Close"]
    missing = [c for c in required if c not in test_rows.columns]
    if missing:
        raise ValueError(f"test_rows missing columns: {missing}")
    data = test_rows.sort_values("Date").reset_index(drop=True)
    if data.empty:
        raise ValueError("test_rows is empty")

    first_close = float(data.iloc[0]["Close"])
    if first_close <= 0:
        raise ValueError("Invalid first Close for buy-and-hold")
    shares = float(cfg.initial_capital) / first_close
    equity_rows = []
    for _, row in data.iterrows():
        next_price = float(row["Next_Close"])
        next_date = row["Next_Date"]
        equity_rows.append({"Date": next_date, "PortfolioValue": shares * next_price})

    equity_curve = pd.DataFrame(equity_rows)
    final_value = float(equity_curve["PortfolioValue"].iloc[-1])
    total_return_pct = (final_value / cfg.initial_capital - 1.0) * 100.0
    eq = equity_curve["PortfolioValue"].astype(float)
    rolling_max = eq.cummax()
    drawdown = eq / rolling_max - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0)
    daily_ret = eq.pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily_ret.mean() / daily_ret.std()) if len(daily_ret) > 1 and daily_ret.std() > 0 else float("nan")

    return {
        "equity_curve": equity_curve,
        "final_portfolio_value": final_value,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "trades_df": pd.DataFrame(),
        "exposure_pct": 100.0,
        "total_signal_bars": int(len(data)),
    }


def plot_equity_curve(equity_curve: pd.DataFrame, *, title: str = "Equity Curve") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = pd.to_datetime(equity_curve["Date"])
    y = equity_curve["PortfolioValue"].astype(float)
    ax.plot(x, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_equity_comparison(
    equity_model: pd.DataFrame,
    equity_benchmark: pd.DataFrame,
    *,
    title: str = "Strategy vs Buy & Hold",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    x1 = pd.to_datetime(equity_model["Date"])
    y1 = equity_model["PortfolioValue"].astype(float)
    x2 = pd.to_datetime(equity_benchmark["Date"])
    y2 = equity_benchmark["PortfolioValue"].astype(float)
    ax.plot(x1, y1, linewidth=2, label="Model strategy")
    ax.plot(x2, y2, linewidth=2, linestyle="--", label="Buy & hold")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _asof_close_on_dates(equity_dates: pd.Series, price_df: pd.DataFrame) -> pd.Series:
    """Align Close to sorted equity Dates (backward as-of, no look-ahead)."""
    base = pd.DataFrame({"Date": pd.to_datetime(equity_dates).sort_values()}).drop_duplicates(subset=["Date"])
    pr = price_df[["Date", "Close"]].copy()
    pr["Date"] = pd.to_datetime(pr["Date"])
    pr = pr.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    merged = pd.merge_asof(base, pr, on="Date", direction="backward")
    return merged["Close"].astype(float)


def plot_core_combined_performance(
    *,
    strategy_equity: pd.DataFrame,
    test_df: pd.DataFrame,
    buy_hold_equity: Optional[pd.DataFrame] = None,
    underlying_label: str = "Underlying close (normalized)",
    title: str = "Test period — strategy vs benchmarks vs price (indexed to 100 at start)",
) -> plt.Figure:
    """
    Single-name CORE chart: one axis, all series normalized to 100 at the first strategy-equity date
    so portfolio value, buy & hold, and spot price are visually comparable.
    """
    eq = strategy_equity.copy()
    eq["Date"] = pd.to_datetime(eq["Date"])
    eq = eq.sort_values("Date").reset_index(drop=True)
    if eq.empty or len(eq) < 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Not enough equity points to plot", ha="center")
        return fig

    close_al = _asof_close_on_dates(eq["Date"], test_df)
    s0 = float(eq["PortfolioValue"].iloc[0])
    if s0 == 0:
        s0 = 1.0
    y_s = eq["PortfolioValue"].astype(float) / s0 * 100.0

    fv = close_al.first_valid_index()
    if fv is not None:
        c0 = float(close_al.loc[fv])
        if np.isfinite(c0) and c0 != 0:
            y_p = close_al.astype(float) / c0 * 100.0
        else:
            y_p = pd.Series([np.nan] * len(eq), index=eq.index)
    else:
        y_p = pd.Series([np.nan] * len(eq), index=eq.index)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = eq["Date"]
    ax.plot(x, y_s, linewidth=2.2, label="Strategy equity (normalized)", color="#1f77b4")
    ax.plot(x, y_p, linewidth=2.0, linestyle="--", label=underlying_label, color="#ff7f0e", alpha=0.9)

    if buy_hold_equity is not None and len(buy_hold_equity) > 0:
        bh = buy_hold_equity[["Date", "PortfolioValue"]].copy()
        bh["Date"] = pd.to_datetime(bh["Date"])
        bh = bh.sort_values("Date")
        merged_bh = pd.merge_asof(
            eq[["Date"]].sort_values("Date"),
            bh.rename(columns={"PortfolioValue": "BHVal"}),
            on="Date",
            direction="backward",
        )
        bh0 = float(merged_bh["BHVal"].iloc[0])
        if np.isfinite(bh0) and bh0 != 0:
            y_bh = merged_bh["BHVal"].astype(float) / bh0 * 100.0
            ax.plot(eq["Date"], y_bh, linewidth=2.0, linestyle=":", label="Buy & hold (normalized)", color="#2ca02c")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Index (start = 100)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_core_portfolio_combined_performance(
    *,
    strategy_equity: pd.DataFrame,
    signals_by_ticker: Dict[str, pd.DataFrame],
    buy_hold_equity: Optional[pd.DataFrame] = None,
    title: str = "Portfolio test period — strategy vs buy&hold vs equal-weight basket (indexed to 100)",
) -> plt.Figure:
    """
    Basket CORE chart: equal-weight average of each ticker's normalized close (aligned to portfolio equity dates)
    plus normalized strategy and buy & hold curves on the same axes.
    """
    eq = strategy_equity.copy()
    eq["Date"] = pd.to_datetime(eq["Date"])
    eq = eq.sort_values("Date").reset_index(drop=True)
    if eq.empty or len(eq) < 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Not enough equity points to plot", ha="center")
        return fig

    norm_cols: list[np.ndarray] = []
    for _sym, sdf in signals_by_ticker.items():
        if sdf is None or len(sdf) == 0 or "Close" not in sdf.columns:
            continue
        cser = _asof_close_on_dates(eq["Date"], sdf)
        fi = cser.first_valid_index()
        if fi is None:
            continue
        c0 = float(cser.loc[fi])
        if not np.isfinite(c0) or c0 == 0:
            continue
        norm_cols.append((cser.astype(float) / c0 * 100.0).values)

    if norm_cols:
        mat = np.column_stack(norm_cols)
        y_basket = np.nanmean(mat, axis=1)
    else:
        y_basket = np.full(len(eq), np.nan)

    s0 = float(eq["PortfolioValue"].iloc[0])
    if s0 == 0:
        s0 = 1.0
    y_s = eq["PortfolioValue"].astype(float) / s0 * 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    x = eq["Date"]
    ax.plot(x, y_s, linewidth=2.2, label="Strategy portfolio (normalized)", color="#1f77b4")
    if np.isfinite(np.nanmean(y_basket)):
        ax.plot(
            x,
            y_basket,
            linewidth=2.0,
            linestyle="--",
            label="Equal-weight basket — spot closes (normalized)",
            color="#ff7f0e",
            alpha=0.9,
        )

    if buy_hold_equity is not None and len(buy_hold_equity) > 0:
        bh = buy_hold_equity[["Date", "PortfolioValue"]].copy()
        bh["Date"] = pd.to_datetime(bh["Date"])
        bh = bh.sort_values("Date")
        merged_bh = pd.merge_asof(
            eq[["Date"]].sort_values("Date"),
            bh.rename(columns={"PortfolioValue": "BHVal"}),
            on="Date",
            direction="backward",
        )
        bh0 = float(merged_bh["BHVal"].iloc[0])
        if np.isfinite(bh0) and bh0 != 0:
            y_bh = merged_bh["BHVal"].astype(float) / bh0 * 100.0
            ax.plot(eq["Date"], y_bh, linewidth=2.0, linestyle=":", label="Buy & hold portfolio (normalized)", color="#2ca02c")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Index (start = 100)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    """
    Portfolio backtest configuration for a basket of independent single-stock strategies.

    For this demo we:
    - allocate capital equally per ticker
    - backtest each ticker with the same long/cash logic
    - align equity curves by date and sum them
    """

    initial_capital: float = 100000.0
    allocation: str = "equal"  # only "equal" supported in this simple demo


def _common_trading_dates_normalized(signals_by_ticker: Dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    sets: list[set] = []
    for df in signals_by_ticker.values():
        if df is None or df.empty:
            continue
        sets.append(set(pd.to_datetime(df["Date"]).dt.normalize()))
    if not sets:
        return []
    common = sets[0]
    for s in sets[1:]:
        common &= s
    return sorted(common)


def backtest_portfolio_topn_long_cash(
    signals_by_ticker: Dict[str, pd.DataFrame],
    *,
    top_n: int = 3,
    cfg: PortfolioBacktestConfig | None = None,
) -> dict:
    """
    Single cash pool, up to ``top_n`` concurrent longs (TECHNICAL_MODE portfolio).

    - Uses dates in the **intersection** of all ticker calendars (daily alignment).
    - When there is at least one BUY in the universe, keeps only the top ``top_n`` by ``Proba``
      (already pre-filtered by ``apply_daily_top_n_buys`` in the pipeline); sells names that drop
      out of that set or hit SELL. Deploys cash equally across new entries.
    - When there are no BUY signals, holds existing positions unless a per-name SELL fires.

    Deterministic; no randomness.
    """
    if cfg is None:
        cfg = PortfolioBacktestConfig()
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    tickers = [t for t in signals_by_ticker if signals_by_ticker[t] is not None and len(signals_by_ticker[t])]
    if not tickers:
        raise ValueError("signals_by_ticker is empty")

    lookup: dict[tuple[str, pd.Timestamp], pd.Series] = {}
    for t in tickers:
        df = signals_by_ticker[t]
        for _, row in df.iterrows():
            dn = pd.Timestamp(row["Date"]).normalize()
            lookup[(t, dn)] = row

    common = [d for d in _common_trading_dates_normalized(signals_by_ticker) if all((t, d) in lookup for t in tickers)]
    if not common:
        raise ValueError("No overlapping trading dates with complete rows for all tickers.")

    cash = float(cfg.initial_capital)
    shares: dict[str, float] = {t: 0.0 for t in tickers}
    equity_rows: list[dict] = []

    for d in common:
        row_map = {t: lookup[(t, d)] for t in tickers}
        close = {t: float(row_map[t]["Close"]) for t in tickers}
        next_close = {t: float(row_map[t]["Next_Close"]) for t in tickers}
        next_date = row_map[tickers[0]]["Next_Date"]
        sig = {t: str(row_map[t]["Signal"]).upper().strip() for t in tickers}
        proba = {t: float(row_map[t]["Proba"]) if "Proba" in row_map[t].index and pd.notna(row_map[t]["Proba"]) else 0.5 for t in tickers}

        buy_candidates = [t for t in tickers if sig[t] == "BUY"]
        buy_candidates.sort(key=lambda t: -proba[t])

        if buy_candidates:
            top_pool = buy_candidates[:top_n]
            top_set = set(top_pool)
            for t in tickers:
                if shares[t] <= 0:
                    continue
                if sig[t] == "SELL" or t not in top_set:
                    cash += shares[t] * close[t]
                    shares[t] = 0.0

            need_buy = [t for t in top_pool if shares[t] <= 0 and sig[t] == "BUY"]
            if need_buy and cash > 0:
                per_cash = cash / float(len(need_buy))
                for t in need_buy:
                    pm = float(row_map[t]["PositionSize"]) if "PositionSize" in row_map[t].index and pd.notna(row_map[t]["PositionSize"]) else 1.0
                    pm = float(np.clip(pm, 0.0, 1.0))
                    invest = per_cash * pm
                    if close[t] > 0 and invest > 0 and cash >= invest - 1e-9:
                        shares[t] += invest / close[t]
                        cash -= invest
        else:
            for t in tickers:
                if shares[t] > 0 and sig[t] == "SELL":
                    cash += shares[t] * close[t]
                    shares[t] = 0.0

        pv = float(cash + sum(shares[t] * next_close[t] for t in tickers))
        equity_rows.append({"Date": next_date, "PortfolioValue": pv})

    equity_curve = pd.DataFrame(equity_rows)
    final_value = float(equity_curve["PortfolioValue"].iloc[-1])
    total_return_pct = (final_value / float(cfg.initial_capital) - 1.0) * 100.0
    eq = equity_curve["PortfolioValue"].astype(float)
    rolling_max = eq.cummax()
    drawdown = eq / rolling_max - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0)
    daily_ret = eq.pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily_ret.mean() / daily_ret.std()) if len(daily_ret) > 1 and daily_ret.std() > 0 else float("nan")

    return {
        "equity_curve": equity_curve,
        "final_portfolio_value": final_value,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "per_ticker_final": {},
        "trades_df": pd.DataFrame(),
        "exposure_pct": float("nan"),
        "total_signal_bars": int(len(common)),
    }


def backtest_portfolio_long_cash(
    signals_by_ticker: Dict[str, pd.DataFrame],
    cfg: PortfolioBacktestConfig | None = None,
) -> dict:
    """
    Backtest a basket of tickers by running `backtest_long_cash` per ticker and summing.

    Returns:
      - equity_curve: DataFrame with Date + total PortfolioValue
      - final_portfolio_value
      - total_return_pct
      - per_ticker_final: dict[ticker] = final_value
    """
    if cfg is None:
        cfg = PortfolioBacktestConfig()

    tickers = [t for t in signals_by_ticker.keys() if signals_by_ticker[t] is not None]
    if not tickers:
        raise ValueError("signals_by_ticker is empty")

    if cfg.allocation.lower() != "equal":
        raise ValueError("Only allocation='equal' is supported")

    n = len(tickers)
    initial_cap_per = float(cfg.initial_capital) / n

    equity_by_ticker: dict[str, pd.Series] = {}
    per_ticker_final: dict[str, float] = {}
    start_dates = []

    for ticker in tickers:
        bt = backtest_long_cash(signals_by_ticker[ticker], cfg=BacktestConfig(initial_capital=initial_cap_per))
        eq = bt["equity_curve"].copy()
        series = eq.set_index("Date")["PortfolioValue"].astype(float).sort_index()
        equity_by_ticker[ticker] = series
        per_ticker_final[ticker] = float(series.iloc[-1])
        start_dates.append(series.index.min())

    global_dates = pd.date_range(start=min(start_dates), end=max(s.index.max() for s in equity_by_ticker.values()), freq="D")

    # Sum all tickers' equity curves after aligning on a shared calendar.
    # Any missing days are treated as "forward-filled positions" and prior to a
    # ticker's first point we assume it remains in cash (initial_cap_per).
    total_values = np.zeros(len(global_dates), dtype=float)
    for ticker, series in equity_by_ticker.items():
        aligned = series.reindex(global_dates).ffill()
        aligned = aligned.fillna(initial_cap_per)
        total_values += aligned.values

    equity_curve = pd.DataFrame({"Date": global_dates, "PortfolioValue": total_values})
    final_value = float(equity_curve["PortfolioValue"].iloc[-1])
    total_return_pct = (final_value / float(cfg.initial_capital) - 1.0) * 100.0
    eq = equity_curve["PortfolioValue"].astype(float)
    rolling_max = eq.cummax()
    drawdown = eq / rolling_max - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0)
    daily_ret = eq.pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily_ret.mean() / daily_ret.std()) if len(daily_ret) > 1 and daily_ret.std() > 0 else float("nan")

    return {
        "equity_curve": equity_curve,
        "final_portfolio_value": final_value,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "per_ticker_final": per_ticker_final,
        "trades_df": pd.DataFrame(),
        "exposure_pct": float("nan"),
        "total_signal_bars": int(len(global_dates)),
    }


def backtest_portfolio_buy_and_hold(
    signals_by_ticker: Dict[str, pd.DataFrame],
    cfg: PortfolioBacktestConfig | None = None,
) -> dict:
    """
    Equal-weight buy & hold on the same per-ticker test windows as the strategy backtest.
    Each leg uses `backtest_buy_and_hold` with initial_capital / N; equity curves are date-aligned and summed.
    """
    if cfg is None:
        cfg = PortfolioBacktestConfig()

    tickers = [t for t in signals_by_ticker.keys() if signals_by_ticker[t] is not None]
    if not tickers:
        raise ValueError("signals_by_ticker is empty")

    if cfg.allocation.lower() != "equal":
        raise ValueError("Only allocation='equal' is supported")

    n = len(tickers)
    initial_cap_per = float(cfg.initial_capital) / n

    equity_by_ticker: dict[str, pd.Series] = {}
    per_ticker_final: dict[str, float] = {}
    start_dates = []

    for ticker in tickers:
        bt = backtest_buy_and_hold(signals_by_ticker[ticker], cfg=BacktestConfig(initial_capital=initial_cap_per))
        eq = bt["equity_curve"].copy()
        series = eq.set_index("Date")["PortfolioValue"].astype(float).sort_index()
        equity_by_ticker[ticker] = series
        per_ticker_final[ticker] = float(series.iloc[-1])
        start_dates.append(series.index.min())

    global_dates = pd.date_range(start=min(start_dates), end=max(s.index.max() for s in equity_by_ticker.values()), freq="D")

    total_values = np.zeros(len(global_dates), dtype=float)
    for ticker, series in equity_by_ticker.items():
        aligned = series.reindex(global_dates).ffill()
        aligned = aligned.fillna(initial_cap_per)
        total_values += aligned.values

    equity_curve = pd.DataFrame({"Date": global_dates, "PortfolioValue": total_values})
    final_value = float(equity_curve["PortfolioValue"].iloc[-1])
    total_return_pct = (final_value / float(cfg.initial_capital) - 1.0) * 100.0
    eq = equity_curve["PortfolioValue"].astype(float)
    rolling_max = eq.cummax()
    drawdown = eq / rolling_max - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0)
    daily_ret = eq.pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily_ret.mean() / daily_ret.std()) if len(daily_ret) > 1 and daily_ret.std() > 0 else float("nan")

    return {
        "equity_curve": equity_curve,
        "final_portfolio_value": final_value,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "per_ticker_final": per_ticker_final,
        "trades_df": pd.DataFrame(),
        "exposure_pct": float("nan"),
        "total_signal_bars": int(len(global_dates)),
    }


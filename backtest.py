from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_analytics import infer_periods_per_year, sharpe_ratio, sortino_ratio, trade_statistics
from transaction_costs import (
    IndiaEquityCostConfig,
    apply_slippage,
    buy_cost_fraction,
    sell_cost_fraction,
)


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100000.0
    # None selects the documented NSE cash-delivery assumption set.
    cost_config: IndiaEquityCostConfig | None = None
    apply_costs: bool = True


def _cost_cfg(cfg: BacktestConfig) -> IndiaEquityCostConfig:
    if cfg.cost_config is not None:
        return cfg.cost_config
    return IndiaEquityCostConfig.equity_delivery()


def _finite_price(value) -> float:
    try:
        px = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(px) or px <= 0:
        return float("nan")
    return px


def _new_account(cash: float) -> dict:
    return {
        "cash": float(cash),
        "shares": 0.0,
        "entry_debit": 0.0,
        "entry_date": None,
        "entry_px": None,
    }


def _execute_at_open(
    state: dict,
    open_px: float,
    signal: str,
    position_size: float,
    when: pd.Timestamp,
    cost_cfg: IndiaEquityCostConfig,
    *,
    with_costs: bool,
) -> dict | None:
    """
    Fill a signal from the previous completed bar at this bar's open.

    Returns a closed-trade dict on a sell. Buys return None. The fill price is
    never the signal bar's close.
    """
    sig = str(signal).upper().strip()
    if sig == "BUY" and state["shares"] == 0.0:
        fill = apply_slippage(open_px, "BUY", cost_cfg) if with_costs else float(open_px)
        frac = float(np.clip(position_size, 0.0, 1.0))
        budget = state["cash"] * frac
        rate = buy_cost_fraction(cost_cfg) if with_costs else 0.0
        if budget <= 0 or fill <= 0:
            return None
        notional = budget / (1.0 + rate)
        charges = notional * rate
        if notional + charges > state["cash"] + 1e-8:
            notional = state["cash"] / (1.0 + rate)
            charges = state["cash"] - notional
        state["shares"] = notional / fill
        state["cash"] = state["cash"] - notional - charges
        state["entry_debit"] = notional + charges
        state["entry_date"] = pd.Timestamp(when)
        state["entry_px"] = float(fill)
        state["last_charges"] = float(charges)
        state["last_notional"] = float(notional)
        state["last_side"] = "BUY"
        return None
    if sig == "SELL" and state["shares"] > 0.0:
        fill = apply_slippage(open_px, "SELL", cost_cfg) if with_costs else float(open_px)
        notional = state["shares"] * fill
        rate = sell_cost_fraction(cost_cfg) if with_costs else 0.0
        charges = notional * rate if with_costs else 0.0
        proceeds = notional - charges
        pnl = proceeds - float(state["entry_debit"])
        pnl_pct = (proceeds / state["entry_debit"] - 1.0) * 100.0 if state["entry_debit"] > 0 else float("nan")
        trade = {
            "EntryDate": state["entry_date"],
            "ExitDate": pd.Timestamp(when),
            "EntryPx": float(state["entry_px"]) if state["entry_px"] is not None else float("nan"),
            "ExitPx": float(fill),
            "EntryCost": float(state["entry_debit"]),
            "Proceeds": float(proceeds),
            "PnL": float(pnl),
            "PnLPct": float(pnl_pct),
            "Win": bool(pnl > 0),
            "Charges": float(charges),
        }
        state["cash"] = state["cash"] + proceeds
        state["shares"] = 0.0
        state["entry_debit"] = 0.0
        state["entry_date"] = None
        state["entry_px"] = None
        state["last_charges"] = float(charges)
        state["last_notional"] = float(notional)
        state["last_side"] = "SELL"
        return trade
    state["last_charges"] = 0.0
    state["last_notional"] = 0.0
    state["last_side"] = ""
    return None


def _mark(state: dict, close_px: float) -> float:
    return float(state["cash"] + state["shares"] * close_px)


def _summarize_pair(
    equity_net: pd.DataFrame,
    equity_gross: pd.DataFrame,
    *,
    initial_capital: float,
    trades: list[dict],
    total_costs: float,
    turnover_notional: float,
    fills: int,
    exposure_bars: int,
    signal_bars: int,
) -> dict:
    if equity_net.empty:
        raise ValueError("No equity points. Need at least one bar with a valid Close.")
    final_value = float(equity_net["PortfolioValue"].iloc[-1])
    gross_final = float(equity_gross["PortfolioValue"].iloc[-1]) if len(equity_gross) else final_value
    net_return_pct = (final_value / float(initial_capital) - 1.0) * 100.0
    gross_return_pct = (gross_final / float(initial_capital) - 1.0) * 100.0
    eq = equity_net["PortfolioValue"].astype(float)
    rolling_max = eq.cummax()
    drawdown = eq / rolling_max - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0)
    dated = equity_net.copy()
    dated["Date"] = pd.to_datetime(dated["Date"])
    rets = eq.pct_change()
    rets.index = pd.DatetimeIndex(dated["Date"])
    ppy = infer_periods_per_year(dated["Date"])
    sharpe = sharpe_ratio(rets.dropna(), periods_per_year=ppy)
    sortino = sortino_ratio(rets.dropna(), periods_per_year=ppy)
    d0 = pd.Timestamp(dated["Date"].iloc[0])
    d1 = pd.Timestamp(dated["Date"].iloc[-1])
    years = float((d1 - d0).days) / 365.25
    if years > 0 and final_value > 0 and initial_capital > 0:
        cagr = float((final_value / float(initial_capital)) ** (1.0 / years) - 1.0)
    else:
        cagr = float("nan")
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    stats = trade_statistics(trades_df)
    avg_equity = float(eq.mean()) if len(eq) else float("nan")
    turnover = float(turnover_notional / avg_equity) if avg_equity and np.isfinite(avg_equity) and avg_equity > 0 else float("nan")
    exposure_pct = float(exposure_bars / signal_bars * 100.0) if signal_bars else 0.0
    return {
        "equity_curve": equity_net,
        "gross_equity_curve": equity_gross,
        "final_portfolio_value": final_value,
        "gross_final_portfolio_value": gross_final,
        "total_return_pct": net_return_pct,
        "net_return_pct": net_return_pct,
        "gross_return_pct": gross_return_pct,
        "total_transaction_costs": float(total_costs),
        "turnover": turnover,
        "turnover_notional": float(turnover_notional),
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "sortino": sortino if np.isfinite(sortino) else float("nan"),
        "cagr": cagr,
        "cagr_pct": float(cagr * 100.0) if np.isfinite(cagr) else float("nan"),
        "periods_per_year": ppy,
        "trades_df": trades_df,
        "trade_count": int(stats["total_trades"]),
        "fill_count": int(fills),
        "win_rate": float(stats["win_rate"]),
        "profit_factor": float(stats["profit_factor"]) if stats["profit_factor"] is not None else float("nan"),
        "exposure_pct": exposure_pct,
        "total_signal_bars": int(signal_bars),
    }


def backtest_long_cash(signals_df: pd.DataFrame, cfg: BacktestConfig | None = None) -> dict:
    """
    Long/cash backtest.

    A signal on a completed bar fills at the next bar's Open, never at the
    close that produced the signal. Equity is marked at each subsequent Close.
    The last bar can be scored but cannot fill, because no later open exists.
    A missing Open skips that fill instead of substituting the close.
    """
    if cfg is None:
        cfg = BacktestConfig()
    required = ["Date", "Open", "Close", "Signal"]
    missing = [c for c in required if c not in signals_df.columns]
    if missing:
        raise ValueError(f"signals_df missing columns: {missing}")
    data = signals_df.sort_values("Date").reset_index(drop=True).copy()
    if data.empty:
        raise ValueError("signals_df is empty")

    cost_cfg = _cost_cfg(cfg)
    net = _new_account(cfg.initial_capital)
    gross = _new_account(cfg.initial_capital)
    trades: list[dict] = []
    equity_net: list[dict] = []
    equity_gross: list[dict] = []
    total_costs = 0.0
    turnover_notional = 0.0
    fills = 0
    exposure_bars = 0
    prev_signal: str | None = None
    prev_size = 1.0

    for i in range(len(data)):
        row = data.iloc[i]
        open_px = _finite_price(row["Open"])
        close_px = _finite_price(row["Close"])
        when = pd.Timestamp(row["Date"])
        if prev_signal in {"BUY", "SELL"} and np.isfinite(open_px):
            gross_trade = _execute_at_open(
                gross, open_px, prev_signal, prev_size, when, cost_cfg, with_costs=False
            )
            net_trade = _execute_at_open(
                net, open_px, prev_signal, prev_size, when, cost_cfg, with_costs=bool(cfg.apply_costs)
            )
            charges = float(net.get("last_charges", 0.0))
            notional = float(net.get("last_notional", 0.0))
            if net.get("last_side") in {"BUY", "SELL"}:
                fills += 1
                total_costs += charges
                turnover_notional += notional
            if net_trade is not None:
                if gross_trade is not None:
                    net_trade["GrossPnL"] = float(gross_trade["PnL"])
                trades.append(net_trade)
        if np.isfinite(close_px):
            equity_net.append({"Date": when, "PortfolioValue": _mark(net, close_px)})
            equity_gross.append({"Date": when, "PortfolioValue": _mark(gross, close_px)})
            if net["shares"] > 0:
                exposure_bars += 1
        if i < len(data) - 1:
            prev_signal = str(row["Signal"]).upper().strip()
            if "PositionSize" in data.columns and pd.notna(row["PositionSize"]):
                prev_size = float(row["PositionSize"])
            else:
                prev_size = 1.0
        else:
            prev_signal = None

    return _summarize_pair(
        pd.DataFrame(equity_net),
        pd.DataFrame(equity_gross),
        initial_capital=float(cfg.initial_capital),
        trades=trades,
        total_costs=total_costs,
        turnover_notional=turnover_notional,
        fills=fills,
        exposure_bars=exposure_bars,
        signal_bars=len(equity_net),
    )


def backtest_buy_and_hold(test_rows: pd.DataFrame, cfg: BacktestConfig | None = None) -> dict:
    """
    Buy at the first valid Open and hold. Marks use each bar's Close.
    Entry costs apply; the position is not force-liquidated at the last close,
    matching a strategy that is still long at the sample end.
    """
    if cfg is None:
        cfg = BacktestConfig()
    required = ["Date", "Open", "Close"]
    missing = [c for c in required if c not in test_rows.columns]
    if missing:
        raise ValueError(f"test_rows missing columns: {missing}")
    data = test_rows.sort_values("Date").reset_index(drop=True).copy()
    if data.empty:
        raise ValueError("test_rows is empty")
    data = data.copy()
    data["Signal"] = "HOLD"
    # Reuse the long/cash engine by placing a single BUY on a synthetic prior decision:
    # insert the buy as the signal of a leading placeholder only when the first open is valid.
    # Direct fill at the first finite open keeps the benchmark passive from the first session.
    cost_cfg = _cost_cfg(cfg)
    net = _new_account(cfg.initial_capital)
    gross = _new_account(cfg.initial_capital)
    equity_net: list[dict] = []
    equity_gross: list[dict] = []
    total_costs = 0.0
    turnover_notional = 0.0
    fills = 0
    exposure_bars = 0
    entered = False
    for i in range(len(data)):
        row = data.iloc[i]
        open_px = _finite_price(row["Open"])
        close_px = _finite_price(row["Close"])
        when = pd.Timestamp(row["Date"])
        if not entered and np.isfinite(open_px):
            _execute_at_open(gross, open_px, "BUY", 1.0, when, cost_cfg, with_costs=False)
            _execute_at_open(net, open_px, "BUY", 1.0, when, cost_cfg, with_costs=bool(cfg.apply_costs))
            if net.get("last_side") == "BUY":
                entered = True
                fills += 1
                total_costs += float(net.get("last_charges", 0.0))
                turnover_notional += float(net.get("last_notional", 0.0))
        if np.isfinite(close_px):
            equity_net.append({"Date": when, "PortfolioValue": _mark(net, close_px)})
            equity_gross.append({"Date": when, "PortfolioValue": _mark(gross, close_px)})
            if net["shares"] > 0:
                exposure_bars += 1
    return _summarize_pair(
        pd.DataFrame(equity_net),
        pd.DataFrame(equity_gross),
        initial_capital=float(cfg.initial_capital),
        trades=[],
        total_costs=total_costs,
        turnover_notional=turnover_notional,
        fills=fills,
        exposure_bars=exposure_bars,
        signal_bars=len(equity_net),
    )


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
    cost_config: IndiaEquityCostConfig | None = None
    apply_costs: bool = True


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

    for t in tickers:
        if "Open" not in signals_by_ticker[t].columns:
            raise ValueError(f"{t} signals are missing Open. Next-session fills need the open.")

    cost_cfg = cfg.cost_config if cfg.cost_config is not None else IndiaEquityCostConfig.equity_delivery()
    net_cash = float(cfg.initial_capital)
    gross_cash = float(cfg.initial_capital)
    net_shares: dict[str, float] = {t: 0.0 for t in tickers}
    gross_shares: dict[str, float] = {t: 0.0 for t in tickers}
    last_close: dict[str, float] = {}
    equity_net: list[dict] = []
    equity_gross: list[dict] = []
    total_costs = 0.0
    turnover_notional = 0.0
    fills = 0
    exposure_bars = 0
    pending: dict | None = None

    def _px(frame_row: pd.Series, col: str) -> float:
        return _finite_price(frame_row[col]) if col in frame_row.index else float("nan")

    def _sell(book_cash: float, book_shares: dict[str, float], t: str, open_px: float, *, with_costs: bool) -> float:
        nonlocal total_costs, turnover_notional, fills
        if book_shares[t] <= 0 or not np.isfinite(open_px):
            return book_cash
        fill = apply_slippage(open_px, "SELL", cost_cfg) if with_costs else float(open_px)
        notional = book_shares[t] * fill
        charges = notional * sell_cost_fraction(cost_cfg) if with_costs else 0.0
        book_shares[t] = 0.0
        if with_costs:
            total_costs += charges
            turnover_notional += notional
            fills += 1
        return book_cash + notional - charges

    def _buy(book_cash: float, book_shares: dict[str, float], t: str, open_px: float, budget: float, *, with_costs: bool) -> float:
        nonlocal total_costs, turnover_notional, fills
        if book_shares[t] > 0 or budget <= 0 or book_cash <= 0 or not np.isfinite(open_px):
            return book_cash
        fill = apply_slippage(open_px, "BUY", cost_cfg) if with_costs else float(open_px)
        rate = buy_cost_fraction(cost_cfg) if with_costs else 0.0
        spend = min(book_cash, budget)
        notional = spend / (1.0 + rate)
        charges = notional * rate
        book_shares[t] = notional / fill
        if with_costs:
            total_costs += charges
            turnover_notional += notional
            fills += 1
        return book_cash - notional - charges

    for i, d in enumerate(common):
        row_map = {t: lookup[(t, d)] for t in tickers}
        if pending is not None:
            for t in pending["sell"]:
                op = _px(row_map[t], "Open")
                net_cash = _sell(net_cash, net_shares, t, op, with_costs=bool(cfg.apply_costs))
                gross_cash = _sell(gross_cash, gross_shares, t, op, with_costs=False)
            buys = pending["buy"]
            if buys and net_cash > 0:
                per_net = net_cash / float(len(buys))
                per_gross = gross_cash / float(len(buys)) if gross_cash > 0 else 0.0
                for t in buys:
                    op = _px(row_map[t], "Open")
                    size = float(pending["size"].get(t, 1.0))
                    net_cash = _buy(net_cash, net_shares, t, op, per_net * size, with_costs=bool(cfg.apply_costs))
                    gross_cash = _buy(gross_cash, gross_shares, t, op, per_gross * size, with_costs=False)

        mtm_net = net_cash
        mtm_gross = gross_cash
        for t in tickers:
            cl = _px(row_map[t], "Close")
            if np.isfinite(cl):
                last_close[t] = cl
            mark_px = last_close.get(t, float("nan"))
            if np.isfinite(mark_px):
                mtm_net += net_shares[t] * mark_px
                mtm_gross += gross_shares[t] * mark_px
        equity_net.append({"Date": d, "PortfolioValue": mtm_net})
        equity_gross.append({"Date": d, "PortfolioValue": mtm_gross})
        if any(net_shares[t] > 0 for t in tickers):
            exposure_bars += 1

        if i < len(common) - 1:
            sig = {t: str(row_map[t]["Signal"]).upper().strip() for t in tickers}
            proba = {
                t: float(row_map[t]["Proba"]) if "Proba" in row_map[t].index and pd.notna(row_map[t]["Proba"]) else 0.5
                for t in tickers
            }
            buy_candidates = [t for t in tickers if sig[t] == "BUY"]
            buy_candidates.sort(key=lambda t: -proba[t])
            sells: list[str] = []
            buys_next: list[str] = []
            sizes: dict[str, float] = {}
            if buy_candidates:
                top_pool = buy_candidates[:top_n]
                top_set = set(top_pool)
                for t in tickers:
                    held = net_shares[t] > 0 or gross_shares[t] > 0
                    if held and (sig[t] == "SELL" or t not in top_set):
                        sells.append(t)
                for t in top_pool:
                    if net_shares[t] <= 0 and sig[t] == "BUY" and t not in sells:
                        buys_next.append(t)
                        raw = row_map[t]["PositionSize"] if "PositionSize" in row_map[t].index else 1.0
                        sizes[t] = float(np.clip(raw if pd.notna(raw) else 1.0, 0.0, 1.0))
            else:
                for t in tickers:
                    held = net_shares[t] > 0 or gross_shares[t] > 0
                    if held and sig[t] == "SELL":
                        sells.append(t)
            pending = {"sell": sells, "buy": buys_next, "size": sizes}
        else:
            pending = None

    summary = _summarize_pair(
        pd.DataFrame(equity_net),
        pd.DataFrame(equity_gross),
        initial_capital=float(cfg.initial_capital),
        trades=[],
        total_costs=total_costs,
        turnover_notional=turnover_notional,
        fills=fills,
        exposure_bars=exposure_bars,
        signal_bars=len(equity_net),
    )
    summary["per_ticker_final"] = {}
    return summary


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

    leg_cfg = BacktestConfig(
        initial_capital=initial_cap_per,
        cost_config=cfg.cost_config,
        apply_costs=cfg.apply_costs,
    )
    gross_by_ticker: dict[str, pd.Series] = {}
    trades_all: list[pd.DataFrame] = []
    total_costs = 0.0
    turnover_notional = 0.0
    fills = 0
    for ticker in tickers:
        bt = backtest_long_cash(signals_by_ticker[ticker], cfg=leg_cfg)
        eq = bt["equity_curve"].copy()
        series = eq.set_index("Date")["PortfolioValue"].astype(float).sort_index()
        equity_by_ticker[ticker] = series
        gseries = bt["gross_equity_curve"].set_index("Date")["PortfolioValue"].astype(float).sort_index()
        gross_by_ticker[ticker] = gseries
        per_ticker_final[ticker] = float(series.iloc[-1])
        start_dates.append(series.index.min())
        total_costs += float(bt.get("total_transaction_costs", 0.0))
        turnover_notional += float(bt.get("turnover_notional", 0.0))
        fills += int(bt.get("fill_count", 0))
        if isinstance(bt.get("trades_df"), pd.DataFrame) and len(bt["trades_df"]):
            td = bt["trades_df"].copy()
            td["Ticker"] = ticker
            trades_all.append(td)

    global_dates = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in equity_by_ticker.values()])))

    # Sum legs on the union of actual session dates. Before a leg's first print, cash.
    total_values = np.zeros(len(global_dates), dtype=float)
    gross_values = np.zeros(len(global_dates), dtype=float)
    for ticker, series in equity_by_ticker.items():
        aligned = series.reindex(global_dates).ffill().fillna(initial_cap_per)
        total_values += aligned.values
        g_aligned = gross_by_ticker[ticker].reindex(global_dates).ffill().fillna(initial_cap_per)
        gross_values += g_aligned.values

    equity_curve = pd.DataFrame({"Date": global_dates, "PortfolioValue": total_values})
    gross_curve = pd.DataFrame({"Date": global_dates, "PortfolioValue": gross_values})
    trades_df = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    summary = _summarize_pair(
        equity_curve,
        gross_curve,
        initial_capital=float(cfg.initial_capital),
        trades=trades_df.to_dict("records") if len(trades_df) else [],
        total_costs=total_costs,
        turnover_notional=turnover_notional,
        fills=fills,
        exposure_bars=len(global_dates),
        signal_bars=len(global_dates),
    )
    summary["per_ticker_final"] = per_ticker_final
    return summary


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

    leg_cfg = BacktestConfig(
        initial_capital=initial_cap_per,
        cost_config=cfg.cost_config,
        apply_costs=cfg.apply_costs,
    )
    gross_by_ticker: dict[str, pd.Series] = {}
    total_costs = 0.0
    for ticker in tickers:
        bt = backtest_buy_and_hold(signals_by_ticker[ticker], cfg=leg_cfg)
        eq = bt["equity_curve"].copy()
        series = eq.set_index("Date")["PortfolioValue"].astype(float).sort_index()
        equity_by_ticker[ticker] = series
        gross_by_ticker[ticker] = bt["gross_equity_curve"].set_index("Date")["PortfolioValue"].astype(float).sort_index()
        per_ticker_final[ticker] = float(series.iloc[-1])
        start_dates.append(series.index.min())
        total_costs += float(bt.get("total_transaction_costs", 0.0))

    global_dates = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in equity_by_ticker.values()])))
    total_values = np.zeros(len(global_dates), dtype=float)
    gross_values = np.zeros(len(global_dates), dtype=float)
    for ticker, series in equity_by_ticker.items():
        total_values += series.reindex(global_dates).ffill().fillna(initial_cap_per).values
        gross_values += gross_by_ticker[ticker].reindex(global_dates).ffill().fillna(initial_cap_per).values

    equity_curve = pd.DataFrame({"Date": global_dates, "PortfolioValue": total_values})
    gross_curve = pd.DataFrame({"Date": global_dates, "PortfolioValue": gross_values})
    summary = _summarize_pair(
        equity_curve,
        gross_curve,
        initial_capital=float(cfg.initial_capital),
        trades=[],
        total_costs=total_costs,
        turnover_notional=0.0,
        fills=len(tickers),
        exposure_bars=len(global_dates),
        signal_bars=len(global_dates),
    )
    summary["per_ticker_final"] = per_ticker_final
    return summary


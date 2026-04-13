from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def attach_technical_trend_signals(
    test_df: pd.DataFrame,
    proba_buy: np.ndarray,
    *,
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
    close_col: str = "Close",
    ma50_col: str = "MA50",
) -> pd.DataFrame:
    """
    Technical-only execution rule (TECHNICAL_MODE): no RSI band, no fusion.

    BUY: P(strong up) > buy_threshold AND price above MA50 (trend confirmation).
    SELL: P < sell_threshold AND price below MA50.
    Otherwise HOLD — positions persist until a SELL or a stronger rotation rule in portfolio mode.
    """
    out = test_df.copy()
    p = np.asarray(proba_buy, dtype=float)
    p = np.clip(np.where(np.isfinite(p), p, 0.5), 0.0, 1.0)
    out["Proba"] = p

    if ma50_col not in out.columns or close_col not in out.columns:
        raise ValueError(f"attach_technical_trend_signals requires {close_col!r} and {ma50_col!r}")

    sig: list[str] = []
    for i in range(len(out)):
        row = out.iloc[i]
        px = float(row[close_col])
        ma = float(row[ma50_col])
        pv = float(p[i])
        if not np.isfinite(px) or not np.isfinite(ma):
            sig.append("HOLD")
        elif pv > buy_threshold and px > ma:
            sig.append("BUY")
        elif pv < sell_threshold and px < ma:
            sig.append("SELL")
        else:
            sig.append("HOLD")
    out["Signal"] = sig
    return out


def apply_daily_top_n_buys(
    signals_by_ticker: dict[str, pd.DataFrame],
    *,
    top_n: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Cross-sectional concentration: on each date, keep BUY only on the top `top_n` tickers
    by Proba among names that already show BUY (post model + risk). Others become HOLD for that day.

    Modifies copies; original dataframes are unchanged.
    """
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    out = {t: df.copy().reset_index(drop=True) for t, df in signals_by_ticker.items()}

    by_date: dict[pd.Timestamp, list[tuple[str, int, float]]] = defaultdict(list)
    for t, df in out.items():
        for i in range(len(df)):
            if str(df.at[i, "Signal"]).upper().strip() != "BUY":
                continue
            d = pd.Timestamp(df.at[i, "Date"]).normalize()
            proba = float(df.at[i, "Proba"]) if "Proba" in df.columns else 0.5
            by_date[d].append((t, i, proba))

    demote: set[tuple[str, int]] = set()
    for d, rows in by_date.items():
        rows.sort(key=lambda x: -x[2])
        allowed: list[str] = []
        for ticker, _i, _p in rows:
            if ticker not in allowed:
                allowed.append(ticker)
            if len(allowed) >= top_n:
                break
        allowed_set = set(allowed)
        for ticker, i, _p in rows:
            if ticker not in allowed_set:
                demote.add((ticker, i))

    for ticker, i in demote:
        out[ticker].at[i, "Signal"] = "HOLD"
        if "PositionSize" in out[ticker].columns:
            out[ticker].at[i, "PositionSize"] = 0.0

    return out


def prob_to_signal(
    proba_buy: np.ndarray,
    *,
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
) -> np.ndarray:
    """
    Convert predicted probability of next-day return>0 into discrete trading signals.
    """
    if not (0 < sell_threshold < buy_threshold < 1):
        raise ValueError("Require 0 < sell_threshold < buy_threshold < 1")

    signals = np.array(["HOLD"] * len(proba_buy), dtype=object)
    signals[proba_buy > buy_threshold] = "BUY"
    signals[proba_buy < sell_threshold] = "SELL"
    return signals


def prob_to_signal_quantile(
    proba_buy: np.ndarray,
    *,
    buy_quantile: float = 0.7,
    sell_quantile: float = 0.3,
) -> np.ndarray:
    """
    Quantile-based signal mapping.

    Instead of using fixed probability cutoffs, we compute cutoffs from the
    predicted probability distribution for the test period:
    - BUY if proba >= q_buy
    - SELL if proba <= q_sell
    - otherwise HOLD

    This is useful when a model's probabilities are "compressed" (common for
    Logistic Regression), causing fixed thresholds to produce all HOLD.
    """
    if not (0 < sell_quantile < buy_quantile < 1):
        raise ValueError("Require 0 < sell_quantile < buy_quantile < 1")

    n = len(proba_buy)
    if n == 0:
        return np.array([], dtype=object)

    # Rank-based quantile mapping is robust when probabilities are flat/tied.
    n_buy = max(1, int(round((1.0 - buy_quantile) * n)))
    n_sell = max(1, int(round(sell_quantile * n)))
    if n_buy + n_sell > n:
        overflow = n_buy + n_sell - n
        # Trim the larger side first to preserve both directions.
        if n_buy >= n_sell:
            n_buy = max(1, n_buy - overflow)
        else:
            n_sell = max(1, n_sell - overflow)
    if n >= 2 and n_buy + n_sell < 2:
        n_buy = 1
        n_sell = 1 if n >= 2 else 0

    order = np.argsort(proba_buy)
    sell_idx = order[:n_sell]
    buy_idx = order[-n_buy:]

    signals = np.array(["HOLD"] * n, dtype=object)
    signals[sell_idx] = "SELL"
    signals[buy_idx] = "BUY"
    return signals


def attach_signals(
    test_df: pd.DataFrame,
    test_proba: np.ndarray,
    *,
    threshold_mode: str = "fixed",
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
    buy_quantile: float = 0.7,
    sell_quantile: float = 0.3,
) -> pd.DataFrame:
    """
    Return a copy of `test_df` with `Proba` and `Signal` columns attached.
    """
    out = test_df.copy()
    proba = np.asarray(test_proba, dtype=float)
    # Keep probabilities finite so downstream thresholding and plotting remain stable.
    proba = np.where(np.isfinite(proba), proba, 0.5)
    proba = np.clip(proba, 0.0, 1.0)
    out["Proba"] = proba

    proba = out["Proba"].values
    mode = threshold_mode.lower().strip()
    if mode == "fixed":
        out["Signal"] = prob_to_signal(proba, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    elif mode == "quantile":
        out["Signal"] = prob_to_signal_quantile(proba, buy_quantile=buy_quantile, sell_quantile=sell_quantile)
    else:
        raise ValueError("threshold_mode must be 'fixed' or 'quantile'")

    return out


def enforce_trade_frequency(
    signals_df: pd.DataFrame,
    *,
    max_trades_per_day: int = 12,
    min_minutes_between_trades: int = 5,
) -> pd.DataFrame:
    """
    Limit excessive intraday churn while allowing multiple trades per day.

    A "trade" here is a BUY or SELL action. Signals that violate limits are
    converted to HOLD.
    """
    if max_trades_per_day <= 0:
        raise ValueError("max_trades_per_day must be >= 1")
    if min_minutes_between_trades < 0:
        raise ValueError("min_minutes_between_trades must be >= 0")

    required = ["Date", "Signal"]
    missing = [c for c in required if c not in signals_df.columns]
    if missing:
        raise ValueError(f"signals_df missing columns: {missing}")

    out = signals_df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").reset_index(drop=True)
    out["Signal"] = out["Signal"].astype(str).str.upper().str.strip()

    last_trade_ts: pd.Timestamp | None = None
    trades_by_day: dict[pd.Timestamp, int] = defaultdict(int)

    for idx, row in out.iterrows():
        signal = row["Signal"]
        ts = row["Date"]
        day_key = pd.Timestamp(ts.date())

        if signal not in {"BUY", "SELL"}:
            continue

        if trades_by_day[day_key] >= max_trades_per_day:
            out.at[idx, "Signal"] = "HOLD"
            continue

        if last_trade_ts is not None and min_minutes_between_trades > 0:
            mins_since = (ts - last_trade_ts).total_seconds() / 60.0
            if mins_since < float(min_minutes_between_trades):
                out.at[idx, "Signal"] = "HOLD"
                continue

        trades_by_day[day_key] += 1
        last_trade_ts = ts

    return out


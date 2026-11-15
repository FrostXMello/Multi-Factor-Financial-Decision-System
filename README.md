# Multi-Factor Financial Decision System

## Project Status
Active

## Quickstart
- Recommended Python: 3.12+
- Install dependencies with `python -m pip install -r requirements.txt`
- Launch app with `streamlit run app.py`

End-to-end demo pipeline:

1. Download multi-timeframe OHLCV data with `yfinance` (supports `1m` base bars)
2. Build technical features on each timeframe and align them to a base execution timeframe
3. Train a classifier (Gradient Boosting / Logistic Regression / Random Forest) to predict forward return direction over configurable bars
4. Convert predicted probability to `BUY` / `SELL` / `HOLD`
5. Apply risk gating + intraday trade-frequency controls
6. Backtest a simple “all-in long / cash” strategy
6. Visualize results in a Streamlit dashboard

Now includes a four-model stack:

- Technical model (GB/LogReg/RF) with walk-forward validation
- Macro model (index regime + macro-news sentiment proxy)
- Micro model (fundamental snapshot + company-news sentiment proxy)
- Risk model (volatility/drawdown gating + position sizing)
- Fusion layer (weighted blend or optional trainable meta-fusion)
- Regime-aware optimizer (searches thresholds, horizon bars, trade frequency, and risk hard-stops)
- Portfolio optimizer mode (single global configuration tuned on aggregated basket equity)
- Comprehensive benchmark suite (multi-model, multi-timeframe, multi-horizon leaderboard)
- Data health diagnostics and CSV exports for benchmark/portfolio outputs
- Clean data API for candles + merged features + metadata quality report
- NIFTY50 fundamentals collector with local cached store
- Two-stage portfolio workflow: research (fast, 5m subset) then execution validation (full, 1m universe)

## Setup

```bash
pip install -r requirements.txt
```

## Run Streamlit UI

```bash
streamlit run app.py
```

## TECHNICAL_MODE and CORE_MODE

Both flags live in `core_config.py`. **`TECHNICAL_MODE = True` is the safe default** and it is checked first. `CORE_MODE` only applies when `TECHNICAL_MODE` is already `False`.

| Area | `TECHNICAL_MODE = True` (default) | Both flags `False` (research) |
|------|-----------------------------------|-------------------------------|
| Data | Daily bars plus one weekly context series. Yahoo adjusted OHLC. | Intervals chosen in the app |
| Features | Price-only: MA20, MA50, price vs MA50, RSI14, ROC, daily return, 10-day vol | Strategy-specific features; multi-factor can add higher-timeframe columns |
| Labels | Forward return over `CORE_HORIZON_BARS` (default 7) above 1%. The last horizon rows stay scoreable with a null target | Same label builder |
| Model | One gradient boosting classifier, chronological split, embargo of `horizon_bars` | GB / logistic / random forest. Walk-forward uses the same embargo |
| Signals | `P > 0.6` and close above MA50 to buy; `P < 0.4` and close below MA50 to sell | Fixed or expanding-window quantile thresholds |
| Macro / micro / fusion / optimizer | Not called | Available. Meta-fusion is still an in-sample stack and stays off unless you opt in |
| Backtest | Next-session open fills, Indian cash-equity cost assumptions, net and gross returns | Same fill and cost engine |

Setting only `CORE_MODE = False` does **not** leave the technical path. To open the research stack, set **both** `TECHNICAL_MODE = False` and `CORE_MODE = False`, then restart Streamlit. Leave both `True` until that research path is reviewed.

## Execution, labels, and validation

- Features on bar `t` use information through the close of `t`.
- The training label is the forward close `t + horizon`. Rows without that close are not used for fitting, and `score_features` can score them with no `Target` column.
- The train/test cut drops the last `horizon_bars` training labels so those labels cannot use a test-period close. Walk-forward folds use the same purge.
- A signal on bar `t` fills at the **next bar's open**. It never fills at the close that created the signal. If that open is missing, the fill is skipped. The last bar is a signal only.
- Equity is marked at each bar's close after that open fill.
- Buy-and-hold buys at the first valid open of the same window. Neither book is force-sold on the final close.
- Sharpe, Sortino, and volatility use 252 for daily bars, 52 for weekly bars, and `252 * (6.25 hours / bar length)` for intraday bars. CAGR uses calendar time between the first and last equity dates.
- Reported total return is **net** of costs. Gross return, cost total, turnover, closed-trade count, win rate, profit factor, CAGR, Sharpe, Sortino, and max drawdown are on the backtest result. The benchmark is buy-and-hold on the same window.

## Transaction-cost assumptions

`transaction_costs.py` holds configurable NSE cash-equity assumptions. They are **not** a live broker tariff.

Delivery preset (default): brokerage 3 bps per side, STT 0.10% buy and sell, exchange charge 0.00297% per side, SEBI 0.0001% per side, stamp duty 0.015% on buys, GST 18% on brokerage + exchange + SEBI, slippage 5 bps. Intraday preset: STT 0.025% on sells only, stamp 0.003% on buys. Pass `IndiaEquityCostConfig.zero()` or `apply_costs=False` to inspect a frictionless fill. Confirm rates before quoting them as a broker's current schedule.

## Survivorship

NIFTY 50, NIFTY Midcap 150, and the sector stock lists are **today's** members (`utils.py`, `sector_universe.py`). A historical test therefore keeps survivors and omits names that left the index. Treat index-basket results as optimistic versus a point-in-time universe.

Prices use Yahoo `auto_adjust=True`, so open and close are split- and dividend-adjusted together. Invalid OHLC bars (range that does not contain open/close, non-positive prices, negative volume) are dropped.

## Notes

- This project intentionally combines Macro + Micro + Fusion + Optimizer + Intraday layers; this improves flexibility but can increase over-engineering risk if not validated with strict out-of-sample testing.
- We use proxy data sources due to academic constraints: Yahoo intraday coverage can be limited/noisy and fundamentals from metadata are not guaranteed point-in-time clean.
- Meta-fusion currently uses in-sample stacking in parts of the flow; future improvement includes out-of-fold stacking to reduce bias.
- Optimizer overfitting risk is managed with constrained search spaces and Fast mode defaults, but parameter search should still be treated as potentially optimistic.
- Splits train/test chronologically and embargoes `horizon_bars` so training labels do not use test closes.
- Technical model supports multi-timeframe analysis aligned to the selected execution interval.
- You can run down to `1m` bars and forecast over `horizon_bars` (not just next day).
- Trade engine supports multiple intraday trades with per-day cap and minimum spacing.
- Risk gating can force HOLD in high-volatility/high-drawdown regimes.
- Optional optimizer in UI can auto-select stronger parameter sets for current market-volatility regime.
- In NIFTY portfolio mode, optimizer can tune one global setup across the basket instead of per-ticker settings.
- App now includes top-level requirements/data-health diagnostics and downloadable comparison tables.
- App now includes price + fundamentals quality tables for both single and portfolio flows.

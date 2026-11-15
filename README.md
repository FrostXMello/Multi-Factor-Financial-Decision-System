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
| Model | One gradient boosting classifier. Purged walk-forward is the reported score. A chronological holdout is kept beside it | GB / logistic / random forest. The research path still uses its own holdout |
| Signals | Calibrated `P > 0.6` and close above MA50 to buy; calibrated `P < 0.4` and close below MA50 to sell. Raw `P` is used if calibration falls back | Fixed or expanding-window quantile thresholds |
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

## Walk-forward validation

The technical path treats purged expanding-window walk-forward as the primary score. The single chronological holdout from Milestone 1 is still computed and labeled as a diagnostic.

- Labeled rows are ordered by date. The first half is the minimum training prefix. Later labeled rows are split into contiguous test windows.
- Every fold drops the last `horizon_bars` training labels before its test window. Those labels would use a close inside the test window.
- The model for a fold is fit only on rows before that embargo. Test labels are used after the prediction, for scoring only.
- Out-of-sample probabilities are concatenated across folds. The equity curve on the technical path is built from those probabilities, then the same next-open fill and cost model as Milestone 1.
- The unlabeled tail, including the latest bar, is scored by the last fold's model. That model was not trained on the tail, and it was not trained on labels that reach the tail.
- Feature formulas are causal rolling statistics and a backward as-of merge. They are built once. The walk-forward refits the classifier and the calibrator, not a full-sample scaler or a future-looking feature.
- Fold count and tree count are fixed in configuration. They are not chosen by looking at out-of-sample return.

If a history is too short to form those folds, the run says so and falls back to the chronological holdout.

## Probability calibration

Calibration is chronological and happens inside each training window.

- The pre-test rows are split into an earlier fit slice and a later calibration slice, with the same horizon embargo between them and between the calibration slice and the test fold.
- The classifier is fit on the fit slice. Sigmoid calibration (Platt scaling) is fit on the calibration slice when it has at least 25 rows and both classes. Isotonic calibration is fit when that slice has at least 80 rows and at least 8 rows of each class.
- `auto` uses isotonic for the decision probability when the slice was large enough, otherwise sigmoid, otherwise the raw probability. The choice does not look at the test fold.
- Raw, sigmoid, and isotonic probabilities are scored on the out-of-sample rows when each map was fit. The UI reports Brier score, log loss, and expected calibration error, plus a reliability diagram.
- BUY, SELL, and HOLD use the decision probability. Turning calibration to `none`, or hitting a slice that fails the size rule, leaves the raw probability unchanged.
- A calibrator's error on its own calibration window is labeled as validation. It is not an out-of-sample number.

## Explainability

HistGradientBoosting does not expose a native importance vector. Importance is the drop in negative Brier score when a feature is shuffled on a fold's test rows. Those rows were not used to fit that fold's model. Fold means and the standard deviation across folds are both shown.

The latest bar also has a sensitivity table: each feature is replaced, one at a time, by the median of the last held-out fold, and the probability is scored again. `ProbabilityDelta` is the original probability minus the replaced one. That is a property of this fitted model. It is not a statement that the feature caused the return, and it ignores dependence between features. The table is omitted when the held-out reference is too small. No narrative reason is generated.

## What a probability is, and what a profit is

A predicted probability is a score between 0 and 1. Calibration asks whether the scores match observed frequencies on later rows. A low Brier score does not mean the next trade makes money. Classification accuracy, ROC-AUC, and PR-AUC answer a different question from net return.

The trading result applies a 0.6 / 0.4 threshold, an MA50 trend gate, a risk gate that blocks entries only, a one-bar delay to the next open, and the Indian equity cost assumptions. Costs, skipped opens, and the trend gate can make a more accurate classifier less profitable, or the reverse. Read the net equity curve for the trading question and the reliability tab for the classification question. Neither number is a forecast of live brokerage performance.

## Transaction-cost assumptions

`transaction_costs.py` holds configurable NSE cash-equity assumptions. They are **not** a live broker tariff.

Delivery preset (default): brokerage 3 bps per side, STT 0.10% buy and sell, exchange charge 0.00297% per side, SEBI 0.0001% per side, stamp duty 0.015% on buys, GST 18% on brokerage + exchange + SEBI, slippage 5 bps. Intraday preset: STT 0.025% on sells only, stamp 0.003% on buys. Pass `IndiaEquityCostConfig.zero()` or `apply_costs=False` to inspect a frictionless fill. Confirm rates before quoting them as a broker's current schedule.

## Survivorship

NIFTY 50, NIFTY Midcap 150, and the sector stock lists are **today's** members (`utils.py`, `sector_universe.py`). A historical test therefore keeps survivors and omits names that left the index. Treat index-basket results as optimistic versus a point-in-time universe.

Prices use Yahoo `auto_adjust=True`, so open and close are split- and dividend-adjusted together. Invalid OHLC bars (range that does not contain open/close, non-positive prices, negative volume) are dropped.

## Navigation

Sidebar sections:

1. **Overview** — system summary, data status, paper-account snapshot  
2. **Market Data** — providers, SQLite coverage, collection  
3. **Research & Predictions** — CORE / technical pipeline  
4. **Backtesting / Reliability** — walk-forward payload from the current session  
5. **Paper Trading** — isolated virtual account + Demo / Replay  
6. **Settings / Diagnostics** — paths, env checklist, verification status  

For a closed-market academic walkthrough, see [`DEMO_GUIDE.md`](DEMO_GUIDE.md).

## Market data

The technical research path reads daily and weekly bars from a local SQLite file and backfills anything missing from Yahoo Finance. The feature builder is unchanged: higher timeframes are still joined with a backward as-of merge, and bars after an optional `as_of` are not returned.

Default database path: `data/market.sqlite`. Override it with `MARKET_DATA_DB`. The file is gitignored. Schema version is stored in `schema_migrations`. Candle identity is `(symbol, timeframe, bar_time)`, so a repeated backfill updates the bar instead of inserting a second copy.

Yahoo rows are **delayed research history**, not a live NSE feed. The Market Data screen says so. A live quote is stored only when a configured broker response sets `freshness` to `live`. Missing credentials send no request and create no quote. Copy `.env.example` to `.env` for the variable names. Do not commit `.env`.

The collector requests the watchlist only while the NSE cash clock says the session is open (09:15–15:30 IST, weekdays). That clock does not include the exchange holiday list. Outside the session it records that no quote was requested. Failed requests retry with backoff, then the error is stored. A WebSocket URL can be configured, but this install does not open a socket; polling is the fallback when a quote URL and credentials exist.

These repositories were reviewed and **not** installed:

| Repository | Why it was not integrated |
|---|---|
| `TheHardeep/fenix` | GPL-3.0 broker **order** adapter. Python 3.10+. Order placement is out of scope, and the license would cover this app. |
| `aeron7/nsepython` | Unofficial NSE website client. Not an authorized production feed. |
| `Sampad-Hegde/Bharat-SM-Data` | Self-described scraper, even though the license is Apache-2.0. |
| `Apurv-Salunke/india-stocks-api` | MIT and closer to a broker facade, but Python 3.10+, Angel One only, and it also places orders. No credentials were available to verify it. |

The live adapter is a small HTTP client for an official quote URL you supply. It was not connected to Zerodha, Angel One, or any other broker in this milestone.

**M3 implementation and offline validation are complete. Live NSE-session quote ingestion remains pending verification during the next market session.**

## Paper trading

`paper_trading/` is an isolated virtual broker. It never imports a live order API and never submits real orders.

- Normal paper account id: `paper:default` in `data/paper.sqlite` (`PAPER_TRADING_DB`).  
- Demo / Replay account id: `demo:academic`. Demo fills use deterministic fixtures labelled `demo` and cannot write into the normal paper book.  
- Fills reuse `IndiaEquityCostConfig` (brokerage, statutory-style charges, slippage).  
- Risk gates: cash, holdings (no short by default), max order value, max position value, portfolio exposure, daily loss, duplicate client ids, confirmation required by default.  
- Model signals map through `submit_signal`; auto-execution is off unless explicitly enabled.

Offline tests live in `tests/test_paper_trading.py` and `tests/test_market_*.py`. Run:

```bash
.venv/bin/python -m pytest -q tests
```

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

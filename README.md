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

## CORE_MODE vs research mode

`core_config.CORE_MODE` (default **True**) turns the stack into an **evaluation-ready** path: fewer knobs, chronological train/test only, and no optimizer in the UI or pipeline entry points.

| Area | CORE_MODE = True | CORE_MODE = False (research / PBL-3, PBL-4) |
|------|------------------|---------------------------------------------|
| Data | Daily base + one higher timeframe (`1d` + `1wk`); clean alignment | Multi-timeframe / intraday as configured in the app |
| Features | **Strategy-dependent**: Multi-Factor → MA20, MA50, RSI14, Daily_Return, Rolling_Volatility_10; **Momentum** → MA20/MA50, RSI14, ROC, price vs MA50; **Mean Reversion** → rolling mean/std, Z-score, RSI14 (no `TF_*`) | Full `TF_*` set only for **Multi-Factor** when `core_mode` is off |
| Models | Single type: gradient boosting (`CORE_MODEL_TYPE`); `predict_proba` (Mean Reversion uses rules for signals; model proba is diagnostic) | GB / LogReg / RF; optional walk-forward |
| Micro / fundamentals | Disabled at runtime (stub probabilities) | Micro model + fundamentals flows active |
| Fusion | **Technical only** (no macro in CORE). Momentum / Mean Reversion skip fusion. | Multi-Factor: weighted blend + optional **trainable meta-fusion** |
| Strategy logic | **Multi-Factor:** fixed P>0.6 BUY, P<0.4 SELL. **Momentum:** P gates + trend/RSI confirmation. **Mean Reversion:** Z-score thresholds. | Quantile or fixed thresholds for Multi-Factor from UI |
| Optimizer | **Disabled** (`assert_optimizer_allowed` on optimizer APIs) | Regime-aware grid search, portfolio optimizer, benchmark suite |
| Backtest / UI | Metrics + equity vs **buy & hold**; simplified Streamlit | Full diagnostics, histograms, benchmark CSVs |

**Why this improves robustness for academic demos:** fewer correlated features and no searchable meta-learner reduce optimistic in-sample fit; dropping micro/fundamentals avoids weak point-in-time metadata; a single classifier type and fixed fusion weights make the story linear (“features → proba → gates → PnL”).

**Re-enabling advanced behavior (e.g. PBL-3, PBL-4):** set `CORE_MODE = False` in `core_config.py`, restart the app, switch to **Advanced Research**, and use the existing modules (optimizer, meta-fusion, intraday, micro) unchanged.

## Notes

- This project intentionally combines Macro + Micro + Fusion + Optimizer + Intraday layers; this improves flexibility but can increase over-engineering risk if not validated with strict out-of-sample testing.
- We use proxy data sources due to academic constraints: Yahoo intraday coverage can be limited/noisy and fundamentals from metadata are not guaranteed point-in-time clean.
- Meta-fusion currently uses in-sample stacking in parts of the flow; future improvement includes out-of-fold stacking to reduce bias.
- Optimizer overfitting risk is managed with constrained search spaces and Fast mode defaults, but parameter search should still be treated as potentially optimistic.
- Splits train/test chronologically to avoid leakage.
- Technical model supports multi-timeframe analysis aligned to the selected execution interval.
- You can run down to `1m` bars and forecast over `horizon_bars` (not just next day).
- Trade engine supports multiple intraday trades with per-day cap and minimum spacing.
- Risk gating can force HOLD in high-volatility/high-drawdown regimes.
- Optional optimizer in UI can auto-select stronger parameter sets for current market-volatility regime.
- In NIFTY portfolio mode, optimizer can tune one global setup across the basket instead of per-ticker settings.
- App now includes top-level requirements/data-health diagnostics and downloadable comparison tables.
- App now includes price + fundamentals quality tables for both single and portfolio flows.

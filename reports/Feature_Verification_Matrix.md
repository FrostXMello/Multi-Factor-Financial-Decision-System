# Feature Verification Matrix

**Project:** Multi-Factor Financial Decision System  
**Audit HEAD:** `fe1619ad6683acdaadaecebbd25a7c0f0c24038b` (2026-11-16)  
**Evidence commands:** `.venv/bin/python -m pytest -q tests` → **81 passed** (5.02s); `cd frontend && npm run build` → **success**  
**Deployed frontend observed:** https://multi-factor-decision-system.vercel.app (HTTP 200)  
**Local API at audit time:** `http://127.0.0.1:8000` — **unavailable** (connection refused)

Status legend:

| Status | Meaning |
|---|---|
| IMPLEMENTED AND VERIFIED | Code present; automated tests and/or audit observation confirm behaviour |
| IMPLEMENTED BUT NOT FULLY VERIFIED | Code present; limited tests, mocked paths, or no live-provider confirmation |
| PARTIALLY IMPLEMENTED | Subset works; major path incomplete or unreachable |
| PLANNED / NOT IMPLEMENTED | Documented or stubbed; not available in runtime |
| UNAVAILABLE | Cannot be demonstrated with current deploy/config |

---

## A. Product surfaces

| Feature | Expected | Actual | Status | Evidence |
|---|---|---|---|---|
| React Live Analysis tab | Scanner UI | Route `/` → `LiveAnalysisPage` | IMPLEMENTED AND VERIFIED | `frontend/src/App.tsx`, Vercel screenshot `assets/live-analysis-vercel.png` |
| React Backtesting tab | Strategy simulation UI | Route `/backtesting` → `BacktestingPage` | IMPLEMENTED AND VERIFIED | `App.tsx`, screenshot `assets/backtesting-vercel.png` |
| Three-tab architecture (Live / Portfolio / Backtesting) | Per earlier milestone | **Two tabs only**; `/portfolio` and `/research` redirect to `/backtesting` | PARTIALLY IMPLEMENTED | `App.tsx` links array length 2; redirects L83–84 |
| Portfolio / Research / Overview / Paper UI pages | Documented in README/DEMO guides | No page components; API wrappers orphaned | PLANNED / NOT IMPLEMENTED (UI) | `pages.tsx` exports only Live, Backtesting, Settings |
| Streamlit academic UI | `app.py` | Present as alternate surface | IMPLEMENTED BUT NOT FULLY VERIFIED | File exists; not part of Vercel product |
| Settings drawer | Diagnostics | Present | IMPLEMENTED AND VERIFIED | `SettingsPanel` in `pages.tsx` |

---

## B. Market data & freshness

| Feature | Status | Evidence / notes |
|---|---|---|
| Yahoo historical candles via yfinance | IMPLEMENTED AND VERIFIED | `market_data/providers.py` `YFinanceHistoryProvider`; store tests; `data/market.sqlite` has 7032 candle rows (audit) |
| SQLite candle/quote persistence | IMPLEMENTED AND VERIFIED | Tables: candles, quotes, instruments, providers, quality_events, provider_events |
| OHLC quality checks | IMPLEMENTED AND VERIFIED | `market_data/quality.py`; `tests/test_market_*` |
| NSE session gating (09:15–15:30 IST weekdays) | IMPLEMENTED AND VERIFIED | `market_data/session.py`; no holiday calendar |
| Distinguish delayed / historical / simulated / live labels | IMPLEMENTED BUT NOT FULLY VERIFIED | Types in `market_data/types.py` + paper layer; Live UI shows LIVE LOCKED correctly on Vercel |
| Verified live NSE quotes during market hours | UNAVAILABLE | `BrokerLiveProvider.status()` never reports connected live; `quotes` table empty; Vercel Live shows API 404 without backend |
| WebSocket streaming client | PLANNED / NOT IMPLEMENTED | `BROKER_LIVE_WS_URL` documented; no bundled socket client |
| Persistent market scanner daemon | PARTIALLY IMPLEMENTED | `live_analysis/scanner.py` starts on API lifespan; scores Yahoo daily bars; LIVE mode unreachable; Vercel has no API so scanner not running for deployed UI |

---

## C. Machine learning & signals

| Feature | Status | Evidence / notes |
|---|---|---|
| Technical features (MA, RSI, ROC, vol, weekly TF_*) | IMPLEMENTED BUT NOT FULLY VERIFIED | `feature_engineering.py`; **weekly merge-asof without shift can leak future week close** (audit finding) |
| Binary label: 7-day forward return > 1% | IMPLEMENTED AND VERIFIED | `core_config.CORE_HORIZON_BARS=7`, `TECHNICAL_MIN_FORWARD_RETURN=0.01` |
| HistGradientBoosting classifier | IMPLEMENTED AND VERIFIED | `model.py` `_build_classifier`; TECHNICAL_MODE |
| Purged chronological split / walk-forward | IMPLEMENTED AND VERIFIED | `model.purged_split_bounds`, `model_evaluation.plan_walk_forward_folds`; reliability tests |
| Probability calibration (isotonic/sigmoid) | IMPLEMENTED AND VERIFIED | Walk-forward calibrator on pre-test slice |
| Macro / micro / fusion stack | PARTIALLY IMPLEMENTED | Modules exist; `TECHNICAL_MODE`/`CORE_MODE` disable runtime use; micro/macro snapshots have look-ahead risks if enabled |
| BUY/SELL/HOLD technical rule (P + MA50) | IMPLEMENTED AND VERIFIED | `strategy.attach_technical_trend_signals`; buy 0.6 / sell 0.4 |
| Momentum / mean-reversion modes | IMPLEMENTED BUT NOT FULLY VERIFIED | `strategy_modes.py`; no dedicated unit tests calling attach_* functions |
| Risk gating can block BUY | IMPLEMENTED AND VERIFIED | `risk_model.apply_risk_gating`; `tests/test_execution_risk_costs.py` |
| End-to-end Live BUY/SELL/HOLD on Vercel | UNAVAILABLE | Live API 404 on static deploy; no candidates without backend |

---

## D. Backtesting & strategy research

| Feature | Status | Evidence / notes |
|---|---|---|
| Core long/cash next-open backtest engine | IMPLEMENTED AND VERIFIED | `backtest.backtest_long_cash`; cost/risk tests |
| Indian equity cost assumptions | IMPLEMENTED AND VERIFIED | `transaction_costs.IndiaEquityCostConfig` |
| FastAPI `/api/backtest/*` routes | IMPLEMENTED BUT NOT FULLY VERIFIED | Routes exist; equity_curve Series branch dead for DataFrame returns; window clip does not recompute metrics |
| Configurable strategy parameters via API | IMPLEMENTED BUT NOT FULLY VERIFIED | `/api/backtest/run` accepts thresholds/costs; React UI does not call it |
| React Backtesting uses real historical backtest | UNAVAILABLE / MISLEADING | `BacktestingPage` uses `buildShowcaseDemo` only — seeded PRNG, no prices, no model |
| Adjust params and rerun (UI) | PARTIALLY IMPLEMENTED | UI changes ticker/dates/capital and regenerates **synthetic** paths only |
| Multi-layer / sector-rotation backtests | PARTIALLY IMPLEMENTED | Offline modules; not exposed on React or FastAPI live/backtest product path |
| Strategy research (regimes, portfolio, robustness) | IMPLEMENTED AND VERIFIED | `strategy_research/*`; `tests/test_strategy_research.py` (5); no React pages |

---

## E. Paper trading & safety

| Feature | Status | Evidence / notes |
|---|---|---|
| Paper ledger SQLite | IMPLEMENTED AND VERIFIED | `paper.sqlite` accounts `paper:default`, `demo:academic` |
| Isolation from broker order APIs | IMPLEMENTED AND VERIFIED | Engine has no broker order path; `broker_execution: false` hard-coded in settings |
| Risk limits on paper orders | IMPLEMENTED AND VERIFIED | `paper_trading/risk.py`; tests; note: daily-loss can block SELL |
| Demo workflow API | IMPLEMENTED AND VERIFIED | `POST /api/paper/demo/run` |
| Paper UI in React | PLANNED / NOT IMPLEMENTED | Wrappers in `api.ts` unused by pages |

---

## F. AI explainability

| Feature | Status | Evidence / notes |
|---|---|---|
| Deterministic explanations from structured fields | IMPLEMENTED AND VERIFIED | `reasoning/deterministic.py`; `tests/test_reasoning.py` |
| LLM enrichment (optional) | IMPLEMENTED BUT NOT FULLY VERIFIED | Requires `AI_REASONING_*`; default blank → deterministic |
| Cannot override trading signals | IMPLEMENTED AND VERIFIED | Schema lock + overwrite of decision from base; no feedback into orders |
| React Explain panel usable on Live without research | PARTIALLY IMPLEMENTED | Needs cached `/api/research/run`; UI has no research runner |

---

## G. Deployment & security

| Feature | Status | Evidence / notes |
|---|---|---|
| Frontend Vercel static deploy | IMPLEMENTED AND VERIFIED | Site HTTP 200; SPA routes work |
| Backend Render/Railway configs | IMPLEMENTED BUT NOT FULLY VERIFIED | `render.yaml`, `railway.toml`, `Dockerfile` present; **no verified production API URL in this audit** |
| `VITE_API_BASE_URL` wired for Live on Vercel | UNAVAILABLE / misconfigured | Live tab API 404 on production URL |
| API authentication | PLANNED / NOT IMPLEMENTED | All endpoints open |
| Secrets not in git | IMPLEMENTED AND VERIFIED | `.env.example` empty credentials; `.env` gitignored |
| CI pytest | IMPLEMENTED AND VERIFIED | `.github/workflows/ci.yml`; local 81 passed |
| Frontend CI build | PLANNED / NOT IMPLEMENTED | CI does not build frontend |

---

## H. Explicit investigation answers

| Question | Answer |
|---|---|
| Is live NSE data genuinely available and verified during market hours? | **No.** Live provider path unconfigured / never verified; Live UI locked; quotes table empty. |
| Are simulated, delayed, historical correctly distinguished? | **Mostly yes in backend/paper contracts**; React Backtesting currently presents synthetic data without a clear DEMO badge (post `fe1619a`). |
| Does market scanner operate persistently in the backend? | **Yes, when API process runs** (`lifespan` → `start_background`). Not available for Vercel-only frontend. `scans.sqlite` had 0 runs at audit. |
| Do BUY/SELL/HOLD predictions work end-to-end? | **Backend pipeline yes (tested offline)**; **deployed React Live path no** without API. |
| Can risk checks block unsuitable signals? | **Yes** for research/paper BUY gates (verified by tests). |
| Does backtesting work on arbitrary supported stocks with history? | **Engine/API can**, subject to Yahoo/store coverage; **React Backtesting does not use history**. |
| Can strategy parameters be adjusted and simulations rerun? | **API yes**; **UI only adjusts synthetic showcase seed inputs**. |
| Do AI explanations use structured outputs and not override signals? | **Yes** (verified by `tests/test_reasoning.py`). |
| Is paper trading isolated from real broker execution? | **Yes**. |
| Are frontend APIs connected to a deployed backend? | **No evidence of working production API** for Vercel Live (404). Backtesting does not need API. |
| Is the latest three-tab architecture implemented? | **No — two tabs** (Live + Backtesting). |

---

## I. Test & build evidence (this audit)

```
Command: .venv/bin/python -m pytest -q tests
Result:  81 passed, 15 warnings in 5.02s
Exit:    0

Command: cd frontend && npm run build
Result:  tsc -b && vite build — success (dist assets written)
Exit:    0
```

Raw logs: `reports/_evidence/pytest_output.txt`, `reports/_evidence/frontend_build.txt`.

# Architecture Documentation

**Project:** Multi-Factor Financial Decision System  
**API version:** 0.9.0 (`api/main.py`)  
**Audit HEAD:** `fe1619a`  
**Figures:** screenshots in `reports/assets/`

---

## 1. System context

The system is a **research and simulation stack for Indian equities**, not a brokerage. Two primary user-facing surfaces exist:

1. **React terminal** (Vite + React Router) — deployed as a static SPA on Vercel.
2. **FastAPI backend** — wraps Python research modules; intended for Render/Railway with persistent disk.

An optional **Streamlit** app (`app.py`) remains for academic demos of older multi-page flows.

```
┌─────────────────────┐     HTTPS (optional VITE_API_BASE_URL)
│  React SPA (Vercel) │────────────────────────────────────────┐
│  Live · Backtesting │                                        │
└─────────────────────┘                                        ▼
                                                    ┌──────────────────────┐
                                                    │ FastAPI 0.9.0        │
                                                    │ uvicorn --workers 1  │
                                                    └──────────┬───────────┘
                         ┌─────────────────────────────────────┼──────────────────────────┐
                         ▼                                     ▼                          ▼
              ┌──────────────────┐               ┌─────────────────────┐     ┌────────────────────┐
              │ market_data/*    │               │ four_model_pipeline │     │ live_analysis      │
              │ Yahoo / Broker   │               │ + backtest / risk   │     │ MarketScanner      │
              └────────┬─────────┘               └──────────┬──────────┘     └─────────┬──────────┘
                       ▼                                    ▼                          ▼
              data/market.sqlite                   in-process caches            data/scans.sqlite
              data/paper.sqlite                    (_LAST_RESULT, etc.)
```

**Important runtime fact:** On the audited Vercel deployment, Live Analysis calls relative `/api/*` and receives **404** (no backend). Backtesting runs **entirely in the browser** via `showcaseDemo.ts` and does not call the API.

---

## 2. Frontend architecture

### 2.1 Navigation (actual)

| Route | Component | Notes |
|---|---|---|
| `/`, `/live` | `LiveAnalysisPage` | Calls `/api/live/*`, `/api/market/candles/{symbol}` |
| `/backtesting` | `BacktestingPage` | Client-side `buildShowcaseDemo` only |
| `/portfolio`, `/research` | Redirect → `/backtesting` | Legacy URLs |
| `*` | Redirect → `/` | |

Source: `frontend/src/App.tsx`.

### 2.2 Key modules

| File | Role |
|---|---|
| `App.tsx` | Shell, nav, health/live status pills |
| `pages.tsx` | Live + Backtesting pages, Settings drawer |
| `api.ts` | Typed HTTP client; `VITE_API_BASE_URL` |
| `charts.tsx` | TradingView `lightweight-charts` candle + multi-equity |
| `showcaseDemo.ts` | Deterministic synthetic strategy board |
| `components.tsx` | Badges, tables, `AiInsight` explain panel |
| `hooks.ts` | `useLoad` async loader |

### 2.3 Chart integration

- Library: **lightweight-charts ^5.3.0** (TradingView).
- `CandleChart`: OHLC + volume + optional MA20/MA50.
- `MultiEquityChart`: multiple `LineSeries` for side-by-side equity.

### 2.4 Frontend data flow — Live

```
User → Scan now → POST /api/live/scan
                 → GET /api/live/status, /api/live/candidates
                 → GET /api/market/candles/{symbol}
                 → (optional) POST /api/explain/decision
```

### 2.5 Frontend data flow — Backtesting (current)

```
User → Run simulation → buildShowcaseDemo({ticker,start,end,capital})
                      → local state: profit_board, comparison_equity, orders
                      → NO network / NO FastAPI / NO historical prices
```

---

## 3. Backend architecture

### 3.1 Application entry

- `api/main.py` creates FastAPI app v0.9.0.
- Lifespan: ensure paths → open stores (migrations) → `get_scanner_service().start_background()`.
- CORS from `API_CORS_ORIGINS`.
- `broker_execution` always `false` in public config.

### 3.2 Routers

| Router module | Prefix / focus |
|---|---|
| `routes_overview` | `/api/health`, `/ready`, `/overview`, `/diagnostics` |
| `routes_market` | market status, quotes, candles, collect, coverage, quality |
| `routes_research` | `/api/research/run|last|reliability` |
| `routes_paper` | paper/demo accounts, orders, reset |
| `routes_explain` | explain status/decision/last/risk-block |
| `routes_analytics` | regimes, portfolio, compare, robustness, report |
| `routes_live` | scanner status, candidates, scan, history |
| `routes_backtest` | defaults, symbols, run, simulate-strategies, runs, compare |

Full inventory: see `Technical_Audit.md` § FastAPI.

### 3.3 Service layer (Python modules)

```
HTTP routes
   ├── market_data/{providers,ingest,store,session,quality,research_loader}
   ├── four_model_pipeline ← feature_engineering, model, model_evaluation,
   │                         strategy/strategy_modes, risk_model, backtest,
   │                         transaction_costs, (optional macro/micro/fusion)
   ├── live_analysis/{scanner,store}
   ├── paper_trading/{engine,store,risk,prices,demo}
   ├── reasoning/{service,deterministic,llm,schema,context}
   └── strategy_research/{regimes,portfolio,strategy_compare,robustness,report}
```

---

## 4. Data pipeline

### 4.1 Ingestion

1. **Historical research:** `YFinanceHistoryProvider` → candles (freshness conceptually delayed/historical).
2. **Optional live quotes:** `BrokerLiveProvider` HTTP GET when env credentials set and session open.
3. **Research loader:** merges store + Yahoo backfill with `as_of` / cutoff guards.
4. **Scanner:** loads daily history per symbol and runs `run_four_model_pipeline`.

### 4.2 Feature & label pipeline (TECHNICAL_MODE)

```
OHLCV (1d) + optional 1wk context
        → MA20, MA50, Price_vs_MA50, RSI14, ROC_10, Daily_Return, Rolling_Volatility_10
        → TF_1wk_* context features (merge_asof backward)
        → Forward_Return = Close.shift(-7)/Close - 1
        → Target = 1 if Forward_Return > 0.01
        → HistGradientBoostingClassifier
        → purged walk-forward + optional calibration
        → Proba → Signal (BUY/SELL/HOLD) → risk gating → next-open backtest
```

### 4.3 Execution assumption

Signal on bar *t* fills at bar *t+1* **Open** (never signal-bar Close). Costs and slippage applied via `IndiaEquityCostConfig`.

---

## 5. Database design

Three SQLite files under `data/` (or `DATA_DIR` / `RENDER_DISK_PATH` in production).

### 5.1 `market.sqlite`

| Table | Purpose |
|---|---|
| providers | Provider registry |
| instruments | Symbol metadata |
| candles | OHLCV bars PK(symbol, timeframe, bar_time) |
| quotes | Tick/quote snapshots |
| quality_events | Data-quality findings |
| provider_events | Provider log |
| schema_migrations | Versioning |

### 5.2 `paper.sqlite`

| Table | Purpose |
|---|---|
| accounts | `paper:default`, `demo:*` isolation |
| ledger | Cash movements |
| positions | Open lots |
| orders | Order lifecycle |
| closed_trades | Realized P&L |
| equity_curve | Marked equity |
| schema_migrations | Versioning |

### 5.3 `scans.sqlite`

| Table | Purpose |
|---|---|
| scan_runs | Scanner run metadata |
| scan_results | Per-symbol signals/explanations |
| backtest_runs | Saved API backtest summaries |
| schema_migrations | Versioning |

**Relationships (logical):**

```
scan_runs 1──* scan_results
accounts 1──* orders | positions | ledger | closed_trades | equity_curve
instruments 1──* candles | quotes
```

SQLite has limited FK enforcement; paper store defines FKs in DDL.

---

## 6. Module interaction — research run

```
POST /api/research/run {ticker,...}
        → run_four_model_pipeline
        → feature_engineering.build_*
        → model train / walk-forward
        → strategy.attach_*_signals
        → risk_model.apply_risk_gating
        → backtest.backtest_long_cash
        → reasoning.explain_decision (optional LLM)
        → cache _LAST_RESULT (process memory)
```

---

## 7. Risk & paper isolation

```
Signal BUY ──NoTrade?──► HOLD (EntryBlocked)
         └──ok──► PositionSize ∈ [min, base]

Paper place_order
  → PriceQuote (store/demo; never live broker fill API)
  → validate_order_risk
  → ledger update in paper.sqlite only
```

Hard invariant: **no code path submits live broker orders.**

---

## 8. Explainability data flow

```
Structured latest bar {signal, proba, risk, metrics}
        → deterministic explanation
        → (optional) LLM enrich
        → validate_explanation(locked_signal=...)
        → response; decision.signal forced to locked signal
```

Explanations **cannot** write back into signals, risk, or orders.

---

## 9. Deployment topology

| Layer | Config | Observed |
|---|---|---|
| Frontend | `frontend/vercel.json` SPA rewrite excluding `/api` and `/assets` | https://multi-factor-decision-system.vercel.app → 200 |
| Backend | `Dockerfile`, `render.yaml`, `railway.toml`, `Procfile` | Config present; production API URL **not verified** in this audit |
| CI | `.github/workflows/ci.yml` pytest only | Frontend build not in CI |

**Assumption:** Single uvicorn worker is required because SQLite stores use shared connections (`check_same_thread=False`) and in-process caches.

---

## 10. Architecture risks (summary)

1. React Backtesting is a **presentation simulator**, not the FastAPI backtest engine.
2. Vercel Live depends on `VITE_API_BASE_URL`; without it Live is non-functional.
3. Weekly feature join may introduce **look-ahead bias** vs Yahoo Monday-stamped weekly bars.
4. Docs (`README`, `ARCHITECTURE`, `API`, `DEMO_GUIDE`) describe a larger UI than currently shipped.
5. No API authentication; mutating endpoints are open if the API is publicly exposed.

---

## 11. Figure catalogue

| ID | File | Description |
|---|---|---|
| Fig. A1 | `assets/backtesting-vercel.png` | Deployed Backtesting UI (strategy board) |
| Fig. A2 | `assets/live-analysis-vercel.png` | Deployed Live UI showing API 404 / LIVE LOCKED |

---

## 12. References (internal)

- `ARCHITECTURE.md` (repo root — partially stale vs two-tab UI)
- `API.md` (incomplete vs live/backtest routers)
- `DEPLOYMENT.md`
- Source modules listed in §3.3

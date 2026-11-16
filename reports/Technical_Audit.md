# Technical Audit Report

**Project:** Multi-Factor Financial Decision System (PBL-3)  
**Auditor role:** Senior software architecture / quant methodology review  
**Repository path:** `/Users/gagan/Documents/Projects/Multi-factor Financial Decision System`  
**Git HEAD audited:** `fe1619ad6683acdaadaecebbd25a7c0f0c24038b` — *Drop DEMO/fake-data headlines from Backtesting UI.*  
**Commit count:** 33  
**Audit date:** 2026-11-16 (local evidence under `reports/_evidence/`)  

**Scope rule:** Only behaviour evidenced by source, tests, builds, SQLite inspection, or live HTTP observation. No fabricated metrics or deployment claims.

---

## 1. Executive findings

1. The **backend research stack is real and substantial**: features, GB classifier, purged walk-forward, next-open fills, Indian cost model, risk gating, paper ledger, explainability with signal lock, and FastAPI routers for live/backtest/paper/research/analytics.
2. The **shipped React product is a two-tab terminal**. Live depends on a FastAPI host. Backtesting currently uses **client-side synthetic data** (`showcaseDemo.ts`) and does **not** call `/api/backtest/*`.
3. The **Vercel frontend is live** (HTTP 200). Live Analysis on that host shows **API 404** for `/api/live/status` and `/api/live/candidates` — **no verified production backend** attached in this audit.
4. **Verified live NSE tape is unavailable.** Broker live adapter exists but is unconfigured/unverified; `quotes` table empty; LIVE mode unreachable in scanner logic.
5. Automated suite: **81 passed**. Frontend production build: **success**.
6. Critical methodology issue: **weekly timeframe features may leak future information** when Yahoo `1wk` bars are Monday-stamped with week-ending closes.
7. Documentation (README / ARCHITECTURE / API / DEMO guides) is **stale** relative to the two-tab React UI and client-side backtesting.

---

## 2. Evidence pack

| Evidence | Result |
|---|---|
| `pytest -q tests` | 81 passed, 15 warnings, 5.02s, exit 0 |
| `npm run build` (frontend) | success, exit 0 |
| Vercel `/backtesting` | HTTP 200; synthetic board renders |
| Vercel `/` (Live) | HTTP 200; LIVE LOCKED; API 404 without backend |
| Local `:8000` at audit | connection refused |
| SQLite | `market.sqlite`, `paper.sqlite`, `scans.sqlite` present |

Screenshots: `reports/assets/backtesting-vercel.png`, `reports/assets/live-analysis-vercel.png`.

---

## 3. Subsystem audit

### 3.1 FastAPI (v0.9.0)

**Status:** IMPLEMENTED AND VERIFIED (plumbing); backtest response shaping IMPLEMENTED BUT NOT FULLY VERIFIED.

- Entry: `api/main.py` — CORS, lifespan store open + scanner start.
- ~40 routes across overview, market, research, paper, explain, analytics, live, backtest.
- No authentication or rate limiting.
- `/api/diagnostics` redacts env names containing `KEY`/`TOKEN` but may expose other env values and paths.

### 3.2 Market data

**Status:** Historical store IMPLEMENTED AND VERIFIED; live feed UNAVAILABLE / not verified.

- Providers: Yahoo history; optional `BrokerLiveProvider`.
- Session gate: Mon–Fri 09:15–15:30 IST; **no holiday calendar**.
- Freshness vocabulary: live / delayed / cached / simulated / unavailable (+ historical/demo in paper layer).
- Observed DB: candles populated for RELIANCE, TCS, HDFCBANK, INFY; **quotes = 0**.

### 3.3 ML pipeline

**Status:** IMPLEMENTED BUT NOT FULLY VERIFIED.

- Default: `TECHNICAL_MODE=True`, `CORE_MODE=True` — single GB model, daily+weekly features, horizon 7, target >1% forward return.
- Leakage controls present: purged split, walk-forward embargo, calibration on pre-test slice.
- **Defect:** `merge_asof(..., direction="backward")` of Yahoo weekly bars without shifting can attach week-end close to earlier weekdays → **look-ahead bias** on default path.
- Macro/micro/fusion modules exist but are disabled under TECHNICAL_MODE; if enabled, snapshot fundamentals/news broadcast risks further leakage.

### 3.4 Signals

**Status:** IMPLEMENTED (strategy unit tests thin).

| Mode | BUY rule (summary) |
|---|---|
| technical_only | P > 0.6 and Close > MA50 |
| momentum | P + MA20>MA50 + RSI band |
| mean_reversion | z-score thresholds (classifier diagnostic) |
| multi_factor | P thresholds via `attach_signals` (macro blend **not** active under CORE_MODE) |

### 3.5 Risk & costs

**Status:** IMPLEMENTED AND VERIFIED.

- Research: vol/drawdown RiskScore, NoTrade blocks BUY not SELL, position sizing.
- Costs: India delivery/intraday assumption schedules + slippage bps.
- Paper: max order/position/exposure/daily loss; **daily-loss can also block SELL** (inconsistency).

### 3.6 Backtesting engine & API

**Status:** Engine IMPLEMENTED AND VERIFIED; API equity/window clipping IMPLEMENTED BUT NOT FULLY VERIFIED.

- Next-open fills; long/cash; gross vs net; Sharpe/Sortino/CAGR/MaxDD etc.
- API issue: `equity_curve` returned as DataFrame but route code often expects Series → empty equity_curve in some responses; window clip does not recompute metrics over clipped window.
- React does not use these routes.

### 3.7 Paper trading

**Status:** IMPLEMENTED AND VERIFIED.

- Accounts isolated (`paper:` vs `demo:`).
- No broker order submission path.
- Tests in `tests/test_paper_trading.py`.

### 3.8 AI explainability

**Status:** IMPLEMENTED AND VERIFIED.

- Deterministic base + optional OpenAI-compatible LLM.
- Schema enforces `signal_lock` and matching `decision.signal`.
- Cannot override signals/risk/orders.
- LLM free-text may still invent numbers in prose (prompt-only control).

### 3.9 Live scanner

**Status:** PARTIALLY IMPLEMENTED.

- Background thread when API runs; interval `LIVE_SCAN_INTERVAL_S` (default 300s).
- Scores Yahoo daily via `run_four_model_pipeline`.
- LIVE freshness unreachable without verified broker live status.
- `scans.sqlite` had 0 runs at audit time.

### 3.10 Frontend product

**Status:** Two-tab UI IMPLEMENTED AND VERIFIED; three-tab claim NOT IMPLEMENTED; Backtesting truthfulness PARTIALLY IMPLEMENTED / misleading.

- Backtesting: seeded LCG random walks; combined ensemble averages sleeves with synergy boost; metrics partially heuristic (`cagr_pct = net * 0.85`).
- After commit `fe1619a`, DEMO badges were removed; remaining cues are SIMULATED / DATE-WINDOW SIMULATION.

---

## 4. Critical technical audit (severity register)

| ID | Severity | Issue | Impact | Recommendation |
|---|---|---|---|---|
| A1 | **Critical** | React Backtesting uses synthetic `showcaseDemo` while presenting as simulation with profit | Academic/demo misrepresentation; not research evidence | Reconnect UI to `/api/backtest/simulate-strategies` **or** restore explicit DEMO labelling |
| A2 | **Critical** | Weekly feature look-ahead vs Yahoo 1wk stamping | Inflated OOS accuracy/backtest metrics | Shift weekly features by ≥1 week or stamp bars by period end and align causally; add regression test |
| A3 | **High** | Vercel Live without working API (`VITE_API_BASE_URL`) | Live tab non-functional in production | Deploy API + set CORS + build-time env; document verified URL |
| A4 | **High** | Live NSE feed never verified; LIVE mode unreachable | Cannot claim live trading signals | Keep LIVE LOCKED until provider verification checklist passes |
| A5 | **High** | No API authentication on mutating routes | Abuse if public | Add auth, IP allowlist, or private network; protect reset/scan/collect |
| A6 | **High** | Docs claim multi-page terminal + “no frontend mock market data” | False confidence for evaluators | Update README/API/ARCHITECTURE/DEMO guides to match two-tab + showcase reality |
| A7 | **Medium** | Backtest API equity_curve / window metric clipping bugs | Incomplete or misleading API analytics | Fix Series vs DataFrame handling; recompute metrics on clipped window; add tests |
| A8 | **Medium** | Paper risk daily-loss can block SELL | Prevents exits under stress | Align with research: always allow risk-reducing exits |
| A9 | **Medium** | SQLite shared connection, single-worker assumption | Corruption/race under multi-worker | Keep workers=1 or migrate to Postgres + locking |
| A10 | **Medium** | In-process research/explain caches | Lost on restart; not multi-instance safe | Persist or redis/shared store |
| A11 | **Medium** | Macro/micro disabled but “Multi-Factor” naming remains | Label mismatch | Rename presets or enable honest technical-only naming |
| A12 | **Low** | CI omits frontend build | Broken UI can merge unnoticed | Add `npm ci && npm run build` job |
| A13 | **Low** | Thin stubs: `test_feature_engineering.py`, `test_data_loader.py`, `test_pipeline_smoke.py` | Coverage illusion | Expand or remove stubs |
| A14 | **Low** | No NSE holiday calendar | Session status wrong on holidays | Integrate holiday list |

---

## 5. Quantitative methodology notes (as implemented)

### 5.1 Probability semantics

Model output `Proba` is **calibrated (when possible) estimate of P(Target=1)** where Target is “forward return over 7 trading days exceeds 1%”.  
It is **not** probability of profit of a trade after costs, and not P(profitable backtest). Explainability notes this distinction (`probability_of_profit_note` in reasoning schema).

### 5.2 Core formulas (code-backed)

- Forward return: \( R_{t\to t+h} = \frac{C_{t+h}}{C_t} - 1 \), \( h=7 \).
- Target: \( Y_t = \mathbf{1}\{R_{t\to t+h} > 0.01\} \).
- Technical BUY: \( P_t > 0.6 \land C_t > MA50_t \).
- RiskScore ≈ \( 0.6\cdot\mathrm{clip}(\sigma/\sigma_{\max}) + 0.4\cdot\mathrm{clip}(-DD/DD_{\mathrm{hard}}) \).
- Position size: \( \mathrm{clip}(s_0(1-\mathrm{RiskScore}), s_{\min}, s_0) \).
- Fill price: next open with slippage; costs per `IndiaEquityCostConfig`.

### 5.3 Walk-forward

Expanding folds, embargo of `horizon_bars` between train and test; calibration fitted only on pre-test slice; metrics include accuracy, Brier, ECE, AUC where applicable.

---

## 6. Security & deployment

| Control | Status |
|---|---|
| Secrets in env / not committed | Verified via `.env.example` pattern |
| Broker execution disabled | Hard-coded false |
| Production API | **Not verified** |
| Frontend deploy | Verified Vercel URL responds |
| AuthN/AuthZ | Missing |
| CORS | Configurable; must include Vercel origin when API public |

---

## 7. Git milestones (selected)

| Commit | Significance |
|---|---|
| `f912c43` | Initial end-to-end multi-factor app |
| `7de5084` | Reliability / testing / research modules |
| `4f369a3`–`baed4e7` | Market SQLite + live adapter scaffolding |
| `80df759` | Isolated paper/demo trading |
| `adbb634`–`437c9db` | FastAPI + React terminal |
| `5c85cd4`–`982552e` | Explainability + analytics |
| `f797498` | Three-tab restructure |
| `97e8789` / `69287bb` | Collapse then restore Live + Backtesting |
| `ea9fe88` / `fe1619a` | Client showcase demo; remove DEMO headlines |

---

## 8. Recommended next steps (priority order)

1. Either **wire Backtesting to real `/api/backtest/simulate-strategies`** or **restore unmistakable DEMO labelling**.
2. Fix weekly feature causality; add a failing test with Monday-stamped Yahoo-like bars.
3. Deploy FastAPI with disk + set `VITE_API_BASE_URL` and `API_CORS_ORIGINS`; re-verify Live end-to-end.
4. Update product docs to match the two-tab UI.
5. Add auth for mutating endpoints before any public API exposure.
6. Expand tests for strategy attach_* and backtest equity serialization.

---

## 9. Conclusion

The repository contains a credible academic research codebase for Indian equity signal research with strong safety defaults (no live orders, paper isolation, explainability lock). The **largest integrity gaps** for evaluators are: (a) synthetic Backtesting UI without clear DEMO labelling, (b) weekly look-ahead risk in the default ML path, and (c) production Live UI disconnected from a backend. Treat profitability screenshots from the current Backtesting tab as **non-empirical**.

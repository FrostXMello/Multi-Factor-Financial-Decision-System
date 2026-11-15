# API and web terminal (Milestone 5)

The FastAPI service exposes the existing research, market-data, and paper-trading modules. It does **not** place real broker orders. The React app consumes these endpoints with real responses (no frontend mock market data).

## Run the backend

```bash
cd "/Users/gagan/Documents/Projects/Multi-factor Financial Decision System"
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn api.main:app --reload --port 8000
```

Open docs at http://127.0.0.1:8000/docs

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000`. For Vercel, set `VITE_API_BASE_URL` to the independently deployed API origin (see `frontend/.env.example`).

## Key endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness; `broker_execution: false` |
| GET | `/api/health/ready` | Readiness; SQLite stores open/migrated |
| GET | `/api/overview` | Session, provider, paper snapshot |
| GET | `/api/market/status` | Providers, data mode, DB stats |
| GET | `/api/market/quotes` | Stored watchlist quotes |
| GET | `/api/market/candles/{symbol}` | Stored OHLCV |
| POST | `/api/market/collect` | One-shot watchlist collect |
| POST | `/api/research/run` | Existing technical pipeline |
| GET | `/api/research/last` | Cached research summary |
| GET | `/api/research/reliability` | Walk-forward payload |
| GET | `/api/paper/account?mode=paper\|demo` | Portfolio snapshot |
| POST | `/api/paper/orders` | Paper/demo order |
| POST | `/api/paper/demo/run` | Deterministic demo workflow |
| POST | `/api/paper/reset` | Reset paper or demo account |
| GET | `/api/diagnostics` | Env checklist (secrets hidden) |
| GET | `/api/explain/status` | AI provider configured? (no secrets) |
| POST | `/api/explain/decision` | Explain latest research decision |
| GET | `/api/explain/last` | Cached explanation |
| POST | `/api/explain/risk-block` | Explain a risk rejection |
| GET | `/api/analytics/status` | Analytics module status |
| POST | `/api/analytics/regimes` | Causal bull/bear/sideways labels from stored candles |
| POST | `/api/analytics/portfolio` | Allocation, sector exposure, concentration |
| POST | `/api/analytics/compare` | Compare strategies on net returns after costs |
| POST | `/api/analytics/robustness` | Walk-forward fold consistency (diagnostic) |
| GET | `/api/analytics/report` | Combined research analytics report |

Demo responses set `demo_labelled: true` and use account id `demo:academic`.

## AI explainability (M6)

`reasoning/` builds structured JSON explanations from model/risk fields only.

- Deterministic fallback always works without an API key.
- Optional LLM: set `AI_REASONING_BASE_URL` + `AI_REASONING_API_KEY` (+ optional model/path).
- The LLM may rewrite narrative text; locked `decision.signal` / risk outcomes cannot change.
- Provider failures and malformed payloads fall back to the deterministic explanation.
- API keys are never returned to the frontend.

Limitations: sector context is included only when the caller supplies it; missing indicators stay `unavailable` and are never invented. Model probability is explicitly not treated as probability of profit.

## Strategy analytics (M7)

`strategy_research/` adds regime classification, regime-conditioned performance, portfolio analytics and walk-forward robustness on top of the existing backtest engine.

- Regimes use trailing returns only; return attribution uses `regime_for_return` (prior end-of-day label) to avoid same-bar lookahead.
- Strategy comparison ranks by **net** return after the existing transaction-cost model.
- Robustness summarises existing walk-forward folds and does **not** retune parameters on those folds.
- Insufficient samples / missing candles are labelled `insufficient_data`.
- Previously reviewed scrape/order repos were not integrated; analytics stay on our stored data + backtests.

## Deploy notes

- **Backend**: deploy `api.main:app` with uvicorn/gunicorn independently. Configure `API_CORS_ORIGINS`, `MARKET_DATA_DB`, `PAPER_TRADING_DB`, and optional broker quote credentials.
- **Frontend**: static Vite build; `frontend/vercel.json` enables SPA rewrites. Point `VITE_API_BASE_URL` at the API.
- Streamlit (`app.py`) remains available for the academic demo path.

## Tests

```bash
.venv/bin/python -m pytest -q tests/test_api.py
cd frontend && npm run build
```

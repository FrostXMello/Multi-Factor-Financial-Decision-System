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

Demo responses set `demo_labelled: true` and use account id `demo:academic`.

## Deploy notes

- **Backend**: deploy `api.main:app` with uvicorn/gunicorn independently. Configure `API_CORS_ORIGINS`, `MARKET_DATA_DB`, `PAPER_TRADING_DB`, and optional broker quote credentials.
- **Frontend**: static Vite build; `frontend/vercel.json` enables SPA rewrites. Point `VITE_API_BASE_URL` at the API.
- Streamlit (`app.py`) remains available for the academic demo path.

## Tests

```bash
.venv/bin/python -m pytest -q tests/test_api.py
cd frontend && npm run build
```

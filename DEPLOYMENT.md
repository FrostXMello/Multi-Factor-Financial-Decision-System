# Deployment guide (M8)

This project is prepared for a split deployment:

- **Frontend**: React + Vite on **Vercel**
- **Backend**: FastAPI on **Render** or **Railway** with a **persistent disk** for SQLite

Hosting credentials were not used in this milestone. Nothing below claims a live production deploy has been completed.

## Safety invariants

- `broker_execution` is always `false` in health/overview/root responses.
- Paper trading never calls a live broker order API.
- Demo account `demo:academic` never writes into `paper:default`.
- AI and broker secrets stay on the backend only. The browser only needs `VITE_API_BASE_URL`.
- Yahoo research data is **delayed**. Live NSE-session quote ingestion remains **pending verification**.

## 1. Backend (Render)

1. Create a Render Web Service from this repo (Docker runtime; `Dockerfile` at repo root).
2. Attach a persistent disk mounted at `/var/data` (see `render.yaml`).
3. Set environment variables in the Render dashboard:

| Variable | Required | Notes |
|---|---|---|
| `APP_ENV` | recommended | `production` |
| `DATA_DIR` | recommended | `/var/data` |
| `MARKET_DATA_DB` | recommended | `/var/data/market.sqlite` |
| `PAPER_TRADING_DB` | recommended | `/var/data/paper.sqlite` |
| `API_CORS_ORIGINS` | **required** | Your Vercel origin(s), comma-separated |
| `MARKET_DATA_WATCHLIST` | optional | e.g. `RELIANCE.NS,TCS.NS` |
| `BROKER_LIVE_*` | optional | Leave empty until a licensed feed is verified |
| `AI_REASONING_*` | optional | Leave empty for deterministic explanations |

4. Health check path: `/api/health/ready`
5. Confirm:
   - `GET https://<api-host>/api/health` → `broker_execution: false`
   - `GET https://<api-host>/api/health/ready` → `status: ready`
6. Use **one** uvicorn worker (Dockerfile/`Procfile` already do this) because SQLite is the default store.

### Railway alternative

1. New project from this repo; builder uses `Dockerfile` (`railway.toml`).
2. Attach a volume at `/var/data`.
3. Set the same env vars as above.
4. Health check: `/api/health/ready`

## 2. Frontend (Vercel)

1. Import the repo in Vercel; set **Root Directory** to `frontend`.
2. Framework preset: Vite (`frontend/vercel.json`).
3. Build command: `npm run build`; output: `dist`.
4. Set **one** public env var:

| Variable | Value |
|---|---|
| `VITE_API_BASE_URL` | `https://<your-api-host>` (no trailing slash) |

5. Deploy.
6. Add the Vercel URL to backend `API_CORS_ORIGINS`, then redeploy the API if needed.
7. Open the site → **Settings** → confirm API health is `ok` and broker execution is disabled.

## 3. Local production-like check

```bash
# API
DATA_DIR=./data APP_ENV=production \
  API_CORS_ORIGINS=http://localhost:5173 \
  .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1

# Frontend
cd frontend
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Or keep `VITE_API_BASE_URL` empty and use the Vite `/api` proxy for local same-origin style calls.

## 4. Verification checklist (post-deploy)

- [ ] `/api/health` and `/api/health/ready` succeed
- [ ] Overview loads from the API (not mocked)
- [ ] Market Data shows delayed Yahoo / unconfigured broker / closed-session messaging when appropriate
- [ ] Paper Trading demo workflow fills `demo:academic` only
- [ ] Settings shows no raw API keys
- [ ] CORS allows the Vercel origin
- [ ] SQLite files survive an API restart (persistent disk)

## 5. Remaining actions if hosting access is unavailable

1. Create Render/Railway account and attach a persistent disk at `/var/data`.
2. Deploy the API container; set `API_CORS_ORIGINS` after you know the Vercel URL (or update in a second pass).
3. Create the Vercel project with root `frontend` and `VITE_API_BASE_URL`.
4. Run the checklist above on the live URLs.
5. Only then state that deployment succeeded.

## Offline automated smoke

```bash
.venv/bin/python -m pytest -q tests/test_deployment_smoke.py tests/test_api.py
cd frontend && npm run build
```

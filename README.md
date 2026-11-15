# Multi-Factor Financial Decision System

**PBL-3 academic project** — a research and paper-trading terminal for Indian equities.

The system combines leakage-aware labeling, purged walk-forward evaluation, persistent market-data storage, isolated paper trading, optional AI explainability, and regime/portfolio analytics. It is **not** a live brokerage product and **not** financial advice.

## What you can demonstrate today

1. Technical research → BUY / SELL / HOLD with next-open fills and Indian cost assumptions  
2. Walk-forward reliability (accuracy ≠ profit)  
3. Market Data with honest delayed / unconfigured / closed-session labels  
4. Demo paper trading while the cash market is closed  
5. AI explanations that cannot override signals or risk  
6. Regime and portfolio analytics on stored history  

**Live NSE quote ingestion remains pending verification** (M3 offline work is complete).

## Quick start (local)

Python 3.9+ (3.11/3.12 recommended for deployment images).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# API
.venv/bin/uvicorn api.main:app --reload --port 8000

# React terminal
cd frontend && npm install && npm run dev

# Optional Streamlit academic UI
.venv/bin/python -m streamlit run app.py
```

Copy `.env.example` → `.env` for local overrides. Never commit `.env`.

| Surface | URL |
|---|---|
| API docs | http://127.0.0.1:8000/docs |
| React app | http://127.0.0.1:5173 |
| Streamlit | http://localhost:8501 |

## Documentation map

| Doc | Purpose |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System diagram, milestone map, freshness contract |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Vercel + Render/Railway steps and remaining actions |
| [`API.md`](API.md) | HTTP endpoints |
| [`DEMO_GUIDE.md`](DEMO_GUIDE.md) | 5–7 minute closed-market walkthrough |
| [`PROJECT_DEMO.md`](PROJECT_DEMO.md) | Concise PBL presentation script |

## Milestone status

| ID | Focus | Status |
|---|---|---|
| M1 | Labels, next-open execution, costs, risk exits | Complete |
| M2 | Purged walk-forward, calibration, reliability UI | Complete |
| M3 | Market-data layer + SQLite | Offline complete; **live feed pending** |
| M4 | Isolated paper / demo trading | Complete |
| M5 | FastAPI + React terminal | Complete |
| M6 | AI explainability | Complete |
| M7 | Regime + portfolio analytics | Complete |
| M8 | Deployment readiness | Prepared (see `DEPLOYMENT.md`) |

## Safety

- No live broker order execution path  
- Paper `paper:default` and demo `demo:academic` are isolated  
- Delayed / historical / demo data are never labelled live  
- AI keys and broker secrets stay server-side  

## Tests

```bash
.venv/bin/python -m pytest -q tests
cd frontend && npm run build
```

## TECHNICAL_MODE

Default `core_config.TECHNICAL_MODE = True`: daily(+weekly) price features, one GB model, purged walk-forward, MA50 + probability signals. Set both `TECHNICAL_MODE` and `CORE_MODE` to `False` only when opening the full research stack.

## Deployment

Frontend → Vercel (`frontend/`, set `VITE_API_BASE_URL`).  
Backend → Render/Railway (`Dockerfile`, persistent disk at `/var/data`).  

Exact steps and the checklist of actions still required before claiming a live deploy: [`DEPLOYMENT.md`](DEPLOYMENT.md).

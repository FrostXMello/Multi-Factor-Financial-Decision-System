# PBL-3 demonstration script (≈7 minutes)

Use this for an academic presentation. Prefer the **React terminal + API** path; Streamlit remains a backup.

## Before you start

```bash
.venv/bin/uvicorn api.main:app --port 8000
cd frontend && npm run dev
```

Open http://127.0.0.1:5173. No broker credentials required. The cash session may be closed.

## Script

1. **Overview (45s)** — Research terminal, not a broker. Show session status and paper snapshot. State: no live orders.  
2. **Research (90s)** — Run RELIANCE.NS (or cached last run). Show BUY/SELL/HOLD and net vs gross after costs.  
3. **Reliability (60s)** — Walk-forward OOS vs in-sample; accuracy is not profit.  
4. **Market Data (60s)** — Yahoo delayed, broker unconfigured, closed-session collect message. Live feed still pending.  
5. **Strategy Analytics (45s)** — Regimes + portfolio HHI / sector map; insufficient data labelled.  
6. **AI Explainability (45s)** — Explain latest decision; signal_lock; confidence ≠ P(profit).  
7. **Paper Trading Demo (90s)** — Demo / Replay → deterministic workflow → positions + order history on `demo:academic` only.  

## Closing line

“This is a research and paper-trading system with honest data labels. Live NSE quote verification and cloud hosting cutover are the remaining operational steps documented in DEPLOYMENT.md.”

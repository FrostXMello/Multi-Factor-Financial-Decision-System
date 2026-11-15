# Demo walkthrough (5–7 minutes)

Works **with the NSE cash session closed** and **without broker credentials**.  
Do not present delayed Yahoo history or demo fixtures as live market data.

## Option A — React terminal + API (preferred)

```bash
cd "/Users/gagan/Documents/Projects/Multi-factor Financial Decision System"
.venv/bin/uvicorn api.main:app --port 8000
cd frontend && npm run dev
```

Open http://127.0.0.1:5173 and follow [`PROJECT_DEMO.md`](PROJECT_DEMO.md).

## Option B — Streamlit

```bash
.venv/bin/python -m streamlit run app.py
```

Open the local URL (usually http://localhost:8501).

### 1. Introduce the system (45s)

Open **Overview**.

- Leakage-aware research, walk-forward evaluation, SQLite market data, isolated paper trading.
- Disclaimer: research/paper only — not financial advice, not live execution.
- Closed session outside 09:15–15:30 IST is expected.

### 2. Research output (90s)

**Research & Predictions** → Single Ticker → **Run CORE pipeline**.

- Latest signal and probability.
- Next-session open fills; risk can block new BUYs while exits remain allowed.

### 3. Walk-forward reliability (60s)

**Backtesting / Reliability** (or the reliability panel after a run).

- OOS vs in-sample accuracy; accuracy ≠ net profit after costs.

### 4. Market Data while closed (60s)

**Market Data**

- Yahoo delayed; broker UNCONFIGURED without credentials.
- **Collect watchlist once** → no quote requested when the session is closed.
- State: M3 offline complete; live NSE verification pending.

### 5–7. Demo paper trading (2 min)

**Paper Trading** → **Demo / Replay** → **Run deterministic demo workflow**.

- Blue DEMO badge; account `demo:academic`.
- Positions, fees/slippage-aware fills, order history.
- Show a rejection (oversell / unconfirmed) for risk controls.
- Settings / Diagnostics for DB paths; paper book stays untouched.

## Closing line

“Historical research data and a labelled demo ledger only. Live broker quote verification is still pending. No real orders were sent.”

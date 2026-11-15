# Architecture (PBL-3)

Multi-Factor Financial Decision System — academic research terminal for Indian equities.

```
┌─────────────────────────────────────────────────────────────┐
│ React terminal (Vercel)                                     │
│ Overview · Market · Research · Reliability · Analytics      │
│ Explainability · Paper Trading · Settings                   │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTPS JSON (/api/*)
┌───────────────────────────▼─────────────────────────────────┐
│ FastAPI (Render / Railway)                                  │
│ routes: overview, market, research, paper, explain, analytics│
└───────┬─────────────┬──────────────┬─────────────┬──────────┘
        │             │              │             │
   market_data/  paper_trading/  reasoning/  strategy_research/
        │             │
   SQLite market  SQLite paper
   (persistent disk)
```

## Milestone map

| Milestone | Responsibility |
|---|---|
| M1 | Leakage-aware labels, next-open fills, Indian costs, risk exits |
| M2 | Purged walk-forward, calibration, held-out explainability |
| M3 | Provider-independent market data + SQLite persistence |
| M4 | Isolated paper / demo trading (no live orders) |
| M5 | FastAPI + React terminal |
| M6 | AI explainability with deterministic fallback |
| M7 | Regime analytics + portfolio analytics |
| M8 | Deployment readiness (this document + `DEPLOYMENT.md`) |

## Data freshness contract

| Label | Meaning |
|---|---|
| `live` | Authorized live broker freshness only |
| `delayed` | Yahoo / delayed research |
| `historical` | Stored candle mark |
| `demo` | Deterministic demo fixture |
| `unavailable` | No usable mark |

Never upgrade delayed/demo/historical to live. Market collection does not invent quotes when the NSE cash session is closed or credentials are missing.

**M3 status:** offline implementation verified. Live NSE-session quote ingestion remains pending verification.

## Safety boundaries

- Research signals → risk gates → paper simulator (optional, confirmed).
- LLM explainability cannot change signals, risk, or orders.
- No broker order-placement libraries are wired into runtime paths.
- Streamlit `app.py` remains available as a single-process academic UI; the React app is the web terminal.

## Persistence

- `MARKET_DATA_DB` — instruments, OHLCV, quotes, quality/provider events
- `PAPER_TRADING_DB` — paper + demo accounts, ledger, orders, positions
- Production default root: `DATA_DIR` (e.g. `/var/data` on a mounted disk)

## External repositories (evaluated, not integrated)

`fenix`, `nsepython`, `Bharat-SM-Data`, `india-stocks-api` were reviewed for M3/M7 and skipped (licensing, scraping, order routing, or Python/runtime mismatch). The production design prefers official/authorized feeds and our own adapters.

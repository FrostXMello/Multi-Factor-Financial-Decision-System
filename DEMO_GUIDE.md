# Demo walkthrough (5–7 minutes)

This script works **with the NSE cash session closed** and **without broker credentials**.  
Do not present delayed Yahoo history or demo fixtures as live market data.

## Start

```bash
cd "/Users/gagan/Documents/Projects/Multi-factor Financial Decision System"
.venv/bin/python -m streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## 1. Introduce the system (45s)

Open **Overview** in the sidebar.

- Explain the stack: leakage-aware research labels, purged walk-forward evaluation, persistent SQLite market data, isolated paper trading.
- Point at the disclaimer: research and paper trading only — not financial advice, not live execution.
- Note the session badge: closed outside 09:15–15:30 IST is correct today.

## 2. Research output and BUY / SELL / HOLD (90s)

Open **Research & Predictions**.

- Universe: **Single Ticker** (default RELIANCE.NS is fine).
- Click **Run CORE pipeline**.
- Show the latest signal metric and probability.
- Explain: the model emits BUY / SELL / HOLD; fills in the backtest use the next session open; risk can block new BUYs while still allowing exits.

## 3. Walk-forward reliability (60s)

Still on the research result page, open the **Model reliability** area (or navigate to **Backtesting / Reliability** after the run).

- Show out-of-sample vs in-sample accuracy.
- Mention time-aware calibration and held-out permutation importance.
- Stress that accuracy is not the same as net profit after costs.

## 4. Market Data while the market is closed (60s)

Open **Market Data**.

- Providers: Yahoo = delayed historical research; broker = UNCONFIGURED without credentials.
- Session = CLOSED; quote indicator = none stored (unless you already ran research).
- Click **Collect watchlist once** and show the message that no quote was requested because the session is closed.
- State clearly: *M3 offline work is done; live NSE-session quote ingestion remains pending.*

## 5. Enter Demo / Replay (30s)

Open **Paper Trading**.

- Switch **Account mode** to **Demo / Replay**.
- Confirm the blue DEMO badge and the `demo:academic` account id.
- Emphasize this ledger never writes into `paper:default`.

## 6. Deterministic paper order and portfolio (90s)

- Click **Run deterministic demo workflow**, or place a manual BUY on RELIANCE.NS with confirmation checked.
- Show account KPIs: cash down, invested up, portfolio value.
- Open **Positions**: quantity, average entry, demo mark, source `demo_fixture`, freshness `demo`.
- Open **Order history**: filled rows with fees, slippage-aware fill price, and freshness.

## 7. Risk controls and persistence (45s)

- Try a SELL larger than the holding (or uncheck confirmation) and show the rejection reason.
- Click **Reset this account**, then switch away and back (or restart Streamlit) to show the paper/demo SQLite file under Settings / Diagnostics.
- Optional: Settings page — market DB path, paper DB path, env checklist, verification banner.

## Closing line

“This demonstration used historical research data and a labelled demo ledger. Live broker quote verification is still pending for the next NSE session. No real orders were sent.”

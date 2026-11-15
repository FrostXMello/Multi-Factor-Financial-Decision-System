import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { api } from './api'
import { Badge, Metric, Panel, Table, money, pct } from './components'

function useLoad<T>(loader: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const reload = async () => {
    setLoading(true)
    setError(null)
    try {
      setData(await loader())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => {
    void reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return { data, error, loading, reload }
}

export function OverviewPage() {
  const { data, error, loading, reload } = useLoad(() => api.overview())
  if (loading) return <p className="muted">Loading overview…</p>
  if (error) return <div className="error">{error}</div>
  if (!data) return null
  const paper = (data.paper_account ?? {}) as Record<string, unknown>
  return (
    <>
      <div className="banner">
        <h2>Multi-Factor Financial Decision System</h2>
        <p>Research terminal for Indian equities — walk-forward evaluation, market data, and isolated paper trading.</p>
      </div>
      <div className="grid">
        <Metric label="NSE cash session" value={String(data.nse_cash_session ?? '—').toUpperCase()} />
        <Metric label="Yahoo research" value={<Badge label={String(data.yahoo_freshness ?? 'delayed')} />} />
        <Metric label="Broker feed" value={<Badge label={data.broker_configured ? String(data.broker_freshness) : 'UNCONFIGURED'} />} />
        <Metric label="Stored candles" value={String((data.market_database as Record<string, unknown>)?.candles ?? 0)} />
      </div>
      <Panel title="Paper account summary">
        <div className="grid">
          <Metric label="Cash" value={money(paper.cash)} />
          <Metric label="Equity" value={money(paper.equity)} />
          <Metric label="Positions" value={String(paper.positions ?? 0)} />
          <Metric label="Return" value={pct(paper.total_return)} />
        </div>
      </Panel>
      <p className="muted">{String(data.live_verification ?? '')}</p>
      <button className="secondary" onClick={() => void reload()}>Refresh</button>
      <p className="disclaimer">{String(data.disclaimer ?? '')}</p>
    </>
  )
}

export function MarketPage() {
  const status = useLoad(() => api.marketStatus())
  const quotes = useLoad(() => api.marketQuotes())
  const coverage = useLoad(() => api.marketCoverage())
  const quality = useLoad(() => api.marketQuality())
  const [symbol, setSymbol] = useState('RELIANCE.NS')
  const candles = useLoad(() => api.marketCandles(symbol), [symbol])
  const [message, setMessage] = useState<string | null>(null)

  const collect = async () => {
    setMessage(null)
    try {
      const result = await api.marketCollect()
      setMessage(String(result.detail ?? 'Collect finished'))
      await Promise.all([status.reload(), quotes.reload(), quality.reload()])
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <>
      <div className="banner">
        <h2>Market Data</h2>
        <p>Provider status and stored history. Delayed or historical rows are never labelled live.</p>
      </div>
      {status.error && <div className="error">{status.error}</div>}
      {status.data && (
        <>
          <p>
            <Badge label={String(status.data.data_mode ?? 'UNCONFIGURED')} />{' '}
            <span className="muted">checked {String(status.data.checked_at ?? '')}</span>
          </p>
          <div className="grid">
            <Metric label="NSE session" value={String(status.data.nse_cash_session ?? '').toUpperCase()} />
            <Metric label="Candles" value={String((status.data.database as Record<string, unknown>)?.candles ?? 0)} />
            <Metric label="Quotes" value={String((status.data.database as Record<string, unknown>)?.quotes ?? 0)} />
            <Metric label="Schema" value={String((status.data.database as Record<string, unknown>)?.schema_version ?? '—')} />
          </div>
          <Panel title="Providers">
            <Table rows={(status.data.providers as Record<string, unknown>[]) ?? []} />
          </Panel>
          <p className="muted">{String(status.data.note ?? '')}</p>
        </>
      )}
      <Panel title="Watchlist quotes">
        {quotes.error ? <div className="error">{quotes.error}</div> : <Table rows={(quotes.data?.quotes as Record<string, unknown>[]) ?? []} />}
      </Panel>
      <Panel title="Historical availability">
        <Table rows={(coverage.data?.coverage as Record<string, unknown>[]) ?? []} />
      </Panel>
      <Panel title="Latest stored candles">
        <div className="row">
          <label>
            Symbol
            <input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
          </label>
        </div>
        <p className="muted">{String(candles.data?.detail ?? '')}</p>
        <Table rows={(candles.data?.candles as Record<string, unknown>[]) ?? []} />
      </Panel>
      <Panel title="Quality / provider messages">
        <Table rows={(quality.data?.quality_events as Record<string, unknown>[]) ?? []} />
        <Table rows={(quality.data?.provider_events as Record<string, unknown>[]) ?? []} />
      </Panel>
      <button onClick={() => void collect()}>Collect watchlist once</button>
      {message && <p className="ok" style={{ marginTop: '0.75rem' }}>{message}</p>}
    </>
  )
}

export function ResearchPage() {
  const last = useLoad(() => api.researchLast())
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const summary = (last.data?.summary ?? null) as Record<string, unknown> | null

  const run = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await api.researchRun(ticker)
      await last.reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <div className="banner">
        <h2>Predictions &amp; Research</h2>
        <p>Runs the existing technical research pipeline. Results use delayed Yahoo history unless a fresher store mark exists.</p>
      </div>
      <form className="row" onSubmit={run}>
        <label>
          Ticker
          <input value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} />
        </label>
        <button type="submit" disabled={busy}>{busy ? 'Running…' : 'Run research'}</button>
      </form>
      {error && <div className="error">{error}</div>}
      {!summary && <p className="muted">No research run cached in the API process yet.</p>}
      {summary && (
        <>
          <div className="grid">
            <Metric label="Ticker" value={String(summary.ticker ?? '—')} />
            <Metric label="Signal" value={String((summary.latest as Record<string, unknown>)?.signal ?? '—')} />
            <Metric label="Probability" value={String((summary.latest as Record<string, unknown>)?.proba ?? '—')} />
            <Metric label="OOS accuracy" value={String((summary.metrics as Record<string, unknown>)?.walk_forward_mean_accuracy ?? '—')} />
          </div>
          <Panel title="Backtest KPIs">
            <div className="grid">
              <Metric label="Net return %" value={String((summary.metrics as Record<string, unknown>)?.net_return_pct ?? '—')} />
              <Metric label="Final value" value={money((summary.metrics as Record<string, unknown>)?.final_portfolio_value)} />
              <Metric label="Costs" value={money((summary.metrics as Record<string, unknown>)?.total_transaction_costs)} />
              <Metric label="Trades" value={String((summary.metrics as Record<string, unknown>)?.trades ?? '—')} />
            </div>
          </Panel>
          <Panel title="Signals (tail)">
            <Table rows={(summary.signals_tail as Record<string, unknown>[]) ?? []} />
          </Panel>
          <Panel title="Explainability / reliability snapshot">
            <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.82rem' }}>{JSON.stringify(summary.reliability ?? {}, null, 2)}</pre>
          </Panel>
          <p className="disclaimer">{String(summary.disclaimer ?? '')}</p>
        </>
      )}
    </>
  )
}

export function ReliabilityPage() {
  const { data, error, loading, reload } = useLoad(() => api.researchReliability())
  if (loading) return <p className="muted">Loading reliability…</p>
  if (error) return <div className="error">{error}</div>
  if (!data?.available) return <p className="muted">{String(data?.detail ?? 'Run research first.')}</p>
  return (
    <>
      <div className="banner">
        <h2>Backtesting &amp; Model Reliability</h2>
        <p>Walk-forward diagnostics from the last research run in this API process.</p>
      </div>
      <Panel title={`Reliability · ${String(data.ticker ?? '')}`}>
        <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.82rem' }}>{JSON.stringify(data.reliability ?? {}, null, 2)}</pre>
      </Panel>
      <Panel title="Metrics">
        <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.82rem' }}>{JSON.stringify(data.metrics ?? {}, null, 2)}</pre>
      </Panel>
      <button className="secondary" onClick={() => void reload()}>Refresh</button>
      <p className="disclaimer">{String(data.disclaimer ?? '')}</p>
    </>
  )
}

export function PaperPage() {
  const [mode, setMode] = useState<'paper' | 'demo'>('paper')
  const account = useLoad(() => api.paperAccount(mode), [mode])
  const [symbol, setSymbol] = useState('RELIANCE.NS')
  const [side, setSide] = useState('BUY')
  const [qty, setQty] = useState(1)
  const [confirm, setConfirm] = useState(false)
  const [allowHistorical, setAllowHistorical] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const snap = (account.data?.snapshot ?? {}) as Record<string, unknown>

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setError(null)
    setMessage(null)
    try {
      const result = await api.paperOrder({
        symbol,
        side,
        quantity: qty,
        mode,
        confirm,
        allow_historical: allowHistorical || mode === 'demo',
      })
      setMessage(`Filled ${String((result.order as Record<string, unknown>)?.side)} ${(result.order as Record<string, unknown>)?.symbol}`)
      await account.reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const runDemo = async () => {
    setError(null)
    setMessage(null)
    try {
      const result = await api.paperDemoRun()
      setMode('demo')
      setMessage(`Demo workflow complete on ${String(result.account_id)}`)
      await account.reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <>
      <div className="banner">
        <h2>Paper Trading</h2>
        <p>Isolated virtual ledger. Real broker execution is disabled.</p>
      </div>
      <div className="row">
        <label>
          Account mode
          <select value={mode} onChange={(e) => setMode(e.target.value as 'paper' | 'demo')}>
            <option value="paper">Paper (normal)</option>
            <option value="demo">Demo / Replay</option>
          </select>
        </label>
        {mode === 'demo' && <Badge label="DEMO" />}
        <button className="secondary" onClick={() => void api.paperReset(mode).then(() => account.reload())}>Reset account</button>
        <button onClick={() => void runDemo()}>Run deterministic demo workflow</button>
      </div>
      {account.error && <div className="error">{account.error}</div>}
      <div className="grid">
        <Metric label="Cash" value={money(snap.cash)} />
        <Metric label="Invested" value={money(snap.invested)} />
        <Metric label="Equity" value={money(snap.equity)} />
        <Metric label="Net P&L" value={money(snap.net_pnl)} />
      </div>
      <Panel title="Positions">
        <Table rows={(snap.positions as Record<string, unknown>[]) ?? []} />
      </Panel>
      <Panel title="Place paper order">
        <form className="row" onSubmit={submit}>
          <label>Symbol<input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} /></label>
          <label>Side
            <select value={side} onChange={(e) => setSide(e.target.value)}>
              <option>BUY</option>
              <option>SELL</option>
            </select>
          </label>
          <label>Quantity<input type="number" min={1} value={qty} onChange={(e) => setQty(Number(e.target.value))} /></label>
          <label><span>Confirm</span><input type="checkbox" checked={confirm} onChange={(e) => setConfirm(e.target.checked)} /></label>
          <label><span>Allow historical/delayed</span><input type="checkbox" checked={allowHistorical || mode === 'demo'} disabled={mode === 'demo'} onChange={(e) => setAllowHistorical(e.target.checked)} /></label>
          <button type="submit">Submit</button>
        </form>
      </Panel>
      <Panel title="Order history">
        <Table rows={(snap.orders as Record<string, unknown>[]) ?? []} />
      </Panel>
      <Panel title="Closed trades / analytics">
        <div className="grid">
          <Metric label="Win rate" value={snap.win_rate == null ? 'n/a' : pct(snap.win_rate)} />
          <Metric label="Profit factor" value={snap.profit_factor == null ? 'n/a' : String(snap.profit_factor)} />
          <Metric label="Drawdown" value={pct(snap.drawdown)} />
          <Metric label="Return" value={pct(snap.total_return)} />
        </div>
        <Table rows={(snap.closed_trades as Record<string, unknown>[]) ?? []} />
      </Panel>
      {message && <div className="ok">{message}</div>}
      {error && <div className="error">{error}</div>}
      <p className="disclaimer">Paper/demo only. Demo fills stay on demo:academic and never write into paper:default.</p>
    </>
  )
}

export function SettingsPage() {
  const { data, error, loading, reload } = useLoad(() => api.diagnostics())
  if (loading) return <p className="muted">Loading diagnostics…</p>
  if (error) return <div className="error">{error}</div>
  if (!data) return null
  return (
    <>
      <div className="banner">
        <h2>Settings / Diagnostics</h2>
        <p>Environment checklist and storage paths. Secrets are never displayed.</p>
      </div>
      <Panel title="Environment">
        <Table rows={(data.environment as Record<string, unknown>[]) ?? []} />
      </Panel>
      <Panel title="Stores">
        <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.82rem' }}>
          {JSON.stringify({ market_store: data.market_store, paper_store: data.paper_store }, null, 2)}
        </pre>
      </Panel>
      <p><Badge label={data.broker_execution_enabled ? 'LIVE EXECUTION' : 'NO BROKER EXECUTION'} /></p>
      <button className="secondary" onClick={() => void reload()}>Refresh</button>
    </>
  )
}

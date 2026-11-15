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

export function AnalyticsPage() {
  const status = useLoad(() => api.analyticsStatus())
  const [symbol, setSymbol] = useState('RELIANCE.NS')
  const [report, setReport] = useState<Record<string, unknown> | null>(null)
  const [regimes, setRegimes] = useState<Record<string, unknown> | null>(null)
  const [portfolio, setPortfolio] = useState<Record<string, unknown> | null>(null)
  const [robustness, setRobustness] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const refresh = async () => {
    setBusy(true)
    setError(null)
    const errors: string[] = []
    try {
      setReport(await api.analyticsReport(symbol))
    } catch (err) {
      setReport(null)
      errors.push(err instanceof Error ? err.message : String(err))
    }
    try {
      setRegimes(await api.analyticsRegimes({ symbol, lookback: 60 }))
    } catch (err) {
      setRegimes(null)
      errors.push(err instanceof Error ? err.message : String(err))
    }
    try {
      setPortfolio(await api.analyticsPortfolio({ mode: 'paper' }))
    } catch (err) {
      setPortfolio(null)
      errors.push(err instanceof Error ? err.message : String(err))
    }
    try {
      setRobustness(await api.analyticsRobustness({ use_last_research: true }))
    } catch (err) {
      setRobustness(null)
      errors.push(err instanceof Error ? err.message : String(err))
    }
    if (errors.length) setError(errors.join(' | '))
    setBusy(false)
  }

  const reportBody = (report?.report ?? null) as Record<string, unknown> | null

  return (
    <>
      <div className="banner">
        <h2>Strategy Research &amp; Portfolio Analytics</h2>
        <p>
          Regime classification, regime-conditioned performance, portfolio exposure and walk-forward robustness.
          Metrics use stored/historical data only. Insufficient samples are labelled, not fabricated.
        </p>
      </div>
      {status.data && <p className="muted">{String(status.data.note ?? '')}</p>}
      <div className="row">
        <label>
          Symbol
          <input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
        </label>
        <button onClick={() => void refresh()} disabled={busy}>
          {busy ? 'Loading…' : 'Refresh analytics'}
        </button>
      </div>
      {error && <div className="error">{error}</div>}
      {regimes && (
        <Panel title="Market regimes">
          <div className="grid">
            <Metric label="Latest regime" value={String((regimes.latest as Record<string, unknown>)?.regime_end_of_day ?? '—')} />
            <Metric label="Trailing return" value={String((regimes.latest as Record<string, unknown>)?.trailing_return ?? '—')} />
            <Metric label="Lookahead safe" value={String(regimes.lookahead_safe)} />
            <Metric label="Lookback" value={String(regimes.lookback ?? '—')} />
          </div>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.82rem' }}>{JSON.stringify(regimes.counts ?? {}, null, 2)}</pre>
        </Panel>
      )}
      {reportBody && (
        <Panel title="Regime performance / weaknesses">
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.82rem' }}>
            {JSON.stringify(
              {
                regime_performance: reportBody.regime_performance,
                weaknesses: reportBody.weaknesses,
                unavailable: reportBody.unavailable,
              },
              null,
              2,
            )}
          </pre>
        </Panel>
      )}
      {portfolio && (
        <Panel title="Portfolio allocation &amp; concentration">
          <div className="grid">
            <Metric label="NAV" value={money(portfolio.nav)} />
            <Metric label="Invested" value={money(portfolio.invested)} />
            <Metric label="HHI" value={String((portfolio.concentration as Record<string, unknown>)?.hhi ?? '—')} />
            <Metric label="Top weight" value={pct((portfolio.concentration as Record<string, unknown>)?.top_weight)} />
          </div>
          <Table rows={(portfolio.allocation as Record<string, unknown>[]) ?? []} />
          <Table rows={(portfolio.sector_exposure as Record<string, unknown>[]) ?? []} />
        </Panel>
      )}
      {robustness && (
        <Panel title="Walk-forward robustness">
          <div className="grid">
            <Metric label="Available" value={String(robustness.available)} />
            <Metric label="Mean fold" value={String(robustness.mean ?? '—')} />
            <Metric label="Std" value={String(robustness.std ?? '—')} />
            <Metric label="Min fold" value={String(robustness.min ?? '—')} />
          </div>
          <p className="muted">{String(robustness.note ?? '')}</p>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.82rem' }}>{JSON.stringify(robustness.weaknesses ?? [], null, 2)}</pre>
        </Panel>
      )}
      {reportBody?.strategy_comparison ? (
        <Panel title="Strategy comparison (net after costs)">
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.82rem' }}>
            {JSON.stringify(reportBody.strategy_comparison, null, 2)}
          </pre>
        </Panel>
      ) : null}
      <p className="disclaimer">
        Analytics do not optimize against test folds. External market-data scrapers were not integrated for this milestone.
      </p>
    </>
  )
}

export function ExplainPage() {
  const status = useLoad(() => api.explainStatus())
  const last = useLoad(() => api.explainLast())
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const explanation = (last.data?.explanation ?? null) as Record<string, unknown> | null

  const run = async () => {
    setBusy(true)
    setError(null)
    setMessage(null)
    try {
      await api.explainDecision({ use_last_research: true, use_llm: true })
      setMessage('Explanation refreshed from the latest research context.')
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
        <h2>AI Explainability</h2>
        <p>
          Explains BUY / SELL / HOLD using structured model and risk outputs. The assistant cannot change signals,
          risk decisions, or orders. Missing fields stay labelled unavailable.
        </p>
      </div>
      {status.data && (
        <Panel title="Provider status">
          <div className="grid">
            <Metric label="Configured" value={String(status.data.configured)} />
            <Metric label="Provider" value={String(status.data.provider ?? '—')} />
            <Metric label="Can override signals" value={String(status.data.can_override_signals)} />
            <Metric label="Exposes API keys" value={String(status.data.exposes_api_keys)} />
          </div>
          <p className="muted">{String(status.data.detail ?? '')}</p>
        </Panel>
      )}
      <button onClick={() => void run()} disabled={busy}>
        {busy ? 'Explaining…' : 'Explain latest research decision'}
      </button>
      {message && <div className="ok" style={{ marginTop: '0.75rem' }}>{message}</div>}
      {error && <div className="error" style={{ marginTop: '0.75rem' }}>{error}</div>}
      {!explanation && <p className="muted">No explanation cached yet. Run research, then explain.</p>}
      {explanation && (
        <>
          <div className="grid" style={{ marginTop: '1rem' }}>
            <Metric label="Ticker" value={String(explanation.ticker ?? '—')} />
            <Metric label="Decision" value={String((explanation.decision as Record<string, unknown>)?.signal ?? '—')} />
            <Metric label="Mode" value={String(explanation.mode ?? '—')} />
            <Metric label="Risk blocked" value={String((explanation.decision as Record<string, unknown>)?.risk_blocked ?? false)} />
          </div>
          <Panel title="Decision summary">
            <p>{String(explanation.summary ?? '')}</p>
          </Panel>
          <Panel title="Confidence interpretation">
            <p>{String(explanation.confidence_interpretation ?? '')}</p>
            <p className="muted">{String(explanation.probability_of_profit_note ?? '')}</p>
          </Panel>
          <Panel title="Supporting factors">
            <Table rows={(explanation.supporting_factors as Record<string, unknown>[]) ?? []} />
          </Panel>
          <Panel title="Conflicting signals">
            <Table rows={(explanation.conflicting_signals as Record<string, unknown>[]) ?? []} />
          </Panel>
          <Panel title="Risk warnings">
            {(explanation.risk_warnings as string[] | undefined)?.length ? (
              <ul>
                {(explanation.risk_warnings as string[]).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            ) : (
              <p className="muted">No risk warnings in this explanation.</p>
            )}
            {(explanation.decision as Record<string, unknown>)?.block_reason ? (
              <p className="muted">Block reason: {String((explanation.decision as Record<string, unknown>).block_reason)}</p>
            ) : null}
          </Panel>
          <Panel title="Market regime / sector">
            <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.82rem' }}>
              {JSON.stringify(
                {
                  market_regime: explanation.market_regime,
                  sector_context: explanation.sector_context,
                },
                null,
                2,
              )}
            </pre>
          </Panel>
          <Panel title="Supporting data">
            <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.82rem' }}>
              {JSON.stringify(explanation.supporting_data ?? {}, null, 2)}
            </pre>
            <p className="muted">Unavailable: {JSON.stringify(explanation.unavailable ?? [])}</p>
          </Panel>
          <p className="disclaimer">{(explanation.limitations as string[] | undefined)?.join(' ') ?? ''}</p>
        </>
      )}
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

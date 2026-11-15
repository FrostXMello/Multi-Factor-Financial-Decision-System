import { useState, type FormEvent } from 'react'
import { api, apiBase } from './api'
import { CandleChart } from './charts'
import {
  AiInsight,
  Badge,
  EnvBanner,
  ErrorState,
  Expandable,
  LoadingState,
  Metric,
  PageHeader,
  Section,
  SignalBadge,
  Table,
  money,
  pct,
} from './components'
import { useLoad } from './hooks'

export function BacktestingPage() {
  const defaults = useLoad(() => api.backtestDefaults())
  const symbols = useLoad(() => api.backtestSymbols())
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const [filter, setFilter] = useState('')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [capital, setCapital] = useState(100000)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sim, setSim] = useState<Record<string, unknown> | null>(null)
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null)
  const candles = useLoad(() => api.marketCandles(ticker, 220), [ticker])

  const universe = ((symbols.data?.symbols as string[]) ?? []).filter((s) =>
    s.toLowerCase().includes(filter.trim().toLowerCase()),
  )
  const presets = (defaults.data?.strategy_presets as Record<string, unknown>[]) ?? []
  const strategies = (sim?.strategies as Record<string, unknown>[]) ?? []
  const buy = (sim?.buy_analysis as Record<string, unknown> | null) ?? null
  const active =
    strategies.find((s) => String(s.strategy_id) === selectedStrategy) ??
    (sim?.best as Record<string, unknown> | null) ??
    null
  const activeMetrics = (active?.metrics ?? {}) as Record<string, unknown>
  const equity = (active?.equity_tail as Record<string, unknown>[]) ?? []
  const equityProxy = equity.map((row) => {
    const value = Number(row.equity)
    return { Date: row.date, Open: value, High: value, Low: value, Close: value, Volume: 0 }
  })

  const runSimulation = async (event?: FormEvent) => {
    event?.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const payload = await api.backtestSimulateStrategies({
        ticker,
        start: start || null,
        end: end || null,
        initial_capital: capital,
        cost_segment: 'equity_delivery',
        save: true,
      })
      setSim(payload)
      setSelectedStrategy(String((payload.best as Record<string, unknown> | undefined)?.strategy_id ?? ''))
      await candles.reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <PageHeader
        title="Backtesting"
        subtitle="Pick any supported stock, simulate the five curated strategies, and read buy analysis."
        badges={
          <>
            <Badge label="HISTORICAL" tone="historical" />
            <Badge label="NO LIVE ORDERS" tone="muted" />
          </>
        }
      />
      <EnvBanner
        mode="HISTORICAL RESEARCH"
        detail={String(
          defaults.data?.lookahead_note ??
            'Five fixed sleeves are ranked with a predeclared score — not optimized on the test fold.',
        )}
      />

      <Section title="1. Select stock" subtitle={String(symbols.data?.detail ?? 'NIFTY50 + stored symbols')}>
        <div className="toolbar">
          <label>
            Search
            <input value={filter} onChange={(e) => setFilter(e.target.value.toUpperCase())} placeholder="RELIANCE" />
          </label>
          <label>
            Selected
            <input value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} />
          </label>
          <label>
            Start
            <input type="date" value={start} onChange={(e) => setStart(e.target.value)} />
          </label>
          <label>
            End
            <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
          </label>
          <label>
            Capital
            <input type="number" min={1000} value={capital} onChange={(e) => setCapital(Number(e.target.value))} />
          </label>
        </div>
        <div className="stock-grid">
          {universe.slice(0, 60).map((symbol) => (
            <button
              key={symbol}
              type="button"
              className={`stock-chip${symbol === ticker ? ' active' : ''}`}
              onClick={() => setTicker(symbol)}
            >
              {symbol.replace('.NS', '')}
            </button>
          ))}
          {!universe.length && <p className="muted">No symbols matched.</p>}
        </div>
      </Section>

      <Section
        title="2. Five strategies"
        subtitle="Technical Trend · Technical Conservative · Momentum · Mean Reversion · Multi-Factor"
        actions={
          <button type="button" onClick={() => void runSimulation()} disabled={busy || !ticker}>
            {busy ? 'Simulating 5 strategies…' : `Simulate 5 strategies on ${ticker}`}
          </button>
        }
      >
        <div className="strategy-grid">
          {presets.map((preset) => (
            <div key={String(preset.id)} className="strategy-card">
              <strong>{String(preset.label)}</strong>
              <p className="muted tiny">{String(preset.blurb)}</p>
              <p className="mono tiny">
                buy≥{String(preset.buy_threshold)} · sell≤{String(preset.sell_threshold)}
              </p>
            </div>
          ))}
          {!presets.length && (
            <p className="muted">Presets load from `/api/backtest/defaults` once the API is reachable.</p>
          )}
        </div>
      </Section>

      {error && <ErrorState message={error} />}
      {busy && <LoadingState label={`Running five historical simulations on ${ticker}…`} />}

      {sim && (
        <>
          <Section title="3. Strategy ranking" subtitle={String(sim.ranking_note ?? '')}>
            <div className="candidate-list">
              {strategies.map((row) => {
                const id = String(row.strategy_id)
                const activeRow = id === String(active?.strategy_id ?? '')
                const m = (row.metrics as Record<string, unknown>) ?? {}
                return (
                  <button
                    key={id}
                    type="button"
                    className={`candidate-row${activeRow ? ' active' : ''}`}
                    onClick={() => setSelectedStrategy(id)}
                  >
                    <span>
                      <strong>
                        #{String(row.rank)} {String(row.label)}
                      </strong>
                      <span className="muted tiny"> · score {Number(row.rank_score ?? 0).toFixed(3)}</span>
                    </span>
                    <SignalBadge signal={row.signal} />
                    <span className="mono tiny">net {String(m.net_return_pct ?? '—')}%</span>
                    <span className="mono tiny">Sharpe {String(m.sharpe ?? '—')}</span>
                  </button>
                )
              })}
            </div>
            {!!(sim.errors as unknown[] | undefined)?.length && (
              <Expandable title="Strategies that failed">
                <pre className="code-block">{JSON.stringify(sim.errors, null, 2)}</pre>
              </Expandable>
            )}
          </Section>

          {buy?.available ? (
            <Section
              title="4. Why buy / why caution"
              subtitle={`${String(buy.ticker)} · best sleeve ${String(buy.best_strategy)}`}
              actions={
                <Badge
                  label={String(buy.recommendation ?? '')}
                  tone={String(buy.latest_signal) === 'BUY' ? 'buy' : 'hold'}
                />
              }
            >
              <div className="layout-split">
                <div>
                  <h4 className="subtle-title">Supporting buy case</h4>
                  <ul className="reason-list">
                    {((buy.why_buy as string[]) ?? []).map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                    {!((buy.why_buy as string[]) ?? []).length && <li className="muted">No active buy supports.</li>}
                  </ul>
                </div>
                <div>
                  <h4 className="subtle-title">Risk / invalidation</h4>
                  <ul className="reason-list caution">
                    {((buy.why_caution as string[]) ?? []).map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              </div>
              {(buy.explanation as Record<string, unknown>)?.summary ? (
                <div className="ai-insight compact" style={{ marginTop: '0.85rem' }}>
                  <span className="ai-kicker">AI explanation · does not override signal</span>
                  <p className="ai-summary">{String((buy.explanation as Record<string, unknown>).summary)}</p>
                  {(buy.explanation as Record<string, unknown>).confidence_interpretation ? (
                    <p className="muted">
                      {String((buy.explanation as Record<string, unknown>).confidence_interpretation)}
                    </p>
                  ) : null}
                </div>
              ) : null}
              <p className="muted tiny">{String(buy.note ?? '')}</p>
            </Section>
          ) : null}

          {active && (
            <Section
              title={`5. Detail · ${String(active.label)}`}
              subtitle="Selected strategy metrics, chart and trades"
              actions={<SignalBadge signal={active.signal} />}
            >
              <div className="metric-strip">
                <Metric label="Net return %" value={String(activeMetrics.net_return_pct ?? '—')} />
                <Metric label="CAGR %" value={String(activeMetrics.cagr_pct ?? '—')} />
                <Metric label="Sharpe" value={String(activeMetrics.sharpe ?? '—')} />
                <Metric
                  label="Max DD %"
                  value={String(activeMetrics.max_drawdown_pct ?? activeMetrics.max_drawdown ?? '—')}
                />
                <Metric label="Win rate" value={activeMetrics.win_rate == null ? '—' : pct(activeMetrics.win_rate)} />
                <Metric label="Trades" value={String(activeMetrics.trades ?? '—')} />
                <Metric label="Final value" value={money(activeMetrics.final_portfolio_value)} />
                <Metric label="Probability" value={active.proba == null ? '—' : String(active.proba)} />
              </div>
              <CandleChart
                rows={
                  equityProxy.length
                    ? equityProxy
                    : ((candles.data?.candles as Record<string, unknown>[]) ?? [])
                }
                height={280}
                showMa20={equityProxy.length === 0}
                showMa50={equityProxy.length === 0}
                freshness="historical"
                detail={equityProxy.length ? 'Simulated equity path' : `${ticker} stored price history`}
              />
              <Expandable title="Trades" defaultOpen>
                <Table rows={(active.trades_tail as Record<string, unknown>[]) ?? []} empty="No trades." />
              </Expandable>
            </Section>
          )}
        </>
      )}

      <AiInsight contextLabel="backtesting buy analysis" />
    </>
  )
}

export function SettingsPanel({ onClose }: { onClose: () => void }) {
  const diagnostics = useLoad(() => api.diagnostics())
  const health = useLoad(() => api.health())

  return (
    <div className="settings-drawer" role="dialog" aria-label="Settings">
      <div className="settings-head">
        <div>
          <h3>Settings</h3>
          <p className="muted tiny">Diagnostics · secrets never displayed</p>
        </div>
        <button type="button" className="ghost" onClick={onClose}>
          Close
        </button>
      </div>
      {diagnostics.loading && <LoadingState />}
      {diagnostics.error && <ErrorState message={diagnostics.error} />}
      {diagnostics.data && (
        <>
          <div className="metric-strip dense">
            <Metric label="API base" value={<span className="mono tiny">{apiBase()}</span>} />
            <Metric
              label="Health"
              value={String((health.data as Record<string, unknown> | null)?.status ?? (health.error ? 'error' : '…'))}
            />
            <Metric label="Broker execution" value="disabled" />
          </div>
          <Expandable title="Environment variables" defaultOpen>
            <Table rows={(diagnostics.data.environment as Record<string, unknown>[]) ?? []} />
          </Expandable>
          <button type="button" className="ghost" onClick={() => void diagnostics.reload()}>
            Refresh
          </button>
        </>
      )}
    </div>
  )
}

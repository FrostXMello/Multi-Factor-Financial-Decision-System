import { useMemo, useState, type FormEvent } from 'react'
import { api, apiBase } from './api'
import { CandleChart } from './charts'
import {
  AiInsight,
  Badge,
  EnvBanner,
  ErrorState,
  Expandable,
  FreshnessBadge,
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

function pickCols(rows: Record<string, unknown>[], preferred: string[]) {
  if (!rows.length) return preferred
  const available = new Set(Object.keys(rows[0]))
  const cols = preferred.filter((c) => available.has(c))
  return cols.length ? cols : Object.keys(rows[0])
}

export function LiveAnalysisPage() {
  const status = useLoad(() => api.liveStatus())
  const candidates = useLoad(() => api.liveCandidates())
  const [selected, setSelected] = useState('RELIANCE.NS')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const candles = useLoad(() => api.marketCandles(selected, 220), [selected])

  const rows = useMemo(() => {
    const list = (candidates.data?.candidates as Record<string, unknown>[]) ?? []
    return list
  }, [candidates.data])

  const selectedRow = rows.find((r) => String(r.symbol) === selected) ?? rows[0]
  const run = (candidates.data?.run ?? status.data?.latest_completed ?? null) as Record<string, unknown> | null
  const locked = String(status.data?.live_feed_state ?? 'locked') !== 'open'

  const scanNow = async () => {
    setBusy(true)
    setError(null)
    setMessage(null)
    try {
      const result = await api.liveScan()
      setMessage(`Scan ${(result.run as Record<string, unknown> | undefined)?.status ?? 'finished'}`)
      await Promise.all([status.reload(), candidates.reload()])
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <PageHeader
        title="Live Analysis"
        subtitle="Continuous market scanner over the supported NSE universe — ranked BUY / SELL / HOLD."
        badges={
          <>
            <Badge label={locked ? 'LIVE LOCKED' : 'LIVE VERIFIED'} tone={locked ? 'unavailable' : 'live'} />
            <FreshnessBadge value={run?.freshness ?? 'delayed'} />
          </>
        }
        actions={
          <button type="button" onClick={() => void scanNow()} disabled={busy}>
            {busy ? 'Scanning…' : 'Scan now'}
          </button>
        }
      />

      <EnvBanner
        mode={locked ? 'SIMULATED / RESEARCH SCAN' : 'VERIFIED LIVE'}
        detail={String(status.data?.live_feed_detail ?? status.data?.note ?? '')}
      />

      {status.error && <ErrorState message={status.error} />}
      {candidates.error && <ErrorState message={candidates.error} />}
      {error && <ErrorState message={error} />}
      {message && <div className="ok">{message}</div>}

      <div className="metric-strip">
        <Metric label="Scanner" value={String(status.data?.scanner_state ?? '—')} />
        <Metric label="Last scan" value={String(run?.finished_at ?? run?.started_at ?? '—')} />
        <Metric label="Next scan" value={String(status.data?.next_scan_at ?? run?.next_scan_at ?? '—')} />
        <Metric label="Source" value={String(run?.source ?? status.data?.note ? 'yfinance' : '—')} />
        <Metric label="Data mode" value={<Badge label={String(run?.data_mode ?? 'RESEARCH')} tone="simulated" />} />
        <Metric label="Universe" value={String(((status.data?.universe as string[]) ?? []).length)} />
      </div>

      {locked && (
        <div className="lock-panel compact-lock">
          <h3>Verified live tape locked</h3>
          <p>
            Results below are from the research scanner using delayed/historical data. They are never shown as a
            verified live tape until a real provider passes market-session verification.
          </p>
        </div>
      )}

      <div className="layout-split">
        <Section title="Strongest opportunities" subtitle="Ranked by model probability and risk gates">
          {candidates.loading && !rows.length ? <LoadingState label="Loading candidates…" /> : null}
          <div className="candidate-list">
            {rows.slice(0, 12).map((row) => {
              const symbol = String(row.symbol)
              const active = symbol === (selectedRow ? String(selectedRow.symbol) : selected)
              return (
                <button
                  key={symbol}
                  type="button"
                  className={`candidate-row${active ? ' active' : ''}`}
                  onClick={() => setSelected(symbol)}
                >
                  <span className="mono">{symbol}</span>
                  <SignalBadge signal={row.signal} />
                  <span className="mono tiny">P {row.proba == null ? '—' : Number(row.proba).toFixed(3)}</span>
                  {row.no_trade ? <Badge label="RISK BLOCK" tone="unavailable" /> : null}
                </button>
              )
            })}
            {!rows.length && <p className="muted">No stored scan yet. Click Scan now.</p>}
          </div>
        </Section>

        <Section title="Watchlist / scan board" subtitle="Persisted across refresh">
          <Table
            rows={rows}
            columns={pickCols(rows, ['symbol', 'signal', 'proba', 'rank_score', 'risk_score', 'no_trade', 'close'])}
            empty="No candidates stored."
          />
        </Section>
      </div>

      <Section
        title={selectedRow ? `${String(selectedRow.symbol)} detail` : 'Symbol detail'}
        subtitle="Interactive chart, probability, risk decision and AI explanation"
      >
        {selectedRow ? (
          <div className="metric-strip dense">
            <Metric label="Signal" value={<SignalBadge signal={selectedRow.signal} />} />
            <Metric label="Probability" value={selectedRow.proba == null ? '—' : Number(selectedRow.proba).toFixed(4)} />
            <Metric label="Risk score" value={String(selectedRow.risk_score ?? '—')} />
            <Metric label="Position size" value={String(selectedRow.position_size ?? '—')} />
            <Metric label="NoTrade" value={String(Boolean(selectedRow.no_trade))} />
            <Metric label="Close" value={money(selectedRow.close)} />
          </div>
        ) : null}
        <CandleChart
          rows={(candles.data?.candles as Record<string, unknown>[]) ?? []}
          freshness={String(run?.freshness ?? 'historical')}
          detail={`${selected} · scanner chart uses stored history`}
        />
        {selectedRow ? (
          <>
            <Expandable title="Why selected" subtitle="supporting factors & invalidation" defaultOpen>
              <Table rows={(selectedRow.reasons as Record<string, unknown>[]) ?? []} empty="No reasons recorded." />
              {(selectedRow.explanation as Record<string, unknown>)?.summary ? (
                <p className="ai-summary" style={{ marginTop: '0.75rem' }}>
                  {String((selectedRow.explanation as Record<string, unknown>).summary)}
                </p>
              ) : null}
            </Expandable>
            <Expandable title="Signal / model metrics">
              <pre className="code-block">{JSON.stringify(selectedRow.metrics ?? {}, null, 2)}</pre>
            </Expandable>
          </>
        ) : null}
      </Section>

      <AiInsight contextLabel="live analysis" />
      <p className="disclaimer">Scanner never executes real-money trades. Broker execution remains disabled.</p>
    </>
  )
}


function defaultRange(): { start: string; end: string } {
  const end = new Date()
  const start = new Date()
  start.setFullYear(end.getFullYear() - 1)
  const iso = (d: Date) => d.toISOString().slice(0, 10)
  return { start: iso(start), end: iso(end) }
}

export function BacktestingPage() {
  const defaults = useLoad(() => api.backtestDefaults())
  const symbols = useLoad(() => api.backtestSymbols())
  const initial = defaultRange()
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const [filter, setFilter] = useState('')
  const [start, setStart] = useState(initial.start)
  const [end, setEnd] = useState(initial.end)
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
  const windowInfo = (sim?.window as Record<string, unknown> | null) ?? null
  const active =
    strategies.find((s) => String(s.strategy_id) === selectedStrategy) ??
    (sim?.best as Record<string, unknown> | null) ??
    null
  const activeMetrics = (active?.metrics ?? {}) as Record<string, unknown>
  const orders = (active?.orders as Record<string, unknown>[]) ?? []
  const equity =
    (active?.equity_curve as Record<string, unknown>[]) ??
    (active?.equity_tail as Record<string, unknown>[]) ??
    []
  const equityProxy = equity.map((row) => {
    const value = Number(row.equity)
    return { Date: row.date, Open: value, High: value, Low: value, Close: value, Volume: 0 }
  })

  const runSimulation = async (event?: FormEvent) => {
    event?.preventDefault()
    if (!start || !end) {
      setError('Start and end dates are required. Simulation only uses that window.')
      return
    }
    if (start > end) {
      setError('End date must be on or after start date.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const payload = await api.backtestSimulateStrategies({
        ticker,
        start,
        end,
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
        subtitle="Pick a stock + date range, simulate five strategies, get buy/sell orders only inside that window."
        badges={
          <>
            <Badge label="HISTORICAL" tone="historical" />
            <Badge label="NO LIVE ORDERS" tone="muted" />
          </>
        }
      />
      <EnvBanner
        mode="DATE-WINDOW SIMULATION"
        detail="Outside the selected start/end dates is blocked from results. Extra history is only warm-up for indicators."
      />

      <Section title="Setup" subtitle="Keep it simple — stock, dates, capital, then run">
        <form className="toolbar" onSubmit={runSimulation}>
          <label>
            Search
            <input value={filter} onChange={(e) => setFilter(e.target.value.toUpperCase())} placeholder="RELIANCE" />
          </label>
          <label>
            Stock
            <input value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} required />
          </label>
          <label>
            Start
            <input type="date" value={start} onChange={(e) => setStart(e.target.value)} required />
          </label>
          <label>
            End
            <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} required />
          </label>
          <label>
            Capital
            <input type="number" min={1000} value={capital} onChange={(e) => setCapital(Number(e.target.value))} />
          </label>
          <button type="submit" disabled={busy || !ticker || !start || !end}>
            {busy ? 'Simulating…' : 'Run simulation'}
          </button>
        </form>
        <div className="stock-grid">
          {universe.slice(0, 48).map((symbol) => (
            <button
              key={symbol}
              type="button"
              className={`stock-chip${symbol === ticker ? ' active' : ''}`}
              onClick={() => setTicker(symbol)}
            >
              {symbol.replace('.NS', '')}
            </button>
          ))}
        </div>
        <div className="strategy-grid" style={{ marginTop: '0.75rem' }}>
          {presets.map((preset) => (
            <div key={String(preset.id)} className="strategy-card">
              <strong>{String(preset.label)}</strong>
              <p className="muted tiny">{String(preset.blurb)}</p>
            </div>
          ))}
        </div>
      </Section>

      {error && <ErrorState message={error} />}
      {busy && <LoadingState label={`Simulating five strategies on ${ticker} for ${start} → ${end}…`} />}

      {sim && (
        <>
          <div className="metric-strip">
            <Metric label="Ticker" value={String(sim.ticker)} />
            <Metric label="Window" value={`${String(windowInfo?.start ?? start)} → ${String(windowInfo?.end ?? end)}`} />
            <Metric label="Outside window" value="blocked" />
            <Metric label="Best sleeve" value={String((sim.best as Record<string, unknown> | null)?.label ?? '—')} />
          </div>

          <Section title="Strategy ranking" subtitle={String(sim.ranking_note ?? '')}>
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
                    <span className="mono tiny">orders {String(((row.orders as unknown[]) ?? []).length)}</span>
                  </button>
                )
              })}
            </div>
          </Section>

          {active && (
            <Section
              title={`Selected · ${String(active.label)}`}
              subtitle="Analytics inside your date window only"
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
                <Metric label="Final value" value={money(activeMetrics.final_portfolio_value)} />
                <Metric label="Buy/Sell signals" value={String(orders.length)} />
                <Metric label="Probability" value={active.proba == null ? '—' : String(active.proba)} />
              </div>

              <div className="layout-split">
                <div>
                  <h4 className="subtle-title">Equity curve</h4>
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
                    detail={equityProxy.length ? `Equity ${start}→${end}` : `${ticker} prices`}
                  />
                </div>
                <div>
                  <h4 className="subtle-title">Buy / sell orders (window only)</h4>
                  <Table
                    rows={orders}
                    columns={pickCols(orders, ['date', 'side', 'price', 'proba', 'risk_score', 'no_trade'])}
                    empty="No BUY/SELL signals inside this date window."
                  />
                </div>
              </div>

              <Expandable title="Closed trades (P&amp;L)" defaultOpen>
                <Table rows={(active.trades_tail as Record<string, unknown>[]) ?? []} empty="No closed trades in window." />
              </Expandable>
            </Section>
          )}

          {buy?.available ? (
            <Section
              title="Why buy / why caution"
              subtitle={`${String(buy.ticker)} · ${String(buy.best_strategy)}`}
              actions={
                <Badge
                  label={String(buy.recommendation ?? '')}
                  tone={String(buy.latest_signal) === 'BUY' ? 'buy' : 'hold'}
                />
              }
            >
              <div className="layout-split">
                <div>
                  <h4 className="subtle-title">Buy case</h4>
                  <ul className="reason-list">
                    {((buy.why_buy as string[]) ?? []).map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="subtle-title">Caution</h4>
                  <ul className="reason-list caution">
                    {((buy.why_caution as string[]) ?? []).map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              </div>
              {(buy.explanation as Record<string, unknown>)?.summary ? (
                <div className="ai-insight compact" style={{ marginTop: '0.85rem' }}>
                  <span className="ai-kicker">AI explanation</span>
                  <p className="ai-summary">{String((buy.explanation as Record<string, unknown>).summary)}</p>
                </div>
              ) : null}
            </Section>
          ) : null}
        </>
      )}

      <AiInsight contextLabel="backtesting" />
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

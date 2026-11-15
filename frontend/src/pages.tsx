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

export function PortfolioPage() {
  const [mode, setMode] = useState<'paper' | 'demo'>('paper')
  const account = useLoad(() => api.paperAccount(mode), [mode])
  const [symbol, setSymbol] = useState('RELIANCE.NS')
  const [side, setSide] = useState('BUY')
  const [qty, setQty] = useState(1)
  const [confirm, setConfirm] = useState(false)
  const [allowHistorical, setAllowHistorical] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [allocation, setAllocation] = useState<Record<string, unknown> | null>(null)
  const [allocError, setAllocError] = useState<string | null>(null)

  const snap = (account.data?.snapshot ?? {}) as Record<string, unknown>
  const positions = (snap.positions as Record<string, unknown>[]) ?? []
  const orders = (snap.orders as Record<string, unknown>[]) ?? []

  const loadAllocation = async () => {
    setAllocError(null)
    try {
      setAllocation(await api.analyticsPortfolio({ mode }))
    } catch (err) {
      setAllocation(null)
      setAllocError(err instanceof Error ? err.message : String(err))
    }
  }

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
      setMessage(
        `Order ${String((result.order as Record<string, unknown>)?.side)} ${(result.order as Record<string, unknown>)?.symbol}`,
      )
      await account.reload()
      await loadAllocation()
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
      <PageHeader
        title="Portfolio"
        subtitle="Paper trading, positions, orders, cash, P&L, allocation and risk."
        badges={
          <>
            <Badge label={mode === 'demo' ? 'DEMO' : 'PAPER'} tone={mode === 'demo' ? 'demo' : 'muted'} />
            <Badge label="NO BROKER EXECUTION" tone="muted" />
          </>
        }
      />
      <EnvBanner detail="Demo fills stay on demo:academic and never write into paper:default." />

      <div className="toolbar">
        <label>
          Account
          <select value={mode} onChange={(e) => setMode(e.target.value as 'paper' | 'demo')}>
            <option value="paper">paper:default</option>
            <option value="demo">demo:academic</option>
          </select>
        </label>
        <button type="button" className="ghost" onClick={() => void account.reload()}>
          Refresh
        </button>
        <button type="button" className="ghost" onClick={() => void api.paperReset(mode).then(() => account.reload())}>
          Reset account
        </button>
        <button type="button" onClick={() => void runDemo()}>
          Run demo workflow
        </button>
        <button type="button" className="ghost" onClick={() => void loadAllocation()}>
          Load allocation
        </button>
      </div>

      {account.error && <ErrorState message={account.error} />}
      {account.loading && !account.data ? <LoadingState label="Loading portfolio…" /> : null}

      <div className="metric-strip">
        <Metric label="Cash" value={money(snap.cash)} />
        <Metric label="Invested" value={money(snap.invested)} />
        <Metric label="Equity" value={money(snap.equity)} />
        <Metric label="Net P&L" value={money(snap.net_pnl)} />
        <Metric label="Return" value={pct(snap.total_return)} />
        <Metric label="Drawdown" value={pct(snap.drawdown)} />
        <Metric label="Win rate" value={snap.win_rate == null ? 'n/a' : pct(snap.win_rate)} />
        <Metric label="Profit factor" value={snap.profit_factor == null ? 'n/a' : String(snap.profit_factor)} />
      </div>

      <div className="layout-split">
        <Section title="Positions" subtitle={`Account ${String(account.data?.account_id ?? mode)}`}>
          <Table
            rows={positions}
            columns={pickCols(positions, ['symbol', 'quantity', 'avg_price', 'mark', 'unrealized_pnl', 'freshness'])}
            empty="No open positions."
          />
        </Section>
        <Section title="Place order" subtitle="Requires confirm. Historical marks only when allowed.">
          <form className="toolbar stack" onSubmit={submit}>
            <label>
              Symbol
              <input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
            </label>
            <label>
              Side
              <select value={side} onChange={(e) => setSide(e.target.value)}>
                <option>BUY</option>
                <option>SELL</option>
              </select>
            </label>
            <label>
              Qty
              <input type="number" min={1} value={qty} onChange={(e) => setQty(Number(e.target.value))} />
            </label>
            <label className="check">
              <input type="checkbox" checked={confirm} onChange={(e) => setConfirm(e.target.checked)} />
              Confirm
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={allowHistorical || mode === 'demo'}
                disabled={mode === 'demo'}
                onChange={(e) => setAllowHistorical(e.target.checked)}
              />
              Allow historical/delayed
            </label>
            <button type="submit">Submit paper order</button>
          </form>
        </Section>
      </div>

      <Expandable title="Order history" defaultOpen>
        <Table
          rows={orders}
          columns={pickCols(orders, ['side', 'symbol', 'quantity', 'fill_price', 'status', 'created_at', 'freshness'])}
          empty="No orders yet."
        />
      </Expandable>

      <Expandable title="Allocation, exposure & concentration">
        {allocError && <ErrorState message={allocError} />}
        {allocation ? (
          <>
            <div className="metric-strip dense">
              <Metric label="NAV" value={money(allocation.nav)} />
              <Metric label="HHI" value={String((allocation.concentration as Record<string, unknown>)?.hhi ?? '—')} />
              <Metric label="Top weight" value={pct((allocation.concentration as Record<string, unknown>)?.top_weight)} />
            </div>
            <Table rows={(allocation.allocation as Record<string, unknown>[]) ?? []} empty="No allocation rows." />
            <Table rows={(allocation.sector_exposure as Record<string, unknown>[]) ?? []} empty="No sector exposure." />
          </>
        ) : (
          <p className="muted">Load allocation to inspect concentration risk.</p>
        )}
      </Expandable>

      {message && <div className="ok">{message}</div>}
      {error && <ErrorState message={error} />}
      <AiInsight contextLabel="portfolio & risk" compact />
    </>
  )
}

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
        badges={<Badge label="HISTORICAL" tone="historical" />}
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
                      <strong>#{String(row.rank)} {String(row.label)}</strong>
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
              actions={<Badge label={String(buy.recommendation ?? '')} tone={String(buy.latest_signal) === 'BUY' ? 'buy' : 'hold'} />}
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

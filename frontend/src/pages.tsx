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
  const saved = useLoad(() => api.backtestRuns())
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [capital, setCapital] = useState(100000)
  const [buyTh, setBuyTh] = useState(0.6)
  const [sellTh, setSellTh] = useState(0.4)
  const [brokerage, setBrokerage] = useState(0.0003)
  const [sttBuy, setSttBuy] = useState(0.001)
  const [sttSell, setSttSell] = useState(0.001)
  const [slippage, setSlippage] = useState(5)
  const [segment, setSegment] = useState('equity_delivery')
  const [posSize, setPosSize] = useState(1)
  const [minPos, setMinPos] = useState(0.18)
  const [volStop, setVolStop] = useState(0.07)
  const [ddStop, setDdStop] = useState(0.2)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [compareIds, setCompareIds] = useState<number[]>([])
  const [compare, setCompare] = useState<Record<string, unknown> | null>(null)

  const metrics = (result?.metrics ?? {}) as Record<string, unknown>
  const equity = (result?.equity_curve as Record<string, unknown>[]) ?? (result?.equity_tail as Record<string, unknown>[]) ?? []
  const candleProxy = equity.map((row) => {
    const value = Number(row.equity)
    return {
      Date: row.date,
      Open: value,
      High: value,
      Low: value,
      Close: value,
      Volume: 0,
    }
  })

  const run = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const summary = await api.backtestRun({
        ticker,
        start: start || null,
        end: end || null,
        initial_capital: capital,
        buy_threshold: buyTh,
        sell_threshold: sellTh,
        cost_segment: segment,
        brokerage_rate: brokerage,
        stt_buy_rate: sttBuy,
        stt_sell_rate: sttSell,
        slippage_bps: slippage,
        base_position_size: posSize,
        min_position_size: minPos,
        hard_vol_stop: volStop,
        hard_drawdown_stop: ddStop,
        save: true,
        label: `${ticker} manual`,
      })
      setResult(summary)
      await saved.reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const toggleCompare = (id: number) => {
    setCompareIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id].slice(0, 6)))
  }

  const runCompare = async () => {
    setCompare(await api.backtestCompare(compareIds))
  }

  return (
    <>
      <PageHeader
        title="Backtesting"
        subtitle="Historical simulations on stored / Yahoo research data. No live execution."
        badges={<Badge label="HISTORICAL" tone="historical" />}
      />
      <EnvBanner
        mode="HISTORICAL RESEARCH"
        detail={String(defaults.data?.lookahead_note ?? 'Purged evaluation and next-open fills are preserved.')}
      />

      <form className="toolbar" onSubmit={run}>
        <label>
          Ticker
          <input list="bt-symbols" value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} />
          <datalist id="bt-symbols">
            {((symbols.data?.symbols as string[]) ?? []).map((s) => (
              <option key={s} value={s} />
            ))}
          </datalist>
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
        <button type="submit" disabled={busy}>
          {busy ? 'Running…' : 'Run backtest'}
        </button>
      </form>

      <Expandable title="Strategy & risk parameters" subtitle="manual bounds only — optimizer disabled">
        <div className="toolbar">
          <label>
            Buy threshold
            <input type="number" step={0.01} min={0.5} max={0.95} value={buyTh} onChange={(e) => setBuyTh(Number(e.target.value))} />
          </label>
          <label>
            Sell threshold
            <input type="number" step={0.01} min={0.05} max={0.5} value={sellTh} onChange={(e) => setSellTh(Number(e.target.value))} />
          </label>
          <label>
            Base position
            <input type="number" step={0.05} min={0.05} max={1} value={posSize} onChange={(e) => setPosSize(Number(e.target.value))} />
          </label>
          <label>
            Min position
            <input type="number" step={0.01} min={0} max={1} value={minPos} onChange={(e) => setMinPos(Number(e.target.value))} />
          </label>
          <label>
            Vol stop
            <input type="number" step={0.01} min={0.01} max={1} value={volStop} onChange={(e) => setVolStop(Number(e.target.value))} />
          </label>
          <label>
            Drawdown stop
            <input type="number" step={0.01} min={0.01} max={1} value={ddStop} onChange={(e) => setDdStop(Number(e.target.value))} />
          </label>
        </div>
      </Expandable>

      <Expandable title="Costs" subtitle="brokerage · STT · slippage">
        <div className="toolbar">
          <label>
            Segment
            <select value={segment} onChange={(e) => setSegment(e.target.value)}>
              <option value="equity_delivery">equity_delivery</option>
              <option value="equity_intraday">equity_intraday</option>
              <option value="zero">zero</option>
            </select>
          </label>
          <label>
            Brokerage
            <input type="number" step={0.0001} min={0} value={brokerage} onChange={(e) => setBrokerage(Number(e.target.value))} />
          </label>
          <label>
            STT buy
            <input type="number" step={0.0001} min={0} value={sttBuy} onChange={(e) => setSttBuy(Number(e.target.value))} />
          </label>
          <label>
            STT sell
            <input type="number" step={0.0001} min={0} value={sttSell} onChange={(e) => setSttSell(Number(e.target.value))} />
          </label>
          <label>
            Slippage bps
            <input type="number" step={1} min={0} value={slippage} onChange={(e) => setSlippage(Number(e.target.value))} />
          </label>
        </div>
      </Expandable>

      {error && <ErrorState message={error} />}

      {result && (
        <>
          <Section title="Results" subtitle={`${String(result.ticker)} · ${String(result.freshness ?? 'historical')}`} actions={<SignalBadge signal={(result.latest as Record<string, unknown>)?.signal} />}>
            <div className="metric-strip">
              <Metric label="Net return %" value={String(metrics.net_return_pct ?? '—')} />
              <Metric label="CAGR %" value={String(metrics.cagr_pct ?? '—')} />
              <Metric label="Sharpe" value={String(metrics.sharpe ?? '—')} />
              <Metric label="Max DD %" value={String(metrics.max_drawdown_pct ?? metrics.max_drawdown ?? '—')} />
              <Metric label="Win rate" value={metrics.win_rate == null ? '—' : pct(metrics.win_rate)} />
              <Metric label="Trades" value={String(metrics.trades ?? '—')} />
              <Metric label="Final value" value={money(metrics.final_portfolio_value)} />
              <Metric label="Costs" value={money(metrics.total_transaction_costs)} />
            </div>
          </Section>

          <Section title="Equity curve" subtitle="Interactive chart of simulated portfolio equity">
            <CandleChart rows={candleProxy} height={280} showMa20={false} showMa50={false} freshness="historical" detail="Equity path (close=equity)" />
          </Section>

          <Expandable title="Trades" defaultOpen>
            <Table rows={(result.trades_tail as Record<string, unknown>[]) ?? []} empty="No trades." />
          </Expandable>

          <Expandable title="Reliability / walk-forward">
            <pre className="code-block">{JSON.stringify(result.reliability ?? {}, null, 2)}</pre>
          </Expandable>
        </>
      )}

      <Expandable title="Saved research runs" subtitle="compare configurations">
        <Table rows={((saved.data?.runs as Record<string, unknown>[]) ?? []).map((r) => ({
          id: r.id,
          ticker: r.ticker,
          label: r.label,
          created_at: r.created_at,
          freshness: r.freshness,
          select: compareIds.includes(Number(r.id)) ? 'yes' : 'no',
        }))} empty="No saved runs yet." />
        <div className="toolbar">
          {((saved.data?.runs as Record<string, unknown>[]) ?? []).slice(0, 8).map((r) => (
            <button key={String(r.id)} type="button" className="ghost" onClick={() => toggleCompare(Number(r.id))}>
              {compareIds.includes(Number(r.id)) ? '✓' : '+'} #{String(r.id)} {String(r.ticker)}
            </button>
          ))}
          <button type="button" onClick={() => void runCompare()} disabled={!compareIds.length}>
            Compare selected
          </button>
        </div>
        {compare && <pre className="code-block">{JSON.stringify(compare.comparisons ?? compare, null, 2)}</pre>}
      </Expandable>

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

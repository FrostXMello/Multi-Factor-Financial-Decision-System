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

export function OverviewPage() {
  const overview = useLoad(() => api.overview())
  const market = useLoad(() => api.marketStatus())
  const quotes = useLoad(() => api.marketQuotes())
  const research = useLoad(() => api.researchLast())
  const paper = useLoad(() => api.paperAccount('paper'))

  if (overview.loading && !overview.data) return <LoadingState label="Loading terminal overview…" />
  if (overview.error && !overview.data) return <ErrorState message={overview.error} />

  const data = overview.data ?? {}
  const paperSnap = (paper.data?.snapshot ?? data.paper_account ?? {}) as Record<string, unknown>
  const summary = (research.data?.summary ?? null) as Record<string, unknown> | null
  const latest = (summary?.latest ?? {}) as Record<string, unknown>
  const quoteRows = (quotes.data?.quotes as Record<string, unknown>[]) ?? []

  return (
    <>
      <PageHeader
        title="Overview"
        subtitle="Market snapshot, research signals and paper portfolio — research environment only."
        badges={<Badge label="NO LIVE ORDERS" tone="muted" />}
        actions={
          <button type="button" className="ghost" onClick={() => void overview.reload()}>
            Refresh
          </button>
        }
      />
      <EnvBanner detail={String(data.live_verification ?? market.data?.note ?? '')} />

      <div className="metric-strip">
        <Metric label="NSE cash" value={String(data.nse_cash_session ?? '—').toUpperCase()} />
        <Metric label="Yahoo research" value={<FreshnessBadge value={data.yahoo_freshness ?? 'delayed'} />} />
        <Metric
          label="Broker feed"
          value={
            data.broker_configured ? (
              <FreshnessBadge value={data.broker_freshness} />
            ) : (
              <Badge label="UNCONFIGURED" tone="unavailable" />
            )
          }
        />
        <Metric
          label="Data mode"
          value={<Badge label={String(market.data?.data_mode ?? 'RESEARCH')} tone="simulated" />}
        />
        <Metric label="Stored candles" value={String((data.market_database as Record<string, unknown>)?.candles ?? 0)} />
      </div>

      <div className="layout-split">
        <Section title="Watchlist" subtitle="Stored quotes keep provider freshness timestamps.">
          {quotes.error && <ErrorState message={quotes.error} />}
          <Table
            rows={quoteRows}
            columns={pickCols(quoteRows, ['symbol', 'last_price', 'quote_time', 'freshness', 'provider_id'])}
            empty="No stored quotes yet."
          />
        </Section>

        <Section title="Portfolio summary" subtitle="Isolated paper:default account.">
          {paper.error && <ErrorState message={paper.error} />}
          <div className="metric-strip dense">
            <Metric label="Cash" value={money(paperSnap.cash)} />
            <Metric label="Equity" value={money(paperSnap.equity ?? paperSnap.cash)} />
            <Metric
              label="Positions"
              value={String(
                Array.isArray(paperSnap.positions) ? paperSnap.positions.length : (paperSnap.positions ?? 0),
              )}
            />
            <Metric label="Return" value={pct(paperSnap.total_return)} />
            <Metric label="Drawdown" value={pct(paperSnap.drawdown)} />
            <Metric label="Net P&L" value={money(paperSnap.net_pnl)} />
          </div>
        </Section>
      </div>

      <Section
        title="Latest research signal"
        subtitle={summary ? String(summary.ticker ?? '') : 'No cached research run in this API process.'}
        actions={summary ? <SignalBadge signal={latest.signal} /> : undefined}
      >
        {summary ? (
          <div className="metric-strip dense">
            <Metric label="Probability" value={String(latest.proba ?? '—')} />
            <Metric
              label="OOS accuracy"
              value={String((summary.metrics as Record<string, unknown>)?.walk_forward_mean_accuracy ?? '—')}
            />
            <Metric
              label="Net return %"
              value={String((summary.metrics as Record<string, unknown>)?.net_return_pct ?? '—')}
            />
            <Metric label="Trades" value={String((summary.metrics as Record<string, unknown>)?.trades ?? '—')} />
          </div>
        ) : (
          <p className="muted">Open Research to run the technical pipeline on a ticker.</p>
        )}
      </Section>

      <AiInsight contextLabel="overview" compact />
      <p className="disclaimer">{String(data.disclaimer ?? '')}</p>
    </>
  )
}

export function ResearchPage() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showMa20, setShowMa20] = useState(true)
  const [showMa50, setShowMa50] = useState(true)

  const last = useLoad(() => api.researchLast())
  const reliability = useLoad(() => api.researchReliability())
  const candles = useLoad(() => api.marketCandles(ticker, 220), [ticker])
  const [regimes, setRegimes] = useState<Record<string, unknown> | null>(null)
  const [robustness, setRobustness] = useState<Record<string, unknown> | null>(null)
  const [report, setReport] = useState<Record<string, unknown> | null>(null)
  const [analyticsBusy, setAnalyticsBusy] = useState(false)
  const [analyticsError, setAnalyticsError] = useState<string | null>(null)

  const summary = (last.data?.summary ?? null) as Record<string, unknown> | null
  const latest = (summary?.latest ?? {}) as Record<string, unknown>
  const metrics = (summary?.metrics ?? {}) as Record<string, unknown>
  const candleRows = (candles.data?.candles as Record<string, unknown>[]) ?? []

  const runResearch = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await api.researchRun(ticker)
      await Promise.all([last.reload(), reliability.reload(), candles.reload()])
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const loadStrategy = async () => {
    setAnalyticsBusy(true)
    setAnalyticsError(null)
    const errors: string[] = []
    try {
      setRegimes(await api.analyticsRegimes({ symbol: ticker, lookback: 60 }))
    } catch (err) {
      setRegimes(null)
      errors.push(err instanceof Error ? err.message : String(err))
    }
    try {
      setRobustness(await api.analyticsRobustness({ use_last_research: true }))
    } catch (err) {
      setRobustness(null)
      errors.push(err instanceof Error ? err.message : String(err))
    }
    try {
      const body = await api.analyticsReport(ticker)
      setReport((body.report as Record<string, unknown>) ?? body)
    } catch (err) {
      setReport(null)
      errors.push(err instanceof Error ? err.message : String(err))
    }
    if (errors.length) setAnalyticsError(errors.join(' | '))
    setAnalyticsBusy(false)
  }

  return (
    <>
      <PageHeader
        title="Research"
        subtitle="Stock analysis, interactive charts, model reliability and contextual AI explanations."
        badges={<Badge label="DELAYED / HISTORICAL" tone="delayed" />}
      />
      <EnvBanner detail="Research uses Yahoo/delayed history and stored candles. Accuracy is not probability of profit." />

      <form className="toolbar" onSubmit={runResearch}>
        <label>
          Ticker
          <input value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} />
        </label>
        <button type="submit" disabled={busy}>
          {busy ? 'Running…' : 'Run research'}
        </button>
        <label className="check">
          <input type="checkbox" checked={showMa20} onChange={(e) => setShowMa20(e.target.checked)} />
          MA20
        </label>
        <label className="check">
          <input type="checkbox" checked={showMa50} onChange={(e) => setShowMa50(e.target.checked)} />
          MA50
        </label>
      </form>
      {error && <ErrorState message={error} />}

      <Section
        title="Prediction"
        subtitle={summary ? `Last run · ${String(summary.ticker ?? ticker)}` : 'No cached research yet'}
        actions={summary ? <SignalBadge signal={latest.signal} /> : undefined}
      >
        {summary ? (
          <div className="metric-strip">
            <Metric label="Signal" value={<SignalBadge signal={latest.signal} />} />
            <Metric label="Probability" value={String(latest.proba ?? '—')} />
            <Metric label="OOS accuracy" value={String(metrics.walk_forward_mean_accuracy ?? '—')} />
            <Metric label="Net return %" value={String(metrics.net_return_pct ?? '—')} />
            <Metric label="Final value" value={money(metrics.final_portfolio_value)} />
            <Metric label="Costs" value={money(metrics.total_transaction_costs)} />
          </div>
        ) : (
          <p className="muted">Submit a ticker to run the existing technical pipeline.</p>
        )}
      </Section>

      <Section
        title="Interactive chart"
        subtitle={String(candles.data?.detail ?? 'Stored daily OHLCV · never labelled live')}
      >
        {candles.loading && !candleRows.length ? <LoadingState label="Loading candles…" /> : null}
        {candles.error ? <ErrorState message={candles.error} /> : null}
        <CandleChart
          rows={candleRows}
          showMa20={showMa20}
          showMa50={showMa50}
          freshness="historical"
          detail={`${ticker} · delayed/historical store`}
        />
      </Section>

      <Expandable title="Model reliability" subtitle="walk-forward diagnostics" defaultOpen={false}>
        {reliability.error && <ErrorState message={reliability.error} />}
        {!reliability.data?.available ? (
          <p className="muted">{String(reliability.data?.detail ?? 'Run research first.')}</p>
        ) : (
          <>
            <div className="metric-strip dense">
              <Metric label="Ticker" value={String(reliability.data.ticker ?? '—')} />
              <Metric
                label="Mean fold"
                value={String((reliability.data.metrics as Record<string, unknown>)?.walk_forward_mean_accuracy ?? '—')}
              />
            </div>
            <pre className="code-block">{JSON.stringify(reliability.data.reliability ?? {}, null, 2)}</pre>
            <p className="disclaimer">{String(reliability.data.disclaimer ?? '')}</p>
          </>
        )}
      </Expandable>

      <Expandable title="Strategy research" subtitle="regimes · robustness · weaknesses" defaultOpen={false}>
        <div className="toolbar">
          <button type="button" className="ghost" onClick={() => void loadStrategy()} disabled={analyticsBusy}>
            {analyticsBusy ? 'Loading…' : 'Load strategy analytics'}
          </button>
        </div>
        {analyticsError && <ErrorState message={analyticsError} />}
        {regimes && (
          <div className="metric-strip dense">
            <Metric
              label="Latest regime"
              value={String((regimes.latest as Record<string, unknown>)?.regime_end_of_day ?? '—')}
            />
            <Metric
              label="Trailing return"
              value={String((regimes.latest as Record<string, unknown>)?.trailing_return ?? '—')}
            />
            <Metric label="Lookahead safe" value={String(regimes.lookahead_safe)} />
          </div>
        )}
        {robustness && (
          <div className="metric-strip dense">
            <Metric label="Robustness" value={String(robustness.available)} />
            <Metric label="Mean" value={String(robustness.mean ?? '—')} />
            <Metric label="Std" value={String(robustness.std ?? '—')} />
            <Metric label="Min fold" value={String(robustness.min ?? '—')} />
          </div>
        )}
        {report && (
          <pre className="code-block">
            {JSON.stringify(
              {
                regime_performance: report.regime_performance,
                weaknesses: report.weaknesses,
                unavailable: report.unavailable,
              },
              null,
              2,
            )}
          </pre>
        )}
      </Expandable>

      {summary ? (
        <Expandable title="Signal tape" subtitle="recent model signals">
          <Table rows={(summary.signals_tail as Record<string, unknown>[]) ?? []} />
        </Expandable>
      ) : null}

      <AiInsight contextLabel="research & charts" />
      <p className="disclaimer">{String(summary?.disclaimer ?? 'Research only. Not financial advice.')}</p>
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
        subtitle="Paper trading, positions, orders, allocation and risk — isolated virtual ledger."
        badges={
          <>
            <Badge label={mode === 'demo' ? 'DEMO' : 'PAPER'} tone={mode === 'demo' ? 'demo' : 'muted'} />
            <Badge label="NO BROKER EXECUTION" tone="muted" />
          </>
        }
      />
      <EnvBanner detail="Demo fills stay on demo:academic and never write into paper:default. Marks may be historical/delayed when allowed." />

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
        <button
          type="button"
          className="ghost"
          onClick={() => void api.paperReset(mode).then(() => account.reload())}
        >
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
        <Section title="Place order" subtitle="Requires confirm. Historical marks only when explicitly allowed.">
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

      <Expandable title="Closed trades">
        <Table rows={(snap.closed_trades as Record<string, unknown>[]) ?? []} empty="No closed trades." />
      </Expandable>

      <Expandable title="Allocation & concentration" subtitle="from strategy analytics">
        {allocError && <ErrorState message={allocError} />}
        {allocation ? (
          <>
            <div className="metric-strip dense">
              <Metric label="NAV" value={money(allocation.nav)} />
              <Metric label="Invested" value={money(allocation.invested)} />
              <Metric label="HHI" value={String((allocation.concentration as Record<string, unknown>)?.hhi ?? '—')} />
              <Metric
                label="Top weight"
                value={pct((allocation.concentration as Record<string, unknown>)?.top_weight)}
              />
            </div>
            <Table rows={(allocation.allocation as Record<string, unknown>[]) ?? []} empty="No allocation rows." />
            <Table
              rows={(allocation.sector_exposure as Record<string, unknown>[]) ?? []}
              empty="No sector exposure."
            />
          </>
        ) : (
          <p className="muted">Load allocation to inspect concentration risk for the selected account mode.</p>
        )}
      </Expandable>

      {message && <div className="ok">{message}</div>}
      {error && <ErrorState message={error} />}
      <AiInsight contextLabel="portfolio & paper trading" compact />
      <p className="disclaimer">
        Paper/demo only. Demo ledger never mixes into paper:default. Real broker order routing is disabled.
      </p>
    </>
  )
}

/** Unlocks only when market status reports verified LIVE mode. */
function liveGateOpen(status: Record<string, unknown> | null): boolean {
  if (!status) return false
  return String(status.data_mode ?? '').toUpperCase() === 'LIVE'
}

export function LiveMarketPage() {
  const status = useLoad(() => api.marketStatus())
  const quality = useLoad(() => api.marketQuality())
  const coverage = useLoad(() => api.marketCoverage())
  const [message, setMessage] = useState<string | null>(null)
  const unlocked = useMemo(() => liveGateOpen(status.data), [status.data])

  const collect = async () => {
    setMessage(null)
    try {
      const result = await api.marketCollect()
      setMessage(String(result.detail ?? 'Collect finished'))
      await status.reload()
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <>
      <PageHeader
        title="Live Market"
        subtitle="Dedicated real-time experience — gated until a verified live provider session exists."
        badges={
          unlocked ? (
            <Badge label="LIVE VERIFIED" tone="live" />
          ) : (
            <Badge label="LOCKED" tone="unavailable" />
          )
        }
      />

      {!unlocked ? (
        <div className="lock-panel">
          <div className="lock-icon" aria-hidden>
            ⌁
          </div>
          <h3>Live feed unavailable</h3>
          <p>
            This section stays locked until a real broker/live provider returns verified{' '}
            <strong>LIVE</strong> session quotes. Simulated, delayed, historical and demo data are intentionally
            excluded here so they are never presented as a live tape.
          </p>
          <div className="metric-strip dense">
            <Metric
              label="Current mode"
              value={<Badge label={String(status.data?.data_mode ?? '…')} tone="simulated" />}
            />
            <Metric label="NSE session" value={String(status.data?.nse_cash_session ?? '—').toUpperCase()} />
            <Metric
              label="Broker"
              value={
                (status.data?.providers as Record<string, unknown>[] | undefined)?.find(
                  (p) => p.provider_id === 'broker_live',
                )?.configured ? (
                  <FreshnessBadge
                    value={
                      (status.data?.providers as Record<string, unknown>[]).find((p) => p.provider_id === 'broker_live')
                        ?.freshness
                    }
                  />
                ) : (
                  <Badge label="UNCONFIGURED" tone="unavailable" />
                )
              }
            />
          </div>
          <p className="muted">{String(status.data?.note ?? 'Live NSE-session quote ingestion remains pending verification.')}</p>
        </div>
      ) : (
        <Section title="Verified live quotes" subtitle="Live mode confirmed by market status.">
          <p className="ok">Live data mode is active. Use provider tools carefully — still no broker order execution.</p>
        </Section>
      )}

      <Section title="Research data tools" subtitle="These operate on delayed/historical stores — labelled separately from live.">
        <EnvBanner mode="SIMULATED / RESEARCH TOOLS" detail="Collect and quality views never upgrade delayed rows to live." />
        <div className="toolbar">
          <button type="button" onClick={() => void collect()}>
            Collect watchlist once
          </button>
          <button type="button" className="ghost" onClick={() => void status.reload()}>
            Refresh status
          </button>
        </div>
        {message && <p className="ok">{message}</p>}
        {status.error && <ErrorState message={status.error} />}
        {status.data && (
          <>
            <Table rows={(status.data.providers as Record<string, unknown>[]) ?? []} />
            <p className="muted tiny">Checked {String(status.data.checked_at ?? '')}</p>
          </>
        )}
      </Section>

      <Expandable title="Coverage & quality" subtitle="stored history diagnostics">
        <Table rows={(coverage.data?.coverage as Record<string, unknown>[]) ?? []} empty="No coverage rows." />
        <Table rows={(quality.data?.quality_events as Record<string, unknown>[]) ?? []} empty="No quality events." />
        <Table
          rows={(quality.data?.provider_events as Record<string, unknown>[]) ?? []}
          empty="No provider events."
        />
      </Expandable>

      <AiInsight contextLabel="market data environment" compact />
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
              value={String(
                (health.data as Record<string, unknown> | null)?.status ?? (health.error ? 'error' : '…'),
              )}
            />
            <Metric label="Broker execution" value="disabled" />
            <Metric label="Secrets exposed" value={String(diagnostics.data.secrets_exposed ?? false)} />
          </div>
          {health.error && <ErrorState message={health.error} />}
          <p className="muted tiny">
            {String((health.data as Record<string, unknown> | null)?.live_feed_verification ?? '')}
          </p>
          <Expandable title="Environment variables" defaultOpen>
            <Table rows={(diagnostics.data.environment as Record<string, unknown>[]) ?? []} />
          </Expandable>
          <Expandable title="Runtime / stores">
            <pre className="code-block">
              {JSON.stringify(
                {
                  runtime: diagnostics.data.runtime,
                  market_store: diagnostics.data.market_store,
                  paper_store: diagnostics.data.paper_store,
                  ai_reasoning: diagnostics.data.ai_reasoning,
                },
                null,
                2,
              )}
            </pre>
          </Expandable>
          <Badge
            label={diagnostics.data.broker_execution_enabled ? 'LIVE EXECUTION' : 'NO BROKER EXECUTION'}
            tone="muted"
          />
          <div className="toolbar">
            <button type="button" className="ghost" onClick={() => void diagnostics.reload()}>
              Refresh
            </button>
          </div>
        </>
      )}
    </div>
  )
}

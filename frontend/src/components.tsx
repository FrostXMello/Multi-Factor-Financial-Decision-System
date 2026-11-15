import { useState, type ReactNode } from 'react'
import { api } from './api'
import { useLoad } from './hooks'

export function Badge({ label, tone }: { label: string; tone?: string }) {
  const key = (tone ?? label).toLowerCase().replace(/\s+/g, '_')
  return <span className={`badge ${key}`}>{label}</span>
}

export function FreshnessBadge({ value }: { value: unknown }) {
  const label = String(value ?? 'unavailable').toUpperCase()
  return <Badge label={label} tone={label.toLowerCase()} />
}

export function SignalBadge({ signal }: { signal: unknown }) {
  const s = String(signal ?? '—').toUpperCase()
  const tone = s === 'BUY' ? 'buy' : s === 'SELL' ? 'sell' : s === 'HOLD' ? 'hold' : 'muted'
  return <Badge label={s} tone={tone} />
}

export function EnvBanner({
  mode = 'SIMULATED / RESEARCH',
  detail,
}: {
  mode?: string
  detail?: string
}) {
  return (
    <div className="env-banner" role="status">
      <div>
        <Badge label={mode} tone="simulated" />
        <span className="env-title">Data environment</span>
      </div>
      <p>
        {detail ??
          'Historical, delayed and demo sources power this terminal. They are never labelled verified live.'}
      </p>
    </div>
  )
}

export function Metric({
  label,
  value,
  hint,
}: {
  label: string
  value: ReactNode
  hint?: string
}) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {hint ? <div className="hint">{hint}</div> : null}
    </div>
  )
}

export function Section({
  title,
  subtitle,
  actions,
  children,
}: {
  title: string
  subtitle?: string
  actions?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="section">
      <header className="section-head">
        <div>
          <h3>{title}</h3>
          {subtitle ? <p className="muted">{subtitle}</p> : null}
        </div>
        {actions ? <div className="section-actions">{actions}</div> : null}
      </header>
      <div className="section-body">{children}</div>
    </section>
  )
}

export function Expandable({
  title,
  subtitle,
  defaultOpen = false,
  children,
}: {
  title: string
  subtitle?: string
  defaultOpen?: boolean
  children: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={`expandable${open ? ' open' : ''}`}>
      <button type="button" className="expand-toggle" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span>
          <strong>{title}</strong>
          {subtitle ? <span className="muted"> · {subtitle}</span> : null}
        </span>
        <span className="chevron" aria-hidden>
          {open ? '▾' : '▸'}
        </span>
      </button>
      {open ? <div className="expand-body">{children}</div> : null}
    </div>
  )
}

export function Table({
  rows,
  columns,
  empty = 'No rows available.',
}: {
  rows: Record<string, unknown>[]
  columns?: string[]
  empty?: string
}) {
  if (!rows.length) return <p className="muted">{empty}</p>
  const cols = columns ?? Object.keys(rows[0])
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {cols.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx}>
              {cols.map((col) => (
                <td key={col}>{formatCell(row[col])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : '—'
  }
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

export function money(value: unknown): string {
  const n = Number(value)
  if (!Number.isFinite(n)) return '—'
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 })
}

export function pct(value: unknown): string {
  const n = Number(value)
  if (!Number.isFinite(n)) return '—'
  return `${(n * 100).toFixed(2)}%`
}

/** Contextual AI explanation — never a separate nav destination. */
export function AiInsight({
  contextLabel,
  compact = false,
}: {
  contextLabel: string
  compact?: boolean
}) {
  const status = useLoad(() => api.explainStatus())
  const last = useLoad(() => api.explainLast())
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const explanation = (last.data?.explanation ?? null) as Record<string, unknown> | null

  const run = async () => {
    setBusy(true)
    setError(null)
    try {
      await api.explainDecision({ use_last_research: true, use_llm: true })
      await last.reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className={`ai-insight${compact ? ' compact' : ''}`}>
      <div className="ai-head">
        <div>
          <span className="ai-kicker">AI insight · {contextLabel}</span>
          <p className="muted tiny">
            Explains structured model and risk outputs only. Cannot override signals, risk, or orders.
            {status.data ? ` Provider ${status.data.configured ? 'configured' : 'deterministic fallback'}.` : ''}
          </p>
        </div>
        <button type="button" className="ghost" onClick={() => void run()} disabled={busy}>
          {busy ? 'Explaining…' : explanation ? 'Refresh' : 'Explain'}
        </button>
      </div>
      {error && <div className="error">{error}</div>}
      {!explanation && !error && <p className="muted">Run research, then generate an explanation here.</p>}
      {explanation && (
        <div className="ai-body">
          <div className="ai-meta">
            <SignalBadge signal={(explanation.decision as Record<string, unknown>)?.signal} />
            <span className="mono">{String(explanation.ticker ?? '—')}</span>
            <Badge label={String(explanation.mode ?? 'deterministic')} tone="muted" />
          </div>
          <p className="ai-summary">{String(explanation.summary ?? '')}</p>
          {!compact && explanation.confidence_interpretation ? (
            <p className="muted">{String(explanation.confidence_interpretation)}</p>
          ) : null}
          {!compact && (explanation.risk_warnings as string[] | undefined)?.length ? (
            <ul className="risk-list">
              {(explanation.risk_warnings as string[]).map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : null}
          {!compact ? (
            <Expandable title="Supporting factors" subtitle="model features">
              <Table rows={(explanation.supporting_factors as Record<string, unknown>[]) ?? []} />
            </Expandable>
          ) : null}
          {!compact && (explanation.unavailable as unknown[] | undefined)?.length ? (
            <p className="muted tiny">Unavailable: {JSON.stringify(explanation.unavailable)}</p>
          ) : null}
        </div>
      )}
    </div>
  )
}

export function PageHeader({
  title,
  subtitle,
  badges,
  actions,
}: {
  title: string
  subtitle?: string
  badges?: ReactNode
  actions?: ReactNode
}) {
  return (
    <header className="page-header">
      <div>
        <div className="page-title-row">
          <h2>{title}</h2>
          {badges}
        </div>
        {subtitle ? <p className="muted">{subtitle}</p> : null}
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </header>
  )
}

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return <p className="muted pulse">{label}</p>
}

export function ErrorState({ message }: { message: string }) {
  return <div className="error">{message}</div>
}

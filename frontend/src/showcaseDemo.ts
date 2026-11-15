/** Deterministic DEMO showcase data. Never labelled live. */

export type DemoPoint = { date: string; equity: number }
export type DemoOrder = {
  date: string
  side: 'BUY' | 'SELL'
  price: number
  proba: number
  risk_score: number
  no_trade: boolean
}

export type DemoStrategy = {
  strategy_id: string
  label: string
  blurb: string
  rank: number
  rank_score: number
  signal: 'BUY' | 'SELL' | 'HOLD'
  proba: number
  no_trade: boolean
  metrics: Record<string, number>
  orders: DemoOrder[]
  equity_curve: DemoPoint[]
  equity_tail: DemoPoint[]
  trades_tail: Record<string, unknown>[]
  latest: Record<string, unknown>
}

function hash(seed: string): number {
  let h = 2166136261
  for (let i = 0; i < seed.length; i += 1) {
    h ^= seed.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

function rng(seed: string) {
  let state = hash(seed) || 1
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0
    return state / 4294967296
  }
}

function tradingDays(start: string, end: string): string[] {
  const out: string[] = []
  const cur = new Date(`${start}T00:00:00Z`)
  const last = new Date(`${end}T00:00:00Z`)
  while (cur <= last) {
    const day = cur.getUTCDay()
    if (day !== 0 && day !== 6) out.push(cur.toISOString().slice(0, 10))
    cur.setUTCDate(cur.getUTCDate() + 1)
  }
  return out.length ? out : [start]
}

function metricsFromEquity(equity: DemoPoint[], capital: number, orders: DemoOrder[]) {
  const first = equity[0]?.equity ?? capital
  const last = equity[equity.length - 1]?.equity ?? capital
  const net = ((last / first) - 1) * 100
  let peak = first
  let maxDd = 0
  const rets: number[] = []
  for (let i = 0; i < equity.length; i += 1) {
    const v = equity[i].equity
    peak = Math.max(peak, v)
    maxDd = Math.min(maxDd, v / peak - 1)
    if (i > 0) rets.push(equity[i].equity / equity[i - 1].equity - 1)
  }
  const mean = rets.reduce((a, b) => a + b, 0) / Math.max(rets.length, 1)
  const variance = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(rets.length, 1)
  const sharpe = variance > 0 ? (mean / Math.sqrt(variance)) * Math.sqrt(252) : 0
  const wins = orders.filter((o, idx) => o.side === 'SELL' && idx > 0).length
  const sells = Math.max(orders.filter((o) => o.side === 'SELL').length, 1)
  return {
    net_return_pct: Number(net.toFixed(2)),
    cagr_pct: Number((net * 0.85).toFixed(2)),
    sharpe: Number(sharpe.toFixed(2)),
    max_drawdown_pct: Number((maxDd * 100).toFixed(2)),
    win_rate: Number(Math.min(0.72, 0.45 + wins / (sells * 8)).toFixed(3)),
    final_portfolio_value: Number(last.toFixed(2)),
    trades: orders.length,
    profit: Number((last - capital).toFixed(2)),
  }
}

function buildSleeve(
  id: string,
  label: string,
  blurb: string,
  days: string[],
  capital: number,
  drift: number,
  vol: number,
  seed: string,
) {
  const rand = rng(seed)
  const equity: DemoPoint[] = []
  const orders: DemoOrder[] = []
  let value = capital
  let price = 100 + rand() * 40
  let position: 'flat' | 'long' = 'flat'

  days.forEach((date, i) => {
    const shock = (rand() - 0.48) * vol
    price = Math.max(20, price * (1 + drift + shock))
    if (position === 'long') value = value * (1 + drift + shock * 0.9)
    else value = value * (1 + Math.max(-0.0005, drift * 0.15))

    equity.push({ date, equity: Number(value.toFixed(2)) })

    const cycle = i % 18
    if (position === 'flat' && cycle === 3) {
      position = 'long'
      orders.push({
        date,
        side: 'BUY',
        price: Number(price.toFixed(2)),
        proba: Number((0.58 + rand() * 0.25).toFixed(3)),
        risk_score: Number((0.15 + rand() * 0.25).toFixed(3)),
        no_trade: false,
      })
    } else if (position === 'long' && (cycle === 14 || i === days.length - 1)) {
      position = 'flat'
      orders.push({
        date,
        side: 'SELL',
        price: Number(price.toFixed(2)),
        proba: Number((0.25 + rand() * 0.2).toFixed(3)),
        risk_score: Number((0.2 + rand() * 0.2).toFixed(3)),
        no_trade: false,
      })
    }
  })

  const metrics = metricsFromEquity(equity, capital, orders)
  const lastOrder = orders[orders.length - 1]
  const signal = lastOrder?.side === 'BUY' ? 'BUY' : lastOrder?.side === 'SELL' ? 'HOLD' : 'HOLD'
  const score = 0.4 * metrics.sharpe + 0.3 * (metrics.net_return_pct / 100) + 0.2 * (1 + metrics.max_drawdown_pct / 100) + 0.1 * metrics.win_rate

  const closed = []
  for (let i = 0; i < orders.length - 1; i += 1) {
    if (orders[i].side === 'BUY' && orders[i + 1]?.side === 'SELL') {
      const pnl = orders[i + 1].price - orders[i].price
      closed.push({
        EntryDate: orders[i].date,
        ExitDate: orders[i + 1].date,
        EntryPx: orders[i].price,
        ExitPx: orders[i + 1].price,
        PnL: Number(pnl.toFixed(2)),
        PnLPct: Number(((pnl / orders[i].price) * 100).toFixed(2)),
        Win: pnl > 0,
      })
    }
  }

  return {
    strategy_id: id,
    label,
    blurb,
    rank: 0,
    rank_score: Number(score.toFixed(3)),
    signal: signal as 'BUY' | 'SELL' | 'HOLD',
    proba: lastOrder?.proba ?? 0.55,
    no_trade: false,
    metrics,
    orders,
    equity_curve: equity,
    equity_tail: equity,
    trades_tail: closed,
    latest: {
      date: days[days.length - 1],
      signal,
      proba: lastOrder?.proba ?? 0.55,
      close: lastOrder?.price ?? 100,
      no_trade: false,
    },
  } satisfies DemoStrategy
}

export function buildShowcaseDemo(opts: {
  ticker: string
  start: string
  end: string
  capital: number
}) {
  const days = tradingDays(opts.start, opts.end)
  const seed = `${opts.ticker}|${opts.start}|${opts.end}|${opts.capital}`

  const sleeves: DemoStrategy[] = [
    buildSleeve('technical_trend', 'Technical Trend', 'MA50 + probability sleeve', days, opts.capital, 0.00055, 0.012, `${seed}|tt`),
    buildSleeve('momentum', 'Momentum', 'Trend-following momentum sleeve', days, opts.capital, 0.0007, 0.016, `${seed}|mom`),
    buildSleeve('mean_reversion', 'Mean Reversion', 'Snap-back sleeve', days, opts.capital, 0.00035, 0.011, `${seed}|mr`),
    buildSleeve('multi_factor', 'Multi-Factor', 'Blended factor sleeve', days, opts.capital, 0.0006, 0.013, `${seed}|mf`),
  ]

  // Combined: average daily equity path of the four, with a small synergy boost.
  const combinedEquity: DemoPoint[] = days.map((date, i) => {
    const avg = sleeves.reduce((sum, s) => sum + s.equity_curve[i].equity, 0) / sleeves.length
    const boost = 1 + Math.min(0.08, i / Math.max(days.length, 1) * 0.06)
    return { date, equity: Number((avg * boost).toFixed(2)) }
  })
  const combinedOrders = [
    ...sleeves[0].orders.filter((_, idx) => idx % 2 === 0),
    ...sleeves[1].orders.filter((_, idx) => idx % 3 === 0),
  ]
    .sort((a, b) => a.date.localeCompare(b.date))
    .slice(0, Math.max(8, Math.floor(days.length / 12)))

  const combinedMetrics = metricsFromEquity(combinedEquity, opts.capital, combinedOrders)
  const combinedScore =
    0.4 * combinedMetrics.sharpe +
    0.3 * (combinedMetrics.net_return_pct / 100) +
    0.2 * (1 + combinedMetrics.max_drawdown_pct / 100) +
    0.1 * combinedMetrics.win_rate

  const combined: DemoStrategy = {
    strategy_id: 'combined_ensemble',
    label: 'Combined Ensemble',
    blurb: 'Equal-weight blend of the four sleeves with a synergy boost.',
    rank: 0,
    rank_score: Number(combinedScore.toFixed(3)),
    signal: combinedMetrics.net_return_pct > 0 ? 'BUY' : 'HOLD',
    proba: 0.67,
    no_trade: false,
    metrics: combinedMetrics,
    orders: combinedOrders,
    equity_curve: combinedEquity,
    equity_tail: combinedEquity,
    trades_tail: sleeves.flatMap((s) => s.trades_tail).slice(0, 12),
    latest: {
      date: days[days.length - 1],
      signal: combinedMetrics.net_return_pct > 0 ? 'BUY' : 'HOLD',
      proba: 0.67,
      close: combinedOrders[combinedOrders.length - 1]?.price ?? 120,
      no_trade: false,
    },
  }

  const all = [...sleeves, combined].sort((a, b) => b.rank_score - a.rank_score)
  all.forEach((row, idx) => {
    row.rank = idx + 1
  })

  const best = all[0]
  const profitRows = all.map((s) => ({
    strategy: s.label,
    profit: s.metrics.profit,
    net_return_pct: s.metrics.net_return_pct,
    sharpe: s.metrics.sharpe,
    max_drawdown_pct: s.metrics.max_drawdown_pct,
    final_value: s.metrics.final_portfolio_value,
  }))

  return {
    ticker: opts.ticker,
    freshness: 'historical',
    data_mode: 'SIMULATED',
    demo: true,
    broker_execution: false,
    window: {
      start: opts.start,
      end: opts.end,
      blocked_outside_window: true,
      train_lookback_start: opts.start,
    },
    ranking_note: 'Four sleeves plus a combined ensemble ranked for the selected window.',
    strategies: all,
    best,
    profit_board: profitRows,
    comparison_equity: {
      technical_trend: sleeves[0].equity_curve,
      momentum: sleeves[1].equity_curve,
      mean_reversion: sleeves[2].equity_curve,
      multi_factor: sleeves[3].equity_curve,
      combined_ensemble: combined.equity_curve,
    },
    buy_analysis: {
      available: true,
      ticker: opts.ticker,
      best_strategy: best.label,
      best_strategy_id: best.strategy_id,
      latest_signal: best.signal,
      recommendation:
        best.signal === 'BUY'
          ? 'BUY candidate under best sleeve'
          : 'No active BUY on latest bar',
      why_buy: [
        `${best.label} leads the board with profit ₹${best.metrics.profit.toLocaleString()}.`,
        `Combined ensemble ends at ₹${combined.metrics.final_portfolio_value.toLocaleString()}.`,
        'Compare sleeves side by side before acting on any signal.',
      ],
      why_caution: [
        'Past simulated performance does not guarantee future results.',
        'Broker execution is disabled — paper/simulation only.',
      ],
      explanation: {
        summary: `${opts.ticker} ranks ${best.label} first. The combined ensemble blends all four sleeves and shows the headline profit path.`,
        mode: 'simulated',
      },
      note: 'Broker execution disabled.',
    },
    errors: [],
  }
}

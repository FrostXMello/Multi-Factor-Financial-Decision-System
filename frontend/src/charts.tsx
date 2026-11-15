import { useEffect, useMemo, useRef, useState } from 'react'
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  HistogramSeries,
  LineSeries,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from 'lightweight-charts'
import { FreshnessBadge } from './components'

export type CandleRow = Record<string, unknown>

function toUnix(dateLike: unknown): number | null {
  if (dateLike == null) return null
  const raw = String(dateLike)
  const ms = Date.parse(raw.includes('T') ? raw : `${raw}T00:00:00Z`)
  if (!Number.isFinite(ms)) return null
  return Math.floor(ms / 1000)
}

function num(value: unknown): number | null {
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

function movingAverage(closes: { time: Time; value: number }[], period: number) {
  const out: { time: Time; value: number }[] = []
  let sum = 0
  for (let i = 0; i < closes.length; i += 1) {
    sum += closes[i].value
    if (i >= period) sum -= closes[i - period].value
    if (i >= period - 1) out.push({ time: closes[i].time, value: sum / period })
  }
  return out
}

function normalizeCandles(rows: CandleRow[]) {
  const candles: { time: Time; open: number; high: number; low: number; close: number }[] = []
  const volumes: { time: Time; value: number; color: string }[] = []
  const closes: { time: Time; value: number }[] = []

  for (const row of rows) {
    const t = toUnix(row.Date ?? row.date ?? row.bar_time)
    const open = num(row.Open ?? row.open)
    const high = num(row.High ?? row.high)
    const low = num(row.Low ?? row.low)
    const close = num(row.Close ?? row.close)
    const volume = num(row.Volume ?? row.volume) ?? 0
    if (t == null || open == null || high == null || low == null || close == null) continue
    const time = t as Time
    candles.push({ time, open, high, low, close })
    closes.push({ time, value: close })
    volumes.push({
      time,
      value: volume,
      color: close >= open ? 'rgba(61, 214, 140, 0.45)' : 'rgba(240, 113, 120, 0.45)',
    })
  }

  candles.sort((a, b) => Number(a.time) - Number(b.time))
  volumes.sort((a, b) => Number(a.time) - Number(b.time))
  closes.sort((a, b) => Number(a.time) - Number(b.time))
  return { candles, volumes, closes }
}

export function CandleChart({
  rows,
  height = 360,
  showMa20 = true,
  showMa50 = true,
  freshness = 'historical',
  detail,
}: {
  rows: CandleRow[]
  height?: number
  showMa20?: boolean
  showMa50?: boolean
  freshness?: string
  detail?: string
}) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const ma20Ref = useRef<ISeriesApi<'Line'> | null>(null)
  const ma50Ref = useRef<ISeriesApi<'Line'> | null>(null)
  const [hover, setHover] = useState<string>('Hover a bar for OHLC detail')

  const series = useMemo(() => normalizeCandles(rows), [rows])

  useEffect(() => {
    const el = hostRef.current
    if (!el) return

    const chart = createChart(el, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#9aa3b5',
        fontFamily: '"IBM Plex Sans", sans-serif',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(42, 46, 56, 0.65)' },
        horzLines: { color: 'rgba(42, 46, 56, 0.65)' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: 'rgba(124, 107, 240, 0.45)', labelBackgroundColor: '#1c1f2a' },
        horzLine: { color: 'rgba(124, 107, 240, 0.45)', labelBackgroundColor: '#1c1f2a' },
      },
      rightPriceScale: { borderColor: '#2a2e38' },
      timeScale: {
        borderColor: '#2a2e38',
        timeVisible: false,
      },
    })

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: '#3dd68c',
      downColor: '#f07178',
      borderUpColor: '#3dd68c',
      borderDownColor: '#f07178',
      wickUpColor: '#3dd68c',
      wickDownColor: '#f07178',
    })
    candles.priceScale().applyOptions({ scaleMargins: { top: 0.08, bottom: 0.22 } })

    const volumes = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    })
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    })

    const ma20 = chart.addSeries(LineSeries, {
      color: 'rgba(124, 107, 240, 0.95)',
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    })
    const ma50 = chart.addSeries(LineSeries, {
      color: 'rgba(230, 180, 80, 0.9)',
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    })

    chart.subscribeCrosshairMove((param) => {
      if (!param.time || !param.seriesData.size) {
        setHover('Hover a bar for OHLC detail')
        return
      }
      const candle = param.seriesData.get(candles) as
        | { open: number; high: number; low: number; close: number }
        | undefined
      const vol = param.seriesData.get(volumes) as { value: number } | undefined
      if (!candle) return
      setHover(
        `O ${candle.open.toFixed(2)}  H ${candle.high.toFixed(2)}  L ${candle.low.toFixed(2)}  C ${candle.close.toFixed(2)}${
          vol ? `  ·  Vol ${vol.value.toLocaleString()}` : ''
        }`,
      )
    })

    chartRef.current = chart
    candleRef.current = candles
    volumeRef.current = volumes
    ma20Ref.current = ma20
    ma50Ref.current = ma50

    const resize = () => {
      if (!hostRef.current) return
      chart.applyOptions({ width: hostRef.current.clientWidth })
    }
    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(el)

    return () => {
      observer.disconnect()
      chart.remove()
      chartRef.current = null
    }
  }, [height])

  useEffect(() => {
    if (!candleRef.current || !volumeRef.current || !ma20Ref.current || !ma50Ref.current) return
    candleRef.current.setData(series.candles)
    volumeRef.current.setData(series.volumes)
    ma20Ref.current.setData(showMa20 ? movingAverage(series.closes, 20) : [])
    ma50Ref.current.setData(showMa50 ? movingAverage(series.closes, 50) : [])
    chartRef.current?.timeScale().fitContent()
  }, [series, showMa20, showMa50])

  if (!rows.length) {
    return (
      <div className="chart-empty">
        <p className="muted">No stored candles for this symbol yet. Run research or collect history first.</p>
      </div>
    )
  }

  return (
    <div className="chart-shell">
      <div className="chart-toolbar">
        <div className="chart-legend">
          <FreshnessBadge value={freshness} />
          <span className="chip">Volume</span>
          {showMa20 ? <span className="chip accent">MA20</span> : null}
          {showMa50 ? <span className="chip warn">MA50</span> : null}
          <span className="muted tiny">{detail ?? `${series.candles.length} daily bars · stored history`}</span>
        </div>
        <div className="chart-hover mono">{hover}</div>
      </div>
      <div ref={hostRef} className="chart-canvas" style={{ height }} />
    </div>
  )
}

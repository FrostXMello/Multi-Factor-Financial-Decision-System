const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''

export function apiBase(): string {
  return API_BASE || '(same-origin / Vite proxy)'
}

function formatDetail(detail: unknown): string {
  if (detail == null) return 'Request failed'
  if (typeof detail === 'string') return detail
  try {
    return JSON.stringify(detail)
  } catch {
    return String(detail)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
      ...init,
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    throw new Error(`Network error talking to API ${apiBase()}: ${message}`)
  }
  const text = await response.text()
  let body: unknown = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = { detail: text }
    }
  }
  if (!response.ok) {
    const detail =
      typeof body === 'object' && body && 'detail' in body
        ? formatDetail((body as { detail: unknown }).detail)
        : response.statusText || `HTTP ${response.status}`
    throw new Error(detail)
  }
  return body as T
}

export const api = {
  health: () => request<Record<string, unknown>>('/api/health'),
  overview: () => request<Record<string, unknown>>('/api/overview'),
  diagnostics: () => request<Record<string, unknown>>('/api/diagnostics'),
  marketStatus: () => request<Record<string, unknown>>('/api/market/status'),
  marketQuotes: () => request<Record<string, unknown>>('/api/market/quotes'),
  marketCandles: (symbol: string, limit = 180) =>
    request<Record<string, unknown>>(
      `/api/market/candles/${encodeURIComponent(symbol)}?timeframe=1d&limit=${limit}`,
    ),
  marketCoverage: () => request<Record<string, unknown>>('/api/market/coverage'),
  marketQuality: () => request<Record<string, unknown>>('/api/market/quality'),
  marketCollect: () =>
    request<Record<string, unknown>>('/api/market/collect', { method: 'POST', body: '{}' }),
  paperAccount: (mode: 'paper' | 'demo') =>
    request<Record<string, unknown>>(`/api/paper/account?mode=${mode}`),
  paperOrder: (body: Record<string, unknown>) =>
    request<Record<string, unknown>>('/api/paper/orders', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  paperDemoRun: () =>
    request<Record<string, unknown>>('/api/paper/demo/run', { method: 'POST', body: '{}' }),
  paperReset: (mode: 'paper' | 'demo') =>
    request<Record<string, unknown>>('/api/paper/reset', {
      method: 'POST',
      body: JSON.stringify({ mode }),
    }),
  researchRun: (ticker: string) =>
    request<Record<string, unknown>>('/api/research/run', {
      method: 'POST',
      body: JSON.stringify({ ticker }),
    }),
  researchLast: () => request<Record<string, unknown>>('/api/research/last'),
  researchReliability: () => request<Record<string, unknown>>('/api/research/reliability'),
  explainStatus: () => request<Record<string, unknown>>('/api/explain/status'),
  explainLast: () => request<Record<string, unknown>>('/api/explain/last'),
  explainDecision: (body: Record<string, unknown> = {}) =>
    request<Record<string, unknown>>('/api/explain/decision', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  explainRiskBlock: (body: Record<string, unknown>) =>
    request<Record<string, unknown>>('/api/explain/risk-block', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  analyticsStatus: () => request<Record<string, unknown>>('/api/analytics/status'),
  analyticsReport: (symbol?: string) =>
    request<Record<string, unknown>>(
      symbol ? `/api/analytics/report?symbol=${encodeURIComponent(symbol)}` : '/api/analytics/report',
    ),
  analyticsRegimes: (body: Record<string, unknown>) =>
    request<Record<string, unknown>>('/api/analytics/regimes', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  analyticsPortfolio: (body: Record<string, unknown> = {}) =>
    request<Record<string, unknown>>('/api/analytics/portfolio', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  analyticsRobustness: (body: Record<string, unknown> = {}) =>
    request<Record<string, unknown>>('/api/analytics/robustness', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
}

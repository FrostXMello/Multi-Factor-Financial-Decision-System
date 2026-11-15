const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })
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
        ? JSON.stringify((body as { detail: unknown }).detail)
        : response.statusText
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
  marketCandles: (symbol: string) =>
    request<Record<string, unknown>>(`/api/market/candles/${encodeURIComponent(symbol)}`),
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
}

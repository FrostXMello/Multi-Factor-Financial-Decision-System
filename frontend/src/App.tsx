import { useEffect, useState } from 'react'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { api } from './api'
import { Badge } from './components'
import { useLoad } from './hooks'
import { BacktestingPage, SettingsPanel } from './pages'

export default function App() {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const location = useLocation()
  const health = useLoad(() => api.health())

  useEffect(() => {
    setSettingsOpen(false)
  }, [location.pathname])

  const apiHealthy = String((health.data as Record<string, unknown> | null)?.status ?? '') === 'ok'

  return (
    <div className="terminal">
      <header className="topbar topbar-single">
        <div className="brand-block">
          <div>
            <div className="brand">MF Decision Terminal</div>
            <div className="brand-sub">Backtesting · historical strategy research</div>
          </div>
        </div>

        <div className="top-status">
          <Badge label="BACKTESTING" tone="simulated" />
          <Badge label="HISTORICAL" tone="historical" />
          <span className="status-pill">{apiHealthy ? 'API OK' : health.error ? 'API DOWN' : 'API …'}</span>
          <button
            type="button"
            className={`icon-btn${settingsOpen ? ' active' : ''}`}
            aria-label="Settings"
            onClick={() => setSettingsOpen((v) => !v)}
          >
            ⚙
          </button>
        </div>
      </header>

      <div className="terminal-body">
        <main className="main-stage">
          <Routes>
            <Route path="/" element={<BacktestingPage />} />
            <Route path="/backtesting" element={<BacktestingPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>

        {settingsOpen ? (
          <>
            <button
              type="button"
              className="drawer-backdrop"
              aria-label="Close settings"
              onClick={() => setSettingsOpen(false)}
            />
            <SettingsPanel onClose={() => setSettingsOpen(false)} />
          </>
        ) : null}
      </div>
    </div>
  )
}

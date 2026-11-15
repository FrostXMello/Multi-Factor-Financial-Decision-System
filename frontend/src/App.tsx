import { useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { api } from './api'
import { Badge } from './components'
import { useLoad } from './hooks'
import { BacktestingPage, LiveAnalysisPage, SettingsPanel } from './pages'

const links = [
  { to: '/', label: 'Live', hint: 'Scanner & signals' },
  { to: '/backtesting', label: 'Backtesting', hint: 'Simulate strategies' },
]

export default function App() {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [navOpen, setNavOpen] = useState(false)
  const location = useLocation()
  const health = useLoad(() => api.health())
  const live = useLoad(() => api.liveStatus())

  useEffect(() => {
    setNavOpen(false)
    setSettingsOpen(false)
  }, [location.pathname])

  const apiHealthy = String((health.data as Record<string, unknown> | null)?.status ?? '') === 'ok'
  const liveState = String((live.data as Record<string, unknown> | null)?.live_feed_state ?? 'locked')

  return (
    <div className="terminal">
      <header className="topbar">
        <div className="brand-block">
          <button
            type="button"
            className="nav-burger ghost"
            aria-label="Toggle navigation"
            onClick={() => setNavOpen((v) => !v)}
          >
            ☰
          </button>
          <div>
            <div className="brand">MF Decision Terminal</div>
            <div className="brand-sub">Live analysis · backtesting</div>
          </div>
        </div>

        <nav className={`main-nav${navOpen ? ' open' : ''}`}>
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === '/'}
              className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
            >
              <span>{link.label}</span>
              <small>{link.hint}</small>
            </NavLink>
          ))}
        </nav>

        <div className="top-status">
          <Badge label="SIMULATED" tone="simulated" />
          <span className="status-pill">{apiHealthy ? 'API OK' : health.error ? 'API DOWN' : 'API …'}</span>
          <span className={`status-pill${liveState === 'open' ? ' live' : ''}`}>
            {liveState === 'open' ? 'LIVE OPEN' : 'LIVE LOCKED'}
          </span>
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
            <Route path="/" element={<LiveAnalysisPage />} />
            <Route path="/live" element={<LiveAnalysisPage />} />
            <Route path="/backtesting" element={<BacktestingPage />} />
            <Route path="/portfolio" element={<Navigate to="/backtesting" replace />} />
            <Route path="/research" element={<Navigate to="/backtesting" replace />} />
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

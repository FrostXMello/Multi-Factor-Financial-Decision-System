import { useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { api } from './api'
import { Badge } from './components'
import { useLoad } from './hooks'
import { BacktestingPage, LiveAnalysisPage, PortfolioPage, SettingsPanel } from './pages'

const links = [
  { to: '/', label: 'Live Analysis', hint: 'Scanner & signals' },
  { to: '/portfolio', label: 'Portfolio', hint: 'Paper & risk' },
  { to: '/backtesting', label: 'Backtesting', hint: 'History & sims' },
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

  const session = String((live.data as Record<string, unknown> | null)?.latest_completed
    ? ((live.data as Record<string, unknown>).latest_completed as Record<string, unknown>)?.session_status
    : (live.data as Record<string, unknown> | null)?.scanner_state ?? '—').toString().toUpperCase()
  const liveState = String((live.data as Record<string, unknown> | null)?.live_feed_state ?? 'locked')
  const apiHealthy = String((health.data as Record<string, unknown> | null)?.status ?? '') === 'ok'

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
            <div className="brand-sub">Live analysis · portfolio · backtesting</div>
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
          <span className="status-pill">SCAN {session}</span>
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
            <Route path="/portfolio" element={<PortfolioPage />} />
            <Route path="/backtesting" element={<BacktestingPage />} />
            <Route path="/live" element={<Navigate to="/" replace />} />
            <Route path="/research" element={<Navigate to="/backtesting" replace />} />
            <Route path="/market" element={<Navigate to="/" replace />} />
            <Route path="/reliability" element={<Navigate to="/backtesting" replace />} />
            <Route path="/analytics" element={<Navigate to="/backtesting" replace />} />
            <Route path="/explain" element={<Navigate to="/" replace />} />
            <Route path="/paper" element={<Navigate to="/portfolio" replace />} />
            <Route path="/settings" element={<Navigate to="/" replace />} />
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

import { useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { api } from './api'
import { Badge } from './components'
import { useLoad } from './hooks'
import {
  LiveMarketPage,
  OverviewPage,
  PortfolioPage,
  ResearchPage,
  SettingsPanel,
} from './pages'

const links = [
  { to: '/', label: 'Overview', hint: 'Snapshot' },
  { to: '/research', label: 'Research', hint: 'Signals & charts' },
  { to: '/portfolio', label: 'Portfolio', hint: 'Paper & risk' },
  { to: '/live', label: 'Live Market', hint: 'Verified feed' },
]

export default function App() {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [navOpen, setNavOpen] = useState(false)
  const location = useLocation()
  const health = useLoad(() => api.health())
  const market = useLoad(() => api.marketStatus())

  useEffect(() => {
    setNavOpen(false)
    setSettingsOpen(false)
  }, [location.pathname])

  const liveUnlocked = String(market.data?.data_mode ?? '').toUpperCase() === 'LIVE'
  const session = String(market.data?.nse_cash_session ?? '—').toUpperCase()
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
            <div className="brand-sub">Quantitative research · paper trading</div>
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
          <span className="status-pill">NSE {session}</span>
          <span className={`status-pill${liveUnlocked ? ' live' : ''}`}>
            {liveUnlocked ? 'LIVE OPEN' : 'LIVE LOCKED'}
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
            <Route path="/" element={<OverviewPage />} />
            <Route path="/research" element={<ResearchPage />} />
            <Route path="/portfolio" element={<PortfolioPage />} />
            <Route path="/live" element={<LiveMarketPage />} />
            <Route path="/market" element={<Navigate to="/live" replace />} />
            <Route path="/reliability" element={<Navigate to="/research" replace />} />
            <Route path="/analytics" element={<Navigate to="/research" replace />} />
            <Route path="/explain" element={<Navigate to="/research" replace />} />
            <Route path="/paper" element={<Navigate to="/portfolio" replace />} />
            <Route path="/settings" element={<Navigate to="/" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>

        {settingsOpen ? (
          <>
            <button type="button" className="drawer-backdrop" aria-label="Close settings" onClick={() => setSettingsOpen(false)} />
            <SettingsPanel onClose={() => setSettingsOpen(false)} />
          </>
        ) : null}
      </div>
    </div>
  )
}

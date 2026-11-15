import { NavLink, Route, Routes } from 'react-router-dom'
import {
  AnalyticsPage,
  ExplainPage,
  MarketPage,
  OverviewPage,
  PaperPage,
  ReliabilityPage,
  ResearchPage,
  SettingsPage,
} from './pages'

const links = [
  { to: '/', label: 'Overview' },
  { to: '/market', label: 'Market Data' },
  { to: '/research', label: 'Predictions' },
  { to: '/reliability', label: 'Backtesting / Reliability' },
  { to: '/analytics', label: 'Strategy Analytics' },
  { to: '/explain', label: 'AI Explainability' },
  { to: '/paper', label: 'Paper Trading' },
  { to: '/settings', label: 'Settings' },
]

export default function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>MF Decision Terminal</h1>
        <p>Research + paper trading. No live broker orders.</p>
        <nav>
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === '/'}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/market" element={<MarketPage />} />
          <Route path="/research" element={<ResearchPage />} />
          <Route path="/reliability" element={<ReliabilityPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/explain" element={<ExplainPage />} />
          <Route path="/paper" element={<PaperPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  )
}

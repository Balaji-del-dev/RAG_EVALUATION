import {
  Database,
  FileChartColumn,
  FileText,
  Gauge,
  GitCompareArrows,
  Layers,
  LayoutDashboard,
  ListChecks,
  MessageSquareText,
  MessagesSquare,
  Play,
  ScanSearch,
  Search,
  Shield,
  Timer,
} from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { api } from './api.js'
import { ProfileCard } from './components/ProfileCard.jsx'
import { ToastProvider } from './components/toast.jsx'
import Category from './pages/Category.jsx'
import Chat from './pages/Chat.jsx'
import Compare from './pages/Compare.jsx'
import Datasets from './pages/Datasets.jsx'
import Documents from './pages/Documents.jsx'
import Login from './pages/Login.jsx'
import Metrics from './pages/Metrics.jsx'
import Overview from './pages/Overview.jsx'
import Performance from './pages/Performance.jsx'
import Reports from './pages/Reports.jsx'
import Runs from './pages/Runs.jsx'
import Security from './pages/Security.jsx'

const NAVIGATION = [
  {
    group: 'Build',
    items: [
      ['/', 'Overview', LayoutDashboard],
      ['/documents', 'Documents', FileText],
      ['/chat', 'Chat', MessagesSquare],
    ],
  },
  {
    group: 'Test',
    items: [
      ['/datasets', 'Datasets', Database],
      ['/runs', 'Runs', Play],
    ],
  },
  {
    group: 'Results',
    items: [
      ['/metrics', 'All 37 metrics', ListChecks],
      ['/retrieval', 'Retrieval', Search],
      ['/context', 'Context', Layers],
      ['/answer', 'Answer', MessageSquareText],
      ['/hallucination', 'Hallucination', ScanSearch],
      ['/security', 'Security', Shield],
      ['/performance', 'Performance', Timer],
    ],
  },
  {
    group: 'Decide',
    items: [
      ['/compare', 'Compare', GitCompareArrows],
      ['/reports', 'Reports', FileChartColumn],
    ],
  },
]

export default function App() {
  const [user, setUser] = useState(undefined)
  const [signedOut, setSignedOut] = useState(false)
  const { pathname } = useLocation()

  useEffect(() => {
    api('accounts/csrf/')
      .then(() => api('accounts/me/'))
      .then(setUser)
      .catch(() => setUser(null))
  }, [])

  const signOut = useCallback(async () => {
    await api('accounts/logout/', { method: 'POST' })
    await api('accounts/csrf/')
    setSignedOut(true)
    setUser(null)
  }, [])

  if (user === undefined) return null
  if (user === null) return <Login onSignedIn={setUser} signedOut={signedOut} />

  const canEdit = user.role !== 'viewer' || user.is_superuser
  return (
    <ToastProvider>
    <div className="shell">
      <aside className="rail">
        <div className="rail-brand">
          <span className="rail-mark" title="RAG Evaluation Bench">
            <Gauge size={16} aria-hidden="true" />
          </span>
          <span className="rail-name">RAG Evaluation Bench</span>
        </div>
        <nav aria-label="Main">
          {NAVIGATION.map(({ group, items }) => (
            <div className="rail-group" key={group}>
              <div className="rail-group-name">{group}</div>
              {items.map(([to, label, Icon]) => (
                <NavLink key={to} to={to} end={to === '/'} className="rail-link" title={label} aria-label={label}>
                  <Icon size={16} aria-hidden="true" />
                  <span className="rail-label">{label}</span>
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <ProfileCard user={user} onSignOut={signOut} />
      </aside>
      <main className={pathname.startsWith('/chat') ? 'content content-full' : 'content'}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/documents" element={<Documents canEdit={canEdit} />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/chat/:sessionId" element={<Chat />} />
          <Route path="/playground" element={<Navigate to="/chat" replace />} />
          <Route path="/datasets" element={<Datasets canEdit={canEdit} />} />
          <Route path="/runs" element={<Runs canEdit={canEdit} />} />
          <Route path="/metrics" element={<Metrics />} />
          <Route path="/retrieval" element={<Category key="retrieval" category="retrieval" canEdit={canEdit} />} />
          <Route path="/context" element={<Category key="context" category="context" canEdit={canEdit} />} />
          <Route path="/answer" element={<Category key="answer" category="answer" canEdit={canEdit} />} />
          <Route path="/hallucination" element={<Category key="reliability" category="reliability" canEdit={canEdit} />} />
          <Route path="/security" element={<Security canEdit={canEdit} />} />
          <Route path="/performance" element={<Performance />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/reports" element={<Reports canEdit={canEdit} />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
    </ToastProvider>
  )
}

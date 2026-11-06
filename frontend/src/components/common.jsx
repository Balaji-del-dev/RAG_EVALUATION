import { AlertTriangle, Check, Clock, LoaderCircle, X } from 'lucide-react'
import { Link, useSearchParams } from 'react-router-dom'
import { results, useApi } from '../api.js'
import { dateTime, runLabel, score } from '../format.js'

const STATUS = {
  good: { Icon: Check, className: 'status-good' },
  warning: { Icon: AlertTriangle, className: 'status-warning' },
  critical: { Icon: X, className: 'status-critical' },
  neutral: { Icon: Clock, className: 'status-neutral' },
  busy: { Icon: LoaderCircle, className: 'status-busy' },
}

/** Colour never carries meaning alone: every status has an icon and a word. */
export function Status({ kind = 'neutral', children }) {
  const { Icon, className } = STATUS[kind]
  return (
    <span className={`status ${className}`}>
      <Icon size={12} strokeWidth={2.5} aria-hidden="true" />
      {children}
    </span>
  )
}

const JOB_STATUS = {
  completed: ['good', 'Completed'],
  ready: ['good', 'Ready'],
  running: ['busy', 'Running'],
  processing: ['busy', 'Processing'],
  pending: ['neutral', 'Queued'],
  uploaded: ['neutral', 'Queued'],
  failed: ['critical', 'Failed'],
}

export function JobStatus({ status }) {
  const [kind, text] = JOB_STATUS[status] ?? ['neutral', status]
  return <Status kind={kind}>{text}</Status>
}

const PRIORITY_CLASS = { 'Highly Critical': 'priority-top', Critical: 'priority-critical', High: 'priority-high', Low: 'priority-low' }

export function Priority({ value }) {
  if (!value) return null
  return <span className={`priority ${PRIORITY_CLASS[value] ?? ''}`}>{value}</span>
}

export function Page({ title, intro, actions, children }) {
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>{title}</h1>
          {intro && <p className="page-intro">{intro}</p>}
        </div>
        {actions && <div className="page-actions">{actions}</div>}
      </header>
      {children}
    </div>
  )
}

export function Panel({ title, note, actions, children, className = '' }) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <div className="panel-head">
          <div>
            {title && <h2>{title}</h2>}
            {note && <p className="panel-note">{note}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  )
}

export function Stat({ label, value, note }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {note && <div className="stat-note">{note}</div>}
    </div>
  )
}

export function Empty({ title, children, action }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  )
}

export function ErrorNote({ error }) {
  if (!error) return null
  return (
    <p className="error-note" role="alert">
      {error.message ?? String(error)}
    </p>
  )
}

/** Without a worker, queued work just sits there. Say so, rather than leaving it to look stuck. */
export function WorkerWarning() {
  const { data } = useApi('monitoring/health/', { pollWhile: (health) => health && !health.worker })
  if (!data || data.worker) return null
  return (
    <p className="error-note" role="alert">
      The background worker is not running, so uploaded documents and runs are waiting. They continue when it starts.
      Start it with scripts/dev.sh or scripts/worker.sh.
    </p>
  )
}

export function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  )
}

/**
 * Picks which run a results page shows, keeping the choice in the URL so a view can be linked.
 * `accepts` limits the list, so the Security page offers only runs that have security results.
 */
export function useRunChoice(accepts = () => true, order = null) {
  const [params, setParams] = useSearchParams()
  const { data, error, loading } = useApi('evaluation/runs/?status=completed&limit=100')
  const runs = results(data).filter(accepts)
  if (order) runs.sort(order)
  const chosen = Number(params.get('run'))
  const run = runs.find((candidate) => candidate.id === chosen) ?? runs[0] ?? null
  const picker = runs.length ? (
    <label className="run-picker">
      <span className="field-label">Run</span>
      <select value={run?.id ?? ''} onChange={(event) => setParams({ run: event.target.value })}>
        {runs.map((candidate) => (
          <option key={candidate.id} value={candidate.id}>
            {runLabel(candidate)} · {candidate.rag_config_name} · {dateTime(candidate.finished_at)}
          </option>
        ))}
      </select>
    </label>
  ) : null
  return { run, picker, error, loading }
}

export function NoRuns({ loading, what = 'a completed run' }) {
  if (loading) return <Empty title="Loading runs" />
  return (
    <Empty
      title="Nothing measured yet"
      action={
        <Link className="button" to="/runs">
          Start a run
        </Link>
      }
    >
      This page shows results from {what}.
    </Empty>
  )
}

export const overall = (run) => score(run?.summary?.overall_quality)

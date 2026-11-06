import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api, results, useApi } from '../api.js'
import { Empty, ErrorNote, Field, JobStatus, Page, Panel, WorkerWarning, overall } from '../components/common.jsx'
import { useToast } from '../components/toast.jsx'
import { dateTime, duration, runLabel } from '../format.js'

const ACTIVE = ['pending', 'running']
const STAGE_TEXT = {
  indexing: 'Building the index',
  answering: 'Answering questions',
  security: 'Running attacks',
  ragas: 'Scoring with Ragas',
  deepeval: 'Scoring with DeepEval',
  scoring: 'Calculating metrics',
}
const TYPES = { quality: 'Quality', security: 'Security', full: 'Quality and security' }

export default function Runs({ canEdit }) {
  const runs = useApi('evaluation/runs/?limit=100', {
    pollWhile: (data) => results(data).some((run) => ACTIVE.includes(run.status)),
  })
  const [problem, setProblem] = useState(null)
  const toast = useToast()

  async function remove(run) {
    if (!window.confirm(`Delete ${runLabel(run)} and all its scores?`)) return
    setProblem(null)
    try {
      await api(`evaluation/runs/${run.id}/`, { method: 'DELETE' })
      toast(`Deleted ${runLabel(run)}`)
      await runs.reload()
    } catch (failure) {
      setProblem(failure)
    }
  }

  const list = results(runs.data)
  return (
    <Page title="Runs" intro="A run sends every question through one configuration, then scores the results. It keeps a record of the exact settings used.">
      <ErrorNote error={runs.error ?? problem} />
      <WorkerWarning />
      {canEdit && <StartRun onStarted={runs.reload} />}
      <Panel title="All runs">
        {!list.length && !runs.loading && <Empty title="No runs yet">Start one above to get your first scores.</Empty>}
        {list.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Configuration</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th className="num">Quality</th>
                  <th className="num">Took</th>
                  <th>Finished</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {list.map((run) => (
                  <tr key={run.id}>
                    <td>
                      {run.status === 'completed' ? <Link to={`/answer?run=${run.id}`}>{runLabel(run)}</Link> : runLabel(run)}
                      {run.error && <div className="cell-error">{run.error}</div>}
                    </td>
                    <td>{run.rag_config_name}</td>
                    <td>{TYPES[run.run_type]}</td>
                    <td>
                      <JobStatus status={run.status} />
                      {run.status === 'running' && (
                        <div className="cell-note">
                          {STAGE_TEXT[run.progress_stage] ?? 'Starting'}
                          {run.progress_stage === 'answering' && ` · ${run.completed_cases + run.failed_cases} of ${run.total_cases}`}
                        </div>
                      )}
                      {run.status === 'running' && run.progress_stage === 'answering' && run.total_cases > 0 && (
                        <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={run.total_cases} aria-valuenow={run.completed_cases + run.failed_cases}>
                          <span style={{ width: `${((run.completed_cases + run.failed_cases) / run.total_cases) * 100}%` }} />
                        </div>
                      )}
                      {run.failed_cases > 0 && <div className="cell-note">{run.failed_cases} questions failed</div>}
                    </td>
                    <td className="num">{overall(run)}</td>
                    <td className="num">
                      {run.started_at && run.finished_at ? duration(new Date(run.finished_at) - new Date(run.started_at)) : '–'}
                    </td>
                    <td className="nowrap">{dateTime(run.finished_at)}</td>
                    <td className="row-actions">
                      {canEdit && !ACTIVE.includes(run.status) && (
                        <button type="button" className="link link-danger" onClick={() => remove(run)}>
                          Delete
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </Page>
  )
}

function StartRun({ onStarted }) {
  const datasets = useApi('evaluation/datasets/?limit=100')
  const configs = useApi('rag/configs/')
  const [form, setForm] = useState({ name: '', run_type: 'quality', dataset: '', rag_config: '', k: 5, ragas: false, deepeval: false })
  const [problem, setProblem] = useState(null)
  const [busy, setBusy] = useState(false)
  const toast = useToast()
  const set = (key, value) => setForm((previous) => ({ ...previous, [key]: value }))

  const datasetId = form.dataset || results(datasets.data)[0]?.id
  const configId = form.rag_config || results(configs.data)[0]?.id
  const needsDataset = form.run_type !== 'security'

  async function start(event) {
    event.preventDefault()
    setBusy(true)
    setProblem(null)
    try {
      await api('evaluation/runs/', {
        method: 'POST',
        body: {
          name: form.name,
          run_type: form.run_type,
          rag_config: Number(configId),
          ...(needsDataset ? { dataset: Number(datasetId) } : {}),
          k: Number(form.k),
          frameworks: [form.ragas && 'ragas', form.deepeval && 'deepeval'].filter(Boolean),
        },
      })
      toast('Run started. Its progress appears below.')
      set('name', '')
      await onStarted()
    } catch (failure) {
      setProblem(failure)
    }
    setBusy(false)
  }

  return (
    <Panel title="Start a run">
      <form className="form-row" onSubmit={start}>
        <Field label="Name">
          <input value={form.name} onChange={(e) => set('name', e.target.value)} maxLength={255} placeholder="Optional" />
        </Field>
        <Field label="What to measure">
          <select value={form.run_type} onChange={(e) => set('run_type', e.target.value)}>
            {Object.entries(TYPES).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Configuration">
          <select value={configId ?? ''} onChange={(e) => set('rag_config', e.target.value)} required>
            {results(configs.data).map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
        </Field>
        {needsDataset && (
          <Field label="Dataset">
            <select value={datasetId ?? ''} onChange={(e) => set('dataset', e.target.value)} required>
              {results(datasets.data).map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} v{item.version}
                </option>
              ))}
            </select>
          </Field>
        )}
        {needsDataset && (
          <Field label="K">
            <input type="number" min={1} max={50} value={form.k} onChange={(e) => set('k', e.target.value)} className="input-short" />
          </Field>
        )}
        {needsDataset && (
          <fieldset className="checks">
            <legend className="field-label">Also score with</legend>
            <label>
              <input type="checkbox" checked={form.ragas} onChange={(e) => set('ragas', e.target.checked)} /> Ragas
            </label>
            <label>
              <input type="checkbox" checked={form.deepeval} onChange={(e) => set('deepeval', e.target.checked)} /> DeepEval
            </label>
          </fieldset>
        )}
        <button className="button button-primary" disabled={busy}>
          {busy ? 'Starting' : 'Start run'}
        </button>
      </form>
      <p className="panel-note">Ragas and DeepEval judge with the LLM, so they add several minutes per question on this machine.</p>
      <ErrorNote error={problem ?? datasets.error ?? configs.error} />
    </Panel>
  )
}

import { Database, Dot, FileText, Layers, Play } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useApi } from '../api.js'
import { Empty, ErrorNote, Page, Panel, Status } from '../components/common.jsx'
import { Rule, RuleScale } from '../components/Rule.jsx'
import { dateTime, duration, metricValue, percent, score } from '../format.js'

// The metric that best answers each stage's question, with where to look for the rest.
const STAGES = [
  { to: '/retrieval', title: 'Does it find the right text?', metric: 'recall_at_k' },
  { to: '/context', title: 'Is the context good?', metric: 'context_recall' },
  { to: '/answer', title: 'Is the answer correct?', metric: 'answer_correctness' },
  { to: '/hallucination', title: 'Does it stay within the evidence?', metric: 'faithfulness' },
  { to: '/hallucination', title: 'Does it admit when it does not know?', metric: 'abstention_accuracy' },
  { to: '/security', title: 'Do attacks succeed?', metric: 'prompt_injection_success_rate' },
]

function findings(metrics) {
  const value = (name) => metrics[name]?.mean
  const lines = []
  if (value('faithfulness') !== undefined) lines.push(`${percent(value('faithfulness'))} of answer claims were backed by the retrieved text.`)
  if (value('abstention_accuracy') !== undefined) lines.push(`It handled ${percent(value('abstention_accuracy'))} of questions correctly when deciding whether to answer at all.`)
  if (value('prompt_injection_success_rate') !== undefined) lines.push(`${percent(value('prompt_injection_success_rate'))} of prompt injections succeeded.`)
  if (value('access_control_violation_rate') !== undefined) lines.push(`${percent(value('access_control_violation_rate'))} of authorization tests reached a forbidden document.`)
  if (value('e2e_latency') !== undefined) lines.push(`95% of answers arrived within ${duration(value('e2e_latency'))}.`)
  return lines
}

function Health() {
  const { data } = useApi('monitoring/health/')
  if (!data) return null
  const services = [
    ['Background worker', data.worker],
    ['PostgreSQL', data.postgresql],
    ['Redis', data.redis],
    ['MinIO', data.minio],
    ['Qdrant', data.qdrant],
    ['Ollama', data.ollama],
    [`BM25 (${data.bm25.backend === 'local' ? 'local fallback' : 'OpenSearch'})`, data.bm25.ok],
  ]
  return (
    <div className="health" aria-label="Service status">
      {services.map(([name, ok]) => (
        <Status key={name} kind={ok ? 'good' : 'critical'}>
          {name}
        </Status>
      ))}
      <Status kind={data.langfuse.configured ? 'good' : 'neutral'}>
        {data.langfuse.configured ? 'Langfuse' : 'Langfuse off'}
      </Status>
    </div>
  )
}

function Tile({ to, Icon, value, label, tint }) {
  return (
    <Link className="tile" to={to}>
      <span className={`tile-icon tile-icon-${tint}`}>
        <Icon size={16} aria-hidden="true" />
      </span>
      <span>
        <div className="tile-value">{value}</div>
        <div className="tile-label">{label}</div>
      </span>
    </Link>
  )
}

export default function Overview() {
  const { data, error, loading } = useApi('reports/summary/')
  const latest = data?.latest_run
  // A quality-only run has no security scores, so those come from the latest run that has them.
  const securityRun = data?.latest_security_run
  const metrics = { ...securityRun?.metrics, ...latest?.metrics }

  return (
    <Page title="Overview" intro="The latest measured result for this RAG pipeline, and whether everything it needs is running.">
      <ErrorNote error={error} />
      <Health />
      {data && (
        <div className="tiles">
          <Tile to="/documents" Icon={FileText} value={data.documents.total} label="Documents" tint="blue" />
          <Tile to="/documents" Icon={Layers} value={data.chunks.toLocaleString()} label="Indexed chunks" tint="violet" />
          <Tile to="/datasets" Icon={Database} value={data.datasets} label="Datasets" tint="emerald" />
          <Tile to="/runs" Icon={Play} value={data.runs.total} label="Evaluation runs" tint="amber" />
        </div>
      )}
      {data && !latest && (
        <Empty
          title="Nothing measured yet"
          action={
            <Link className="button button-primary" to="/runs">
              Start a run
            </Link>
          }
        >
          Run a dataset through a configuration to see scores here. {data.documents.total} documents and {data.datasets} datasets are ready.
        </Empty>
      )}
      {latest && (
        <>
          <section className="verdict">
            <div className="verdict-figure">
              <div className="verdict-number">{score(latest.overall_quality)}</div>
              <div className="verdict-caption">overall quality, out of 1</div>
            </div>
            <div className="verdict-text">
              <h2>{latest.rag_config}</h2>
              <p className="verdict-meta">
                {latest.name} · {latest.dataset ?? 'Security suite'} · {dateTime(latest.finished_at)}
              </p>
              <ul>
                {findings(metrics).map((line) => (
                  <li key={line}>
                    <Dot size={16} strokeWidth={4} aria-hidden="true" />
                    {line}
                  </li>
                ))}
              </ul>
            </div>
          </section>
          <Panel title="The questions that matter" note="Follow a row to see every metric behind it.">
            <div className="stage-list">
              <div className="stage-row stage-row-scale" aria-hidden="true">
                <span />
                <RuleScale format={(v) => v.toFixed(2).replace(/0$/, '').replace(/\.0$/, '')} />
                <span />
              </div>
              {STAGES.map((stage) => {
                const metric = metrics[stage.metric]
                const fromSecurityRun = stage.to === '/security' && securityRun && securityRun.id !== latest.id
                const runId = fromSecurityRun ? securityRun.id : latest.id
                return (
                  <Link className="stage-row" to={`${stage.to}?run=${runId}`} key={stage.metric}>
                    <div className="metric-name">
                      <div className="metric-title">{stage.title}</div>
                      <div className="metric-formula">
                        {metric ? metric.display_name : 'Not scored in this run'}
                        {metric && !metric.higher_is_better ? ' · lower is better' : ''}
                        {metric && fromSecurityRun ? ` · from ${securityRun.name}` : ''}
                      </div>
                    </div>
                    <Rule value={metric?.mean} label={`${stage.title} ${metricValue(metric, metric?.mean)}`} />
                    <div className="metric-value">{metric ? metricValue(metric, metric.mean) : '–'}</div>
                  </Link>
                )
              })}
            </div>
          </Panel>
        </>
      )}
      {data?.recent_runs?.length > 1 && (
        <Panel title="Recent runs">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Configuration</th>
                  <th className="num">Metrics</th>
                  <th className="num">Quality</th>
                  <th>Finished</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_runs.map((run) => (
                  <tr key={run.id}>
                    <td>
                      <Link className="link" to={`/answer?run=${run.id}`}>
                        {run.name}
                      </Link>
                    </td>
                    <td>{run.rag_config}</td>
                    <td className="num">{run.metrics_scored} of 37</td>
                    <td className="num">{score(run.overall_quality)}</td>
                    <td className="nowrap">{dateTime(run.finished_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
      {loading && !data && <Empty title="Loading" />}
    </Page>
  )
}

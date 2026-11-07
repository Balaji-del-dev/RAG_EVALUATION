import { results, useApi } from '../api.js'
import { Empty, ErrorNote, Page, Panel, Stat, Status, useRunChoice } from '../components/common.jsx'
import { Rule, RuleScale } from '../components/Rule.jsx'
import { duration, metricValue, percent } from '../format.js'

const sloValue = (row) =>
  row.unit === 'ms' ? duration(row.value) : row.unit === 'share' ? percent(row.value) : `${Number(row.value).toFixed(1)} tokens/s`
const sloTarget = (row) => {
  const value = row.unit === 'ms' ? duration(row.target) : row.unit === 'share' ? percent(row.target) : `${row.target} tokens/s`
  return row.kind === 'max' ? `At most ${value}` : `At least ${value}`
}

/** Each objective against its target. Status words carry the verdict, so colour is never the only signal. */
function SloTable({ slo }) {
  return (
    <>
      <p className="panel-note">
        {slo.judged === 0 ? 'Not enough answers yet to judge.' : `${slo.met} of ${slo.judged} objectives met.`} Cache hit rate:{' '}
        {percent(slo.cache_hit_rate)}. Cost per question: {slo.cost_per_query ? slo.cost_per_query.toFixed(4) : '0 (local model)'}.
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Objective</th>
              <th>Target</th>
              <th className="num">Measured</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {slo.objectives.map((row) => (
              <tr key={row.key}>
                <td>{row.label}</td>
                <td>{sloTarget(row)}</td>
                <td className="num">{row.met === null ? '–' : sloValue(row)}</td>
                <td>
                  {row.met === null ? (
                    <Status kind="neutral">No data</Status>
                  ) : row.met ? (
                    <Status kind="good">Met</Status>
                  ) : (
                    <Status kind="critical">Missed</Status>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

/** p50, p95 and p99 of one latency on its own scale. Each is a separate chart: retrieval and end-to-end
 * differ by an order of magnitude, and a shared axis would crush one of them. */
function Percentiles({ title, metric, details }) {
  if (!metric || !details) return null
  const rows = [
    ['p50', 'Typical', details.p50],
    ['p95', '95% finish within', details.p95],
    ['p99', 'Slowest', details.p99],
  ]
  const max = Math.max(details.p99, 1) * 1.1
  return (
    <div className="percentile-chart">
      <h3>{title}</h3>
      <div className="percentile-row percentile-row-scale" aria-hidden="true">
        <span />
        <RuleScale max={max} format={duration} />
        <span />
      </div>
      {rows.map(([name, caption, value]) => (
        <div className="percentile-row" key={name}>
          <div className="metric-name">
            <div className="metric-title">{name}</div>
            <div className="metric-formula">{caption}</div>
          </div>
          <Rule value={value} max={max} label={`${title} ${name}: ${duration(value)}`} />
          <div className="metric-value">{duration(value)}</div>
        </div>
      ))}
    </div>
  )
}

export default function Performance() {
  const { run, picker, error } = useRunChoice((candidate) => candidate.run_type !== 'security')
  const metrics = run?.summary?.metrics ?? {}
  const live = useApi('monitoring/metrics/?days=7')
  const scores = useApi(run ? `evaluation/runs/${run.id}/results/?limit=200` : null, { skip: !run })
  const latencyDetails = (name) => {
    const entries = results(scores.data)
    const values = entries.map((entry) => (name === 'retrieval_latency' ? entry.retrieval_ms : entry.total_ms)).sort((a, b) => a - b)
    if (!values.length) return null
    const at = (p) => {
      const position = (values.length - 1) * p
      const low = Math.floor(position)
      const high = Math.ceil(position)
      return values[low] + (values[high] - values[low]) * (position - low)
    }
    return { p50: at(0.5), p95: at(0.95), p99: at(0.99) }
  }

  const liveData = live.data
  return (
    <Page title="Performance" intro="How fast and how costly the pipeline is, in test runs and with real users." actions={picker}>
      <ErrorNote error={error ?? live.error} />
      {!run && <Empty title="No test run to show">Latency from a completed quality run appears here.</Empty>}
      {run && (
        <Panel title="Latency in this run" note="Questions ran one at a time, after a warm-up, so model loading is not counted.">
          <div className="percentile-grid">
            <Percentiles title="Retrieval" metric={metrics.retrieval_latency} details={latencyDetails('retrieval_latency')} />
            <Percentiles title="End to end" metric={metrics.e2e_latency} details={latencyDetails('e2e_latency')} />
          </div>
          <div className="stat-row">
            <Stat label="Tokens per question" value={metrics.token_consumption ? Math.round(metrics.token_consumption.mean) : '–'} />
            <Stat label="Cost per question" value={metrics.cost_per_query ? metricValue(metrics.cost_per_query, metrics.cost_per_query.mean) : '–'} note="Local models are priced at 0 unless a configuration sets a price" />
            <Stat label="Throughput" value={metrics.throughput ? metricValue(metrics.throughput, metrics.throughput.mean) : '–'} note="One at a time. Use Locust for concurrent load" />
            <Stat label="Error rate" value={metrics.error_rate ? percent(metrics.error_rate.mean) : '–'} />
          </div>
        </Panel>
      )}
      <Panel title="Real users, last 7 days" note="From chat and API questions. Test runs are not counted here.">
        {liveData && liveData.requests === 0 && <p className="panel-note">No questions have been asked yet. Try the Chat.</p>}
        {liveData && liveData.requests > 0 && (
          <>
            <div className="stat-row">
              <Stat label="Questions" value={liveData.requests} />
              <Stat label="p50" value={duration(liveData.p50_ms)} />
              <Stat label="p95" value={duration(liveData.p95_ms)} />
              <Stat label="p99" value={duration(liveData.p99_ms)} />
              <Stat label="Error rate" value={percent(liveData.error_rate)} />
              <Stat label="First words, p95" value={liveData.first_token_p95_ms ? duration(liveData.first_token_p95_ms) : '–'} />
              <Stat label="Writing speed" value={liveData.tokens_per_second ? `${liveData.tokens_per_second.toFixed(1)} tok/s` : '–'} />
              <Stat label="From cache" value={percent(liveData.cache_hit_rate)} note={`${liveData.cache_hits} answers`} />
              <Stat label="Total tokens" value={(liveData.input_tokens + liveData.output_tokens).toLocaleString()} />
            </div>
            {liveData.slo && (
              <>
                <h3 className="details-title">Service level objectives</h3>
                <SloTable slo={liveData.slo} />
              </>
            )}
            {liveData.by_model.length > 0 && (
              <>
              <h3 className="details-title">By model</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th className="num">Questions</th>
                      <th className="num">Input tokens</th>
                      <th className="num">Output tokens</th>
                    </tr>
                  </thead>
                  <tbody>
                    {liveData.by_model.map((row) => (
                      <tr key={row.model_name}>
                        <td className="mono">{row.model_name}</td>
                        <td className="num">{row.requests}</td>
                        <td className="num">{row.input_tokens.toLocaleString()}</td>
                        <td className="num">{row.output_tokens.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              </>
            )}
          </>
        )}
      </Panel>
    </Page>
  )
}

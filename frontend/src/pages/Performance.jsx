import { results, useApi } from '../api.js'
import { Empty, ErrorNote, Page, Panel, Stat, useRunChoice } from '../components/common.jsx'
import { Rule, RuleScale } from '../components/Rule.jsx'
import { duration, metricValue, percent } from '../format.js'

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
              <Stat label="Total tokens" value={(liveData.input_tokens + liveData.output_tokens).toLocaleString()} />
            </div>
            {liveData.by_model.length > 0 && (
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
            )}
          </>
        )}
      </Panel>
    </Page>
  )
}

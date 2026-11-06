import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { results, useApi } from '../api.js'
import { Empty, ErrorNote, Page, Panel, Status } from '../components/common.jsx'
import { Rule, RuleScale } from '../components/Rule.jsx'
import { dateTime, metricValue, runLabel, score } from '../format.js'

// Validated all-pairs for colour blindness on the faceplate surface, which allows three at a time.
const RUN_COLORS = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)']
const MAX_RUNS = 3
const CATEGORY_NAMES = {
  retrieval: 'Retrieval',
  context: 'Context',
  answer: 'Answer',
  reliability: 'Hallucination',
  security: 'Security',
  performance: 'Performance',
}
const FRAMEWORK_NAMES = { ragas: 'Ragas', deepeval: 'DeepEval' }

export default function Compare() {
  const [params, setParams] = useSearchParams()
  const runs = useApi('evaluation/runs/?status=completed&limit=100')
  const chosen = useMemo(() => (params.get('runs') ?? '').split(',').map(Number).filter(Boolean), [params])
  const comparison = useApi(`reports/compare/?runs=${chosen.join(',')}`, { skip: chosen.length < 2 })

  function toggle(id) {
    const next = chosen.includes(id) ? chosen.filter((item) => item !== id) : [...chosen, id]
    setParams(next.length ? { runs: next.join(',') } : {})
  }

  const options = results(runs.data)
  const data = chosen.length >= 2 ? comparison.data : null
  return (
    <Page title="Compare" intro="Put runs side by side. Better quality usually costs latency, so the answer is a trade-off to decide, not a winner to read off.">
      <ErrorNote error={runs.error ?? comparison.error} />
      <Panel title="Choose two or three runs">
        {!options.length && !runs.loading && <Empty title="No completed runs yet">Finish at least two runs to compare them.</Empty>}
        <div className="run-chooser">
          {options.map((run) => {
            const position = chosen.indexOf(run.id)
            const selected = position >= 0
            return (
              <label className={`run-chip ${selected ? 'run-chip-on' : ''}`} key={run.id}>
                <input
                  type="checkbox"
                  checked={selected}
                  disabled={!selected && chosen.length >= MAX_RUNS}
                  onChange={() => toggle(run.id)}
                />
                {selected && <span className="legend-dot" style={{ background: RUN_COLORS[position] }} aria-hidden="true" />}
                <span>
                  <span className="run-chip-name">{runLabel(run)}</span>
                  <span className="run-chip-meta">
                    {run.rag_config_name} · {run.run_type} · {dateTime(run.finished_at)}
                  </span>
                </span>
              </label>
            )
          })}
        </div>
      </Panel>
      {data && <Results data={data} />}
    </Page>
  )
}

function Results({ data }) {
  const color = (id) => RUN_COLORS[data.runs.findIndex((run) => run.id === id)]
  const name = (id) => data.runs.find((run) => run.id === id)?.name
  const rows = data.metrics.filter((row) => Object.keys(row.values).length > 0)
  const categories = Object.keys(CATEGORY_NAMES).filter((category) => rows.some((row) => row.category === category))

  return (
    <>
      <Panel title="What differs between these runs">
        {data.differing_settings.length === 0 ? (
          <p className="panel-note">These runs used the same settings, so any difference comes from randomness or from changes in the documents.</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Setting</th>
                  {data.runs.map((run) => (
                    <th key={run.id}>{run.name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.differing_settings.map((row) => (
                  <tr key={row.setting}>
                    <td className="mono">{row.setting}</td>
                    {data.runs.map((run) => (
                      <td className="mono" key={run.id}>
                        {row.values[run.id] || '(none)'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
      <div className="legend legend-sticky" aria-label="Runs">
        {data.runs.map((run) => (
          <span className="legend-item" key={run.id}>
            <span className="legend-dot" style={{ background: color(run.id) }} aria-hidden="true" />
            {run.name} · {run.rag_config}
            {run.overall_quality !== null && run.overall_quality !== undefined && ` · quality ${score(run.overall_quality)}`}
          </span>
        ))}
      </div>
      {categories.map((category) => (
        <Panel
          title={CATEGORY_NAMES[category]}
          note={category === 'performance' ? 'Each row has its own scale, up to its largest value.' : undefined}
          key={category}
        >
          <CategoryRows rows={rows.filter((row) => row.category === category)} runs={data.runs} color={color} name={name} />
        </Panel>
      ))}
    </>
  )
}

function CategoryRows({ rows, runs, color, name }) {
  // Scores share a 0 to 1 scale. If any row uses another unit, each row has its own scale instead.
  const sharedScale = rows.every((row) => !row.unit)
  return (
    <div className="compare-list" style={{ '--runs': runs.length }}>
      <div className="compare-row compare-row-scale" aria-hidden="true">
        <span />
        {sharedScale ? <RuleScale format={(v) => v.toFixed(2).replace(/0$/, '').replace(/\.0$/, '')} /> : <span />}
        {runs.map((run) => (
          <span className="compare-head" key={run.id}>
            {run.name}
          </span>
        ))}
      </div>
      {rows.map((row) => {
        // Rates and scores share a 0–1 rule. Latency, tokens and cost get a rule up to the largest value.
        const max = row.unit ? Math.max(...Object.values(row.values), 1e-9) * 1.1 : 1
        const marks = runs
          .filter((run) => row.values[run.id] !== undefined)
          .map((run) => ({
            key: run.id,
            value: row.values[run.id],
            color: color(run.id),
            name: name(run.id),
            display: metricValue(row, row.values[run.id]),
          }))
        return (
          <div className="compare-row" key={row.key}>
            <div className="metric-name">
              <div className="metric-title">
                {row.display_name}
                {row.framework !== 'builtin' && <span className="metric-framework"> {FRAMEWORK_NAMES[row.framework] ?? row.framework}</span>}
              </div>
              <div className="metric-formula">{row.higher_is_better ? 'higher is better' : 'lower is better'}</div>
            </div>
            <Rule marks={marks} max={max} label={marks.map((mark) => `${mark.name} ${mark.display}`).join(', ')} />
            {runs.map((run) => {
              const isBest = row.best_run === run.id
              return (
                <div className={`compare-value ${isBest ? 'compare-best' : ''}`} key={run.id}>
                  {metricValue(row, row.values[run.id])}
                  {isBest && <Status kind="good">Best</Status>}
                </div>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}

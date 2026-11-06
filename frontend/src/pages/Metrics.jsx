import { Link } from 'react-router-dom'
import { useApi } from '../api.js'
import { ErrorNote, NoRuns, Page, Panel, Priority, useRunChoice } from '../components/common.jsx'
import { Rule } from '../components/Rule.jsx'
import { metricValue } from '../format.js'
import { METRIC_GUIDE } from '../metricGuide.js'

const STAGES = [
  { category: 'retrieval', title: 'Stage 1: Retrieval', goal: 'Does the retriever find the correct documents?', to: '/retrieval' },
  { category: 'context', title: 'Stage 2: Context', goal: 'Is the retrieved context relevant and sufficient for the LLM?', to: '/context' },
  { category: 'answer', title: 'Stage 3: Answer generation', goal: 'Does the LLM generate a correct, relevant and grounded answer?', to: '/answer' },
  { category: 'reliability', title: 'Stage 4: Hallucination and reliability', goal: 'Which answers are unsupported or incorrect?', to: '/hallucination' },
  { category: 'security', title: 'Stage 5: Security and guardrails', goal: 'Does it expose sensitive information or follow malicious instructions?', to: '/security' },
  { category: 'performance', title: 'Stage 6: Performance and cost', goal: 'How efficient is the pipeline in production?', to: '/performance' },
]

// A full run is the only kind that scores all 37, so offer those first.
const fullRunsFirst = (a, b) => (b.run_type === 'full') - (a.run_type === 'full') || b.id - a.id

export default function Metrics() {
  const { run, picker, error, loading } = useRunChoice(() => true, fullRunsFirst)
  const definitions = useApi('evaluation/metrics/')
  const scores = run?.summary?.metrics ?? {}
  const rows = definitions.data ?? []
  const scored = rows.filter((row) => scores[row.name]).length

  return (
    <Page
      title="All 37 metrics"
      intro="The complete roadmap in learning order: what each metric tells you, how it is calculated, and what this pipeline scored."
      actions={picker}
    >
      <ErrorNote error={error ?? definitions.error} />
      {!run && <NoRuns loading={loading} />}
      {run && (
        <p className="roadmap-count">
          {scored} of {rows.length} scored in this run.
          {scored < rows.length && ' Only a run that measures quality and security scores all 37.'}
        </p>
      )}
      {STAGES.map((stage) => (
        <Panel title={stage.title} note={stage.goal} key={stage.category}>
          <div className="roadmap-head" aria-hidden="true">
            <span>#</span>
            <span>Metric</span>
            <span>What you need to understand</span>
            <span>Score, 0 to 1 where it applies</span>
            <span />
          </div>
          {rows
            .filter((row) => row.category === stage.category)
            .map((row) => {
              const score = scores[row.name]
              const reading = score ? metricValue(score, score.mean) : '–'
              const guide = METRIC_GUIDE[row.name]
              return (
                <div className="roadmap-entry" key={row.name}>
                  <Link className="roadmap-row" to={`${stage.to}${run ? `?run=${run.id}` : ''}`}>
                    <span className="roadmap-number">{row.order}</span>
                    <span>
                      <span className="metric-title">{row.display_name}</span> <Priority value={row.priority} />
                      <div className="metric-direction">{row.higher_is_better ? 'higher is better' : 'lower is better'}</div>
                    </span>
                    <span>
                      <div className="metric-question">{row.question}</div>
                      <div className="metric-formula">{row.formula}</div>
                    </span>
                    {row.unit ? <span /> : <Rule value={score?.mean} label={`${row.display_name}: ${reading}`} />}
                    <span className="roadmap-score">{reading}</span>
                  </Link>
                  {guide && (
                    <details className="roadmap-guide">
                      <summary>How it works</summary>
                      <dl>
                        <div>
                          <dt>Technical working</dt>
                          <dd>{guide.working}</dd>
                        </div>
                        <div>
                          <dt>Detection</dt>
                          <dd>{guide.detection}</dd>
                        </div>
                        <div>
                          <dt>When to use it</dt>
                          <dd>{guide.scenario}</dd>
                        </div>
                        <div>
                          <dt>Common tools</dt>
                          <dd>{guide.tools}</dd>
                        </div>
                        {guide.here && (
                          <div>
                            <dt>On this platform</dt>
                            <dd>{guide.here}</dd>
                          </div>
                        )}
                      </dl>
                    </details>
                  )}
                </div>
              )
            })}
        </Panel>
      ))}
    </Page>
  )
}

import { metricValue } from '../format.js'
import { Priority } from './common.jsx'
import { Rule, RuleScale } from './Rule.jsx'

const FRAMEWORK_NAMES = { ragas: 'Ragas', deepeval: 'DeepEval' }

/** All metrics of one category for one run, each on its own rule. Formulas stay visible: they are the point. */
export function MetricList({ summary, definitions, category }) {
  const metrics = Object.entries(summary?.metrics ?? {}).map(([key, metric]) => ({ key, ...metric }))
  const builtin = metrics
    .filter((metric) => metric.category === category && metric.framework === 'builtin')
    .sort((a, b) => a.order - b.order)
  const scored = new Set(builtin.map((metric) => metric.metric))
  const missing = definitions.filter((definition) => definition.category === category && !scored.has(definition.name))
  const hasUnits = builtin.some((metric) => metric.unit)

  return (
    <div className="metric-list">
      {!hasUnits && (
        <div className="metric-row metric-row-scale" aria-hidden="true">
          <span />
          <RuleScale format={(v) => v.toFixed(2).replace(/0$/, '').replace(/\.0$/, '')} />
          <span />
        </div>
      )}
      {builtin.map((metric) => {
        const definition = definitions.find((item) => item.name === metric.metric)
        const others = metrics.filter((item) => item.metric === metric.metric && item.framework !== 'builtin')
        return (
          <div className="metric-row" key={metric.key}>
            <div className="metric-name">
              <div className="metric-title">
                {metric.display_name} <Priority value={metric.priority} />
              </div>
              <div className="metric-question">{definition?.question}</div>
              <div className="metric-formula">{definition?.formula}</div>
            </div>
            {metric.unit ? (
              <div className="metric-spacer" />
            ) : (
              <Rule value={metric.mean} label={`${metric.display_name}: ${metricValue(metric, metric.mean)}`} />
            )}
            <div className="metric-reading">
              <div className="metric-value">{metricValue(metric, metric.mean)}</div>
              <div className="metric-direction">{metric.higher_is_better ? 'higher is better' : 'lower is better'}</div>
              {metric.count > 1 && <div className="metric-direction">{metric.count} questions</div>}
              {others.map((other) => (
                <div className="metric-other" key={other.key}>
                  {FRAMEWORK_NAMES[other.framework] ?? other.framework} {metricValue(other, other.mean)}
                </div>
              ))}
            </div>
          </div>
        )
      })}
      {missing.map((definition) => (
        <div className="metric-row metric-row-missing" key={definition.name}>
          <div className="metric-name">
            <div className="metric-title">
              {definition.display_name} <Priority value={definition.priority} />
            </div>
            <div className="metric-question">{definition.question}</div>
            <div className="metric-formula">{definition.formula}</div>
          </div>
          <div className="metric-spacer" />
          <div className="metric-reading">
            <div className="metric-direction">Not scored in this run</div>
          </div>
        </div>
      ))}
    </div>
  )
}

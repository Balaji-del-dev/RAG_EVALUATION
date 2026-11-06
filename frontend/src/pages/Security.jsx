import { useState } from 'react'
import { results, useApi } from '../api.js'
import { ErrorNote, NoRuns, Page, Panel, Status, useRunChoice } from '../components/common.jsx'
import { MetricList } from '../components/MetricList.jsx'
import { duration } from '../format.js'

const ATTACKS = {
  prompt_injection: 'Prompt injection',
  jailbreak: 'Jailbreak',
  pii: 'PII extraction',
  sensitive_data: 'Secrets',
  access_control: 'Access control',
  toxicity: 'Toxic output',
  benign: 'Legitimate questions',
}

const hasSecurity = (run) => run.run_type !== 'quality'

export default function Security({ canEdit }) {
  const { run, picker, error, loading } = useRunChoice(hasSecurity)
  const definitions = useApi('evaluation/metrics/')
  return (
    <Page
      title="Security"
      intro="Real attacks replayed through the live pipeline as a user with the least access. Lower rates are better."
      actions={picker}
    >
      <ErrorNote error={error ?? definitions.error} />
      {!run && <NoRuns loading={loading} what="a completed security run" />}
      {run && (
        <>
          {canEdit && <TestBreakdown run={run} />}
          <Panel title="Security metrics">
            <MetricList summary={run.summary} definitions={definitions.data ?? []} category="security" />
          </Panel>
          {!canEdit && (
            <Panel title="Test detail">
              <p className="panel-note">Attack prompts and answers are visible to analysts and admins.</p>
            </Panel>
          )}
        </>
      )}
    </Page>
  )
}

function TestBreakdown({ run }) {
  const { data, error } = useApi(`security/results/?run=${run.id}&limit=200`)
  const [open, setOpen] = useState(null)
  const tests = results(data)
  const failing = tests.filter((test) => !test.passed).length

  const byType = Object.keys(ATTACKS)
    .map((type) => {
      const group = tests.filter((test) => test.attack_type === type)
      return { type, total: group.length, passed: group.filter((test) => test.passed).length }
    })
    .filter((group) => group.total)

  const matrix = { tp: 0, fp: 0, tn: 0, fn: 0 }
  for (const test of tests) if (test.details?.counted_as) matrix[test.details.counted_as] += 1

  return (
    <>
      <ErrorNote error={error} />
      <Panel title={tests.length ? `${tests.length - failing} of ${tests.length} tests passed` : 'Tests'}>
        <div className="breakdown">
          {byType.map((group) => (
            <div className="breakdown-row" key={group.type}>
              <div className="breakdown-name">{ATTACKS[group.type]}</div>
              <div
                className="segments"
                role="img"
                aria-label={`${ATTACKS[group.type]}: ${group.passed} of ${group.total} passed`}
              >
                {group.passed > 0 && <span className="segment segment-pass" style={{ flexGrow: group.passed }} />}
                {group.total - group.passed > 0 && <span className="segment segment-fail" style={{ flexGrow: group.total - group.passed }} />}
              </div>
              <div className="breakdown-count">
                {group.passed} of {group.total} passed
              </div>
            </div>
          ))}
        </div>
        <div className="legend" aria-hidden="true">
          <span className="legend-item">
            <span className="legend-swatch segment-pass" /> Passed
          </span>
          <span className="legend-item">
            <span className="legend-swatch segment-fail" /> Failed
          </span>
        </div>
      </Panel>

      <Panel title="How the guards decided" note="Positive means the guard should have acted. Access control is excluded: it is enforced before retrieval, not by these guards.">
        <div className="table-wrap">
          <table className="matrix">
            <thead>
              <tr>
                <th />
                <th className="num">Guard acted</th>
                <th className="num">Guard did nothing</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">Should have acted</th>
                <td className="num">{matrix.tp} caught</td>
                <td className="num">{matrix.fn} missed</td>
              </tr>
              <tr>
                <th scope="row">Should not have acted</th>
                <td className="num">{matrix.fp} false alarms</td>
                <td className="num">{matrix.tn} correct</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Every test">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Test</th>
                <th>Type</th>
                <th>Severity</th>
                <th>Result</th>
                <th>Guard</th>
                <th className="num">Took</th>
              </tr>
            </thead>
            <tbody>
              {tests.map((test) => (
                <TestRows key={test.id} test={test} open={open === test.id} onToggle={() => setOpen(open === test.id ? null : test.id)} />
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  )
}

function TestRows({ test, open, onToggle }) {
  const reason = test.details?.block_reason
  const findings = test.details?.findings?.output
  return (
    <>
      <tr>
        <td>
          <button type="button" className="link" onClick={onToggle} aria-expanded={open}>
            {test.test_name.replaceAll('_', ' ')}
          </button>
        </td>
        <td className="nowrap">{ATTACKS[test.attack_type]}</td>
        <td>{test.severity}</td>
        <td>{test.passed ? <Status kind="good">Passed</Status> : <Status kind="critical">Failed</Status>}</td>
        <td>{test.guard_blocked ? (reason ? 'Blocked' : 'Masked') : 'No action'}</td>
        <td className="num">{duration(test.latency_ms)}</td>
      </tr>
      {open && (
        <tr className="row-detail">
          <td colSpan={6}>
            <div className="detail-grid">
              <div className="detail-block">
                <h4>Attack prompt</h4>
                <p>{test.prompt}</p>
              </div>
              <div className="detail-block">
                <h4>What came back</h4>
                <p>{test.answer}</p>
              </div>
            </div>
            <dl className="readout">
              {reason && (
                <div>
                  <dt>Blocked because</dt>
                  <dd>{reason}</dd>
                </div>
              )}
              {findings && (
                <div>
                  <dt>Masked</dt>
                  <dd>{[...(findings.pii ?? []), ...(findings.secrets ?? [])].join(', ')}</dd>
                </div>
              )}
              <div>
                <dt>Chunks retrieved</dt>
                <dd>{test.details?.retrieved_chunks ?? 0}</dd>
              </div>
              <div>
                <dt>Forbidden chunks</dt>
                <dd>{test.details?.unauthorized_chunks?.length ?? 0}</dd>
              </div>
            </dl>
          </td>
        </tr>
      )}
    </>
  )
}

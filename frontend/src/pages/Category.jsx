import { useState } from 'react'
import { results, useApi } from '../api.js'
import { ErrorNote, NoRuns, Page, Panel, Status, useRunChoice } from '../components/common.jsx'
import { MetricList } from '../components/MetricList.jsx'
import { score } from '../format.js'

const COPY = {
  retrieval: {
    title: 'Retrieval',
    intro: 'Does the retriever find the correct documents? Pure ranking math, checked against hand-labelled evidence.',
  },
  context: {
    title: 'Context',
    intro: 'Is the retrieved context relevant and sufficient for the LLM, regardless of what the LLM then does with it?',
  },
  answer: {
    title: 'Answer',
    intro: 'Does the LLM generate a correct, relevant and grounded answer?',
  },
  reliability: {
    title: 'Hallucination',
    intro: 'Which answers are unsupported or incorrect? Does it make things up, cite wrongly, or answer what it cannot know?',
  },
}

const answerState = (result) => {
  if (result.blocked) return <Status kind="critical">Blocked</Status>
  if (result.abstained) return <Status kind="neutral">Declined</Status>
  return <Status kind="good">Answered</Status>
}

export default function Category({ category, canEdit }) {
  const { title, intro } = COPY[category]
  const { run, picker, error, loading } = useRunChoice((candidate) => candidate.run_type !== 'security')
  const definitions = useApi('evaluation/metrics/')
  return (
    <Page title={title} intro={intro} actions={picker}>
      <ErrorNote error={error ?? definitions.error} />
      {!run && <NoRuns loading={loading} what="a completed quality run" />}
      {run && (
        <>
          <Panel>
            <MetricList summary={run.summary} definitions={definitions.data ?? []} category={category} />
          </Panel>
          {canEdit ? (
            <Questions runId={run.id} category={category} />
          ) : (
            <Panel title="Per-question detail">
              <p className="panel-note">Answers and source text are visible to analysts and admins, because evaluation runs read restricted documents too.</p>
            </Panel>
          )}
        </>
      )}
    </Page>
  )
}

function Questions({ runId, category }) {
  const { data, error } = useApi(`evaluation/runs/${runId}/results/?limit=200`)
  const [open, setOpen] = useState(null)
  const rows = results(data)
  const columns = []
  for (const row of rows) {
    for (const item of row.scores) {
      if (item.category === category && item.framework === 'builtin' && !columns.some((c) => c.metric === item.metric)) {
        columns.push(item)
      }
    }
  }
  columns.sort((a, b) => a.order - b.order)
  return (
    <Panel title="Question by question" note="A dash means the metric does not apply, for example ranking quality on a question with no correct passage. Open a row for the evidence behind its scores.">
      <ErrorNote error={error} />
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Question</th>
              <th>Outcome</th>
              {columns.map((column) => (
                <th className="num" key={column.metric}>
                  {column.display_name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const scores = Object.fromEntries(row.scores.filter((item) => item.framework === 'builtin').map((item) => [item.metric, item]))
              return (
                <QuestionRows
                  key={row.id}
                  row={row}
                  scores={scores}
                  columns={columns}
                  outcome={answerState(row)}
                  open={open === row.id}
                  onToggle={() => setOpen(open === row.id ? null : row.id)}
                />
              )
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

function DetailList({ title, items }) {
  if (!items?.length) return null
  return (
    <div className="detail-block">
      <h4>{title}</h4>
      <ul>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  )
}

function QuestionRows({ row, scores, columns, outcome, open, onToggle }) {
  return (
    <>
      <tr>
        <td>
          <button type="button" className="link link-question" onClick={onToggle} aria-expanded={open}>
            {row.question}
          </button>
        </td>
        <td>{outcome}</td>
        {columns.map((column) => (
          <td className="num" key={column.metric}>
            {score(scores[column.metric]?.score)}
          </td>
        ))}
      </tr>
      {open && (
        <tr className="row-detail">
          <td colSpan={columns.length + 2}>
            <div className="detail-grid">
              <div className="detail-block">
                <h4>Answer</h4>
                <p>{row.answer}</p>
              </div>
              <div className="detail-block">
                <h4>Reference answer</h4>
                <p>{row.reference_answer || 'None: the correct behaviour is to decline.'}</p>
              </div>
            </div>
            <DetailList title="Claims not supported by the context" items={scores.hallucination_rate?.details?.unsupported_claims} />
            <DetailList title="Claims contradicted by the context" items={scores.contradiction_rate?.details?.contradicted_claims} />
            <DetailList title="Reference claims missing from the answer" items={scores.completeness?.details?.missing} />
            <DetailList title="Reference claims missing from the context" items={scores.context_recall?.details?.unsupported_reference_claims} />
            <div className="detail-block">
              <h4>Retrieved context</h4>
              <ol className="chunk-list">
                {row.contexts.map((context, index) => {
                  const chunkId = row.retrieved_chunk_ids[index]
                  const relevant = row.relevant_chunk_ids.includes(chunkId)
                  return (
                    <li key={chunkId}>
                      <div className="chunk-meta">
                        [S{index + 1}] chunk {chunkId}
                        {relevant ? ' · labelled relevant' : ''}
                      </div>
                      <p>{context}</p>
                    </li>
                  )
                })}
              </ol>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

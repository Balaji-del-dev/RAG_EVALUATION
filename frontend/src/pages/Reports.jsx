import { useState } from 'react'
import { results, useApi } from '../api.js'
import { Empty, ErrorNote, Page, Panel, overall } from '../components/common.jsx'
import { dateTime, runLabel } from '../format.js'

export default function Reports({ canEdit }) {
  const runs = useApi('evaluation/runs/?status=completed&limit=100')
  const [open, setOpen] = useState(null)
  const list = results(runs.data)
  return (
    <Page title="Reports" intro="Export results, and see exactly what produced them. Each run records its model, prompt, dataset and package versions, so it can be reproduced later.">
      <ErrorNote error={runs.error} />
      <Panel>
        {!list.length && !runs.loading && <Empty title="No completed runs yet">Finish a run to export its results.</Empty>}
        {list.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Configuration</th>
                  <th className="num">Quality</th>
                  <th className="num">Metrics</th>
                  <th>Finished</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {list.map((run) => (
                  <ReportRows key={run.id} run={run} canEdit={canEdit} open={open === run.id} onToggle={() => setOpen(open === run.id ? null : run.id)} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </Page>
  )
}

function ReportRows({ run, canEdit, open, onToggle }) {
  const snapshot = run.snapshot ?? {}
  const facts = [
    ['Model', snapshot.rag_config?.llm_model],
    ['Model digest', snapshot.llm_digest ? snapshot.llm_digest.slice(0, 16) : ''],
    ['Embedding', snapshot.rag_config?.embedding_model],
    ['Reranker', snapshot.rag_config?.reranker || 'None'],
    ['Chunking', `${snapshot.rag_config?.chunk_strategy}, ${snapshot.rag_config?.chunk_size} tokens, overlap ${snapshot.rag_config?.chunk_overlap}`],
    ['Retrieval', `${snapshot.rag_config?.retrieval_mode}, top ${snapshot.rag_config?.top_k}`],
    ['BM25 backend', snapshot.bm25_backend],
    ['Prompt', snapshot.prompt?.version],
    ['Dataset', snapshot.dataset ? `${snapshot.dataset.name} v${snapshot.dataset.version}` : 'Security suite only'],
    ['Judge models', snapshot.judge ? `${snapshot.judge.nli}, ${snapshot.judge.embedding}` : ''],
    ['Frameworks', Object.entries(snapshot.frameworks ?? {}).map(([name, version]) => `${name} ${version}`).join(', ') || 'Built-in only'],
    ['Packages', Object.entries(snapshot.packages ?? {}).map(([name, version]) => `${name} ${version}`).join(', ')],
  ]
  return (
    <>
      <tr>
        <td>
          <button type="button" className="link" onClick={onToggle} aria-expanded={open}>
            {runLabel(run)}
          </button>
        </td>
        <td>{run.rag_config_name}</td>
        <td className="num">{overall(run)}</td>
        <td className="num">{run.summary?.metrics_scored ?? 0} of 37</td>
        <td className="nowrap">{dateTime(run.finished_at)}</td>
        <td className="row-actions">
          {canEdit ? (
            <>
              <a className="link" href={`/api/reports/runs/${run.id}/export/?type=csv`}>
                Download CSV
              </a>
              <a className="link" href={`/api/reports/runs/${run.id}/export/?type=json`} target="_blank" rel="noreferrer">
                Open JSON
              </a>
            </>
          ) : (
            <span className="cell-note">Analysts can export</span>
          )}
        </td>
      </tr>
      {open && (
        <tr className="row-detail">
          <td colSpan={6}>
            <dl className="facts">
              {facts.map(([label, value]) => (
                <div key={label}>
                  <dt>{label}</dt>
                  <dd>{value || '–'}</dd>
                </div>
              ))}
            </dl>
          </td>
        </tr>
      )}
    </>
  )
}

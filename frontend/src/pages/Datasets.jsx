import { useRef, useState } from 'react'
import { api, results, useApi } from '../api.js'
import { Empty, ErrorNote, Field, Page, Panel, Status } from '../components/common.jsx'
import { FileDrop } from '../components/FileDrop.jsx'
import { useToast } from '../components/toast.jsx'
import { dateTime } from '../format.js'

function Cases({ datasetId }) {
  const { data, error } = useApi(`evaluation/datasets/${datasetId}/`)
  if (error) return <ErrorNote error={error} />
  if (!data) return <p className="panel-note">Loading questions</p>
  return (
    <ul className="case-list">
      {data.cases.map((item) => (
        <li key={item.id}>
          <div className="case-question">{item.question}</div>
          {item.answerable ? (
            <div className="case-reference">{item.reference_answer}</div>
          ) : (
            <Status kind="neutral">Should decline: not in the documents</Status>
          )}
          {item.evidence.length > 0 && <div className="chunk-meta">{item.evidence.length} evidence {item.evidence.length === 1 ? 'passage' : 'passages'}</div>}
        </li>
      ))}
    </ul>
  )
}

export default function Datasets({ canEdit }) {
  const list = useApi('evaluation/datasets/?limit=100')
  const [open, setOpen] = useState(null)
  const [name, setName] = useState('')
  const [version, setVersion] = useState('1')
  const [problem, setProblem] = useState(null)
  const [busy, setBusy] = useState(false)
  const [file, setFile] = useState(null)
  const fileInput = useRef(null)
  const toast = useToast()

  async function upload(event) {
    event.preventDefault()
    const form = new FormData()
    form.append('file', file)
    form.append('name', name)
    form.append('version', version)
    setBusy(true)
    setProblem(null)
    try {
      await api('evaluation/datasets/upload/', { method: 'POST', form })
      toast(`Uploaded dataset ${name}`)
      setName('')
      setFile(null)
      fileInput.current.value = ''
      await list.reload()
    } catch (failure) {
      setProblem(failure)
    }
    setBusy(false)
  }

  const datasets = results(list.data)
  return (
    <Page title="Datasets" intro="Question sets with known answers. A dataset never changes after it is created, so scores from different runs stay comparable.">
      <ErrorNote error={list.error ?? problem} />
      {canEdit && (
        <Panel
          title="Upload a dataset"
          note="JSON or CSV, with fields question, reference_answer, answerable and evidence. In CSV, separate evidence passages with ||."
        >
          <form className="form-row" onSubmit={upload}>
            <FileDrop
              inputRef={fileInput}
              accept=".json,.csv"
              hint="JSON or CSV"
              file={file}
              onFile={setFile}
              wrongTypeHint="To ask questions about a document, upload it on the Documents page."
            />
            <Field label="Name">
              <input value={name} onChange={(e) => setName(e.target.value)} required maxLength={255} />
            </Field>
            <Field label="Version">
              <input value={version} onChange={(e) => setVersion(e.target.value)} required maxLength={32} className="input-short" />
            </Field>
            <button className="button button-primary" disabled={busy || !file}>
              {busy ? 'Uploading' : 'Upload'}
            </button>
          </form>
        </Panel>
      )}
      <Panel>
        {!datasets.length && !list.loading && <Empty title="No datasets yet">Upload one, or run manage.py seed_demo for the sample set.</Empty>}
        {datasets.map((dataset) => (
          <div className="dataset" key={dataset.id}>
            <div className="dataset-head">
              <div>
                <button type="button" className="link link-strong" onClick={() => setOpen(open === dataset.id ? null : dataset.id)} aria-expanded={open === dataset.id}>
                  {dataset.name}
                </button>
                <span className="dataset-meta">
                  {' '}
                  version {dataset.version} · {dataset.case_count} questions · {dateTime(dataset.created_at)}
                </span>
                {dataset.description && <p className="panel-note">{dataset.description}</p>}
              </div>
            </div>
            {open === dataset.id && <Cases datasetId={dataset.id} />}
          </div>
        ))}
      </Panel>
    </Page>
  )
}

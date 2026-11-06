import { FileSpreadsheet, FileText, Globe, Lock, RefreshCw, Trash2, Upload } from 'lucide-react'
import { useRef, useState } from 'react'
import { api, results, useApi } from '../api.js'
import { Empty, ErrorNote, Field, JobStatus, Page, Panel, WorkerWarning } from '../components/common.jsx'
import { FileDrop } from '../components/FileDrop.jsx'
import { useToast } from '../components/toast.jsx'
import { bytes, shortDate } from '../format.js'

const BUSY = ['uploaded', 'processing']
// Matches the server: after this long a job has probably been lost, so allow a retry.
const STUCK_AFTER_MS = 15 * 60 * 1000
const stuck = (document) => Date.now() - new Date(document.updated_at) > STUCK_AFTER_MS

function Chunks({ documentId }) {
  const { data, error, loading } = useApi(`documents/${documentId}/chunks/?limit=200`)
  if (error) return <ErrorNote error={error} />
  if (loading && !data) return <p className="panel-note">Loading chunks</p>
  const chunks = results(data)
  if (!chunks.length) return <p className="panel-note">This document has no chunks yet.</p>
  return (
    <ol className="chunk-list">
      {chunks.map((chunk) => (
        <li key={chunk.id}>
          <div className="chunk-meta">
            Chunk {chunk.chunk_index + 1} · {chunk.token_count} tokens
            {chunk.page_number ? ` · page ${chunk.page_number}` : ''}
          </div>
          <p>{chunk.chunk_text}</p>
        </li>
      ))}
    </ol>
  )
}

export default function Documents({ canEdit }) {
  const list = useApi('documents/?limit=200', {
    pollWhile: (data) => results(data).some((document) => BUSY.includes(document.status)),
  })
  const [open, setOpen] = useState(null)
  const [visibility, setVisibility] = useState('public')
  const [problem, setProblem] = useState(null)
  const [busy, setBusy] = useState(false)
  const [file, setFile] = useState(null)
  const fileInput = useRef(null)
  const toast = useToast()

  async function upload(event) {
    event.preventDefault()
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    form.append('visibility', visibility)
    setBusy(true)
    setProblem(null)
    try {
      await api('documents/upload/', { method: 'POST', form })
      toast(`Uploaded ${file.name}. Processing it now.`)
      fileInput.current.value = ''
      setFile(null)
      list.reload()
    } catch (failure) {
      setProblem(failure)
    }
    setBusy(false)
  }

  async function act(action, done) {
    setProblem(null)
    try {
      await action()
      toast(done)
      await list.reload()
    } catch (failure) {
      setProblem(failure)
    }
  }

  const documents = results(list.data)
  return (
    <Page title="Documents" intro="The files the pipeline can answer from. Restricted files are visible only to their owner, shared users and admins.">
      <ErrorNote error={list.error ?? problem} />
      <WorkerWarning />
      {canEdit && (
        <Panel title="Upload a document" note="Up to 25 MB. Scanned PDF pages are read with OCR.">
          <form className="form-row" onSubmit={upload}>
            <FileDrop inputRef={fileInput} accept=".pdf,.docx,.csv,.txt,.md" hint="PDF, DOCX, CSV, TXT or MD" file={file} onFile={setFile} />
            <Field label="Who can read it">
              <select value={visibility} onChange={(e) => setVisibility(e.target.value)}>
                <option value="public">Everyone signed in</option>
                <option value="restricted">Only allowed users and groups</option>
              </select>
            </Field>
            <button className="button button-primary" disabled={busy || !file}>
              <Upload size={16} aria-hidden="true" />
              {busy ? 'Uploading' : 'Upload'}
            </button>
          </form>
        </Panel>
      )}
      <Panel title={`${documents.length} ${documents.length === 1 ? 'document' : 'documents'}`}>
        {!documents.length && !list.loading && <Empty title="No documents yet">Upload a file to start building the index.</Empty>}
        {documents.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>File</th>
                  <th>Status</th>
                  <th>Access</th>
                  <th className="num">Pages</th>
                  <th className="num">Chunks</th>
                  <th className="num">Size</th>
                  <th>Added</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {documents.map((document) => (
                  <DocumentRows
                    key={document.id}
                    document={document}
                    canEdit={canEdit}
                    open={open === document.id}
                    onToggle={() => setOpen(open === document.id ? null : document.id)}
                    onReprocess={() => act(() => api(`documents/${document.id}/process/`, { method: 'POST' }), `Queued ${document.filename} for reprocessing`)}
                    onDelete={() => {
                      if (window.confirm(`Delete ${document.filename}? It will be removed from storage and every index.`)) {
                        act(() => api(`documents/${document.id}/`, { method: 'DELETE' }), `Deleted ${document.filename}`)
                      }
                    }}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </Page>
  )
}

function DocumentRows({ document, canEdit, open, onToggle, onReprocess, onDelete }) {
  return (
    <>
      <tr>
        <td>
          <div className="file-cell">
            {document.filename.toLowerCase().endsWith('.csv') ? <FileSpreadsheet size={16} aria-hidden="true" /> : <FileText size={16} aria-hidden="true" />}
            <div>
              <button type="button" className="link" onClick={onToggle} aria-expanded={open}>
                {document.filename}
              </button>
              {document.error && <div className="cell-error">{document.error}</div>}
            </div>
          </div>
        </td>
        <td>
          <JobStatus status={document.status} />
        </td>
        <td>
          <span className="access">
            {document.visibility === 'public' ? <Globe size={16} aria-hidden="true" /> : <Lock size={16} aria-hidden="true" />}
            {document.visibility === 'public' ? 'Everyone' : 'Restricted'}
          </span>
        </td>
        <td className="num">{document.page_count}</td>
        <td className="num">{document.chunk_count}</td>
        <td className="num">{bytes(document.size_bytes)}</td>
        <td className="nowrap">{shortDate(document.created_at)}</td>
        <td className="row-actions">
          {canEdit && (
            <>
              <button
                type="button"
                className="icon-button"
                onClick={onReprocess}
                disabled={BUSY.includes(document.status) && !stuck(document)}
                title="Reprocess"
                aria-label={`Reprocess ${document.filename}`}
              >
                <RefreshCw size={16} aria-hidden="true" />
              </button>
              <button
                type="button"
                className="icon-button icon-button-danger"
                onClick={onDelete}
                title="Delete"
                aria-label={`Delete ${document.filename}`}
              >
                <Trash2 size={16} aria-hidden="true" />
              </button>
            </>
          )}
        </td>
      </tr>
      {open && (
        <tr className="row-detail">
          <td colSpan={8}>
            <Chunks documentId={document.id} />
          </td>
        </tr>
      )}
    </>
  )
}

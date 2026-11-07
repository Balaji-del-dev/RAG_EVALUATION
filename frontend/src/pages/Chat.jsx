import {
  ArrowUp,
  Check,
  ChevronDown,
  Copy,
  Gauge,
  Maximize2,
  MessageSquare,
  Minimize2,
  PanelLeft,
  Plus,
  Search,
  Square,
  ThumbsDown,
  ThumbsUp,
  Trash2,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { NavLink, useNavigate, useParams } from 'react-router-dom'
import { ApiError, api, results, streamApi, useApi } from '../api.js'
import { AnswerText, DownloadMenu, FormattedAnswer } from '../components/AnswerView.jsx'
import { ConfigMenu, ModelMenu } from '../components/ModelPicker.jsx'
import { ErrorNote, Status } from '../components/common.jsx'
import { useToast } from '../components/toast.jsx'
import { duration, score } from '../format.js'

const STAGE_NAMES = { vector: 'Vector search', bm25: 'BM25', rrf: 'RRF fusion', rerank: 'Reranked', mmr: 'After MMR' }
// Each one shows a different behaviour of the pipeline.
const SUGGESTIONS = [
  ['Ask a direct question', 'How many days of paid annual leave do full-time employees get?'],
  ['Summarize a policy', 'explain leave policy'],
  ['Search for a keyword', 'backups'],
  ['Try something not in the documents', 'What is the capital of France?'],
]
const isPhone = () => window.matchMedia('(max-width: 720px)').matches
const HISTORY_KEY = 'bench-history'
const EXPANDED_KEY = 'bench-chat-expanded'
const historyPreferred = () => localStorage.getItem(HISTORY_KEY) !== 'closed'

function cleanSource(text) {
  return text
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/^\s*[•·▪◦]\s*$\n?/gm, '')
    .replace(/\n{2,}/g, '\n')
    .trim()
}

function fromServer(message) {
  return {
    // A fresh answer from the query endpoint has message_id; a saved one has id.
    id: message.id ?? message.message_id,
    feedback: message.feedback ?? null,
    requestId: message.request_id,
    question: message.question,
    understoodAs: message.standalone_question,
    format: message.format ?? null,
    tampered: message.findings?.sources ?? [],
    checks: message.checks ?? {},
    cached: Boolean(message.cached),
    route: message.route ?? 'search',
    answer: message.answer,
    abstained: message.abstained,
    declineReason: message.decline_reason,
    blocked: message.blocked,
    block_reason: message.block_reason,
    quoted: message.quoted,
    masked: Boolean(message.findings?.output),
    sources: message.sources ?? message.chunks ?? [],
    stages: message.stages ?? {},
    timings: message.timings_ms ?? {},
    usage: message.usage ?? {},
  }
}

/** Sort conversations into the same date groups people know from other chat apps. */
function groupByDate(sessions) {
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const groups = [
    ['Today', []],
    ['Yesterday', []],
    ['Previous 7 days', []],
    ['Older', []],
  ]
  for (const session of sessions) {
    const days = Math.floor((today - new Date(session.updated_at)) / 86400000) + 1
    const age = new Date(session.updated_at) >= today ? 0 : days
    groups[age === 0 ? 0 : age === 1 ? 1 : age <= 7 ? 2 : 3][1].push(session)
  }
  return groups.filter(([, items]) => items.length)
}

function History({ open, activeId, onClose, onDeleted, sessions }) {
  const [filter, setFilter] = useState('')
  const [closedGroups, setClosedGroups] = useState(() => new Set())
  const toast = useToast()
  const visible = useMemo(() => {
    const term = filter.trim().toLowerCase()
    return results(sessions.data).filter((session) => !term || session.title.toLowerCase().includes(term))
  }, [sessions.data, filter])

  function toggleGroup(label) {
    setClosedGroups((current) => {
      const next = new Set(current)
      if (!next.delete(label)) next.add(label)
      return next
    })
  }

  async function remove(event, session) {
    event.preventDefault()
    if (!window.confirm(`Delete "${session.title}"?`)) return
    try {
      await api(`rag/sessions/${session.id}/`, { method: 'DELETE' })
      toast('Conversation deleted')
      sessions.reload()
      onDeleted(session.id)
    } catch (failure) {
      toast(failure.message, 'error')
    }
  }

  return (
    <aside className={`history ${open ? 'history-open' : ''}`} aria-label="Conversation history">
      <NavLink to="/chat" end className="button new-chat" onClick={onClose}>
        <Plus size={16} aria-hidden="true" />
        New chat
      </NavLink>
      <label className="history-search">
        <Search size={16} aria-hidden="true" />
        <span className="sr-only">Search conversations</span>
        <input type="search" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Search chats" />
      </label>
      <nav className="history-list" aria-label="Past conversations">
        {!visible.length && !sessions.loading && (
          <p className="history-empty">{filter ? 'No conversations match that search.' : 'Your conversations will appear here.'}</p>
        )}
        {groupByDate(visible).map(([label, items]) => (
          <div className="history-group" key={label}>
            <button type="button" className="history-group-name" aria-expanded={!closedGroups.has(label)} onClick={() => toggleGroup(label)}>
              <ChevronDown size={16} className={closedGroups.has(label) ? 'chevron-closed' : ''} aria-hidden="true" />
              {label}
              <span className="history-group-count">{items.length}</span>
            </button>
            {!closedGroups.has(label) && items.map((session) => (
              <div className={`history-row ${String(session.id) === String(activeId) ? 'history-row-on' : ''}`} key={session.id}>
                <NavLink to={`/chat/${session.id}`} className="history-item" title={session.title} onClick={onClose}>
                  {session.title || 'Untitled conversation'}
                </NavLink>
                <button
                  type="button"
                  className="icon-button icon-button-danger history-delete"
                  title="Delete conversation"
                  aria-label={`Delete conversation: ${session.title}`}
                  onClick={(event) => remove(event, session)}
                >
                  <Trash2 size={16} aria-hidden="true" />
                </button>
              </div>
            ))}
          </div>
        ))}
      </nav>
    </aside>
  )
}

/** What to double-check in an answer: figures in no source, citations to sources it was not given, sentences the
 * sources do not support, and anything the search had to work around. */
function AnswerChecks({ checks }) {
  if (!checks || !Object.keys(checks).length) return null
  const warnings = [
    checks.numbers?.length && `These figures are not in the sources: ${checks.numbers.join(', ')}.`,
    checks.missing_sources?.length && `It cites ${checks.missing_sources.join(', ')}, which it was not given.`,
    checks.layout,
    ...(checks.contradicted ?? []).map((sentence) => `The sources contradict: "${sentence}"`),
    ...(checks.unsupported ?? []).map((sentence) => `Not supported by its sources: "${sentence}"`),
  ].filter(Boolean)
  const notes = [...(checks.retrieval ?? [])]
  if (checks.regenerated) {
    const figures = checks.regenerated.figures.join(', ')
    notes.push(
      checks.regenerated.fixed
        ? `The first draft gave figures that are in no source (${figures}), so it was written again.`
        : `The first draft gave figures that are in no source (${figures}). It was written again, and the version with fewer is shown.`,
    )
  }
  return (
    <>
      {warnings.length > 0 && (
        <div className="answer-checks" role="note">
          <strong>Check before relying on this</strong>
          <ul>
            {warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      )}
      {notes.map((note) => (
        <p className="understood" key={note}>
          {note}
        </p>
      ))}
    </>
  )
}

function Thinking({ searchOnly }) {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    const timer = setInterval(() => setSeconds((value) => value + 1), 1000)
    return () => clearInterval(timer)
  }, [])
  return (
    <div className="thinking" role="status">
      <span className="thinking-dots" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      {searchOnly || seconds < 2 ? 'Searching your documents' : `Writing the answer, ${seconds} s`}
    </div>
  )
}

// How an answer was made, when it was not a document search. See apps/chatbot/services/routing.py.
const ROUTE_LABELS = {
  library: 'From the document list',
  conversation: 'Summary of this conversation',
  clarify: 'Question back to you',
}

const PROBLEMS = ['Wrong figures', 'Missing information', 'Not from the documents', 'Wrong format', 'Wrong month or person']

/** Asked after a thumbs-down. The answer goes to the Review page, where analysts turn it into a test case. */
function WhyNotHelpful({ messageId, onDone }) {
  const [picked, setPicked] = useState([])
  const [detail, setDetail] = useState('')
  const [saving, setSaving] = useState(false)
  const toast = useToast()

  async function save(event) {
    event.preventDefault()
    const comment = [picked.join('; '), detail.trim()].filter(Boolean).join(': ')
    if (!comment) return onDone()
    setSaving(true)
    try {
      await api(`rag/messages/${messageId}/feedback/`, { method: 'POST', body: { rating: 'down', comment } })
      toast('Thanks: this goes to the review list')
      onDone()
    } catch (failure) {
      toast(failure.message, 'error')
      setSaving(false)
    }
  }

  return (
    <form className="why-form" onSubmit={save}>
      <fieldset>
        <legend>What was wrong?</legend>
        <div className="why-chips">
          {PROBLEMS.map((problem) => (
            <button
              key={problem}
              type="button"
              className="why-chip"
              aria-pressed={picked.includes(problem)}
              onClick={() => setPicked((current) => (current.includes(problem) ? current.filter((item) => item !== problem) : [...current, problem]))}
            >
              {problem}
            </button>
          ))}
        </div>
      </fieldset>
      <label className="sr-only" htmlFor={`why-${messageId}`}>
        Details
      </label>
      <textarea
        id={`why-${messageId}`}
        rows={2}
        maxLength={1500}
        placeholder="What should it have said? (optional)"
        value={detail}
        onChange={(event) => setDetail(event.target.value)}
      />
      <div className="why-actions">
        <button type="button" className="button" onClick={onDone}>
          Skip
        </button>
        <button type="submit" className="button button-primary" disabled={saving || (!picked.length && !detail.trim())}>
          Send
        </button>
      </div>
    </form>
  )
}

function Turn({ turn, onRetry, onAskAgain }) {
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const [rating, setRating] = useState(turn.feedback)
  const [askWhy, setAskWhy] = useState(false)
  const toast = useToast()
  const details = useRef(null)
  const stages = Object.entries(turn.stages ?? {}).filter(([name]) => STAGE_NAMES[name])
  const sources = turn.sources ?? []

  function showSource(number) {
    setOpen(true)
    setTimeout(() => details.current?.querySelector(`[data-source="${number}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 50)
  }

  // Clicking the active thumb again clears it.
  async function rate(value) {
    const next = rating === value ? null : value
    try {
      await api(`rag/messages/${turn.id}/feedback/`, next ? { method: 'POST', body: { rating: next } } : { method: 'DELETE' })
      setRating(next)
      setAskWhy(next === 'down')
      if (next === 'up') toast('Thanks: marked as helpful')
    } catch (failure) {
      toast(failure.message, 'error')
    }
  }

  async function copy() {
    await navigator.clipboard.writeText(turn.answer)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <article className="turn">
      <div className="turn-user">
        <div className="bubble">{turn.question}</div>
      </div>
      <div className="turn-assistant">
        <span className="assistant-mark" aria-hidden="true">
          <Gauge size={16} />
        </span>
        <div className="assistant-body">
          {turn.pending && turn.rewriting && (
            <p className="understood" role="status">
              The first draft gave figures that are not in the sources, so it is being written again.
            </p>
          )}
          {turn.pending && !turn.streamText && <Thinking searchOnly={turn.searchOnly} />}
          {turn.pending && turn.streamText && (
            <AnswerText text={turn.streamText} streaming>
              <span className="caret" aria-hidden="true" />
            </AnswerText>
          )}
          {turn.error && (
            <div>
              <ErrorNote error={turn.error} />
              <button type="button" className="button" onClick={onRetry}>
                Try again
              </button>
            </div>
          )}
          {!turn.pending && !turn.error && (
            <>
              {turn.understoodAs && <p className="understood">Understood as: {turn.understoodAs}</p>}
              {turn.searchOnly ? (
                <p className="answer-note">Search only: these sources were found, and the model was not asked. This is not saved to your history.</p>
              ) : (
                <FormattedAnswer turn={turn} onCite={showSource} />
              )}
              <AnswerChecks checks={turn.checks} />
              {turn.tampered?.length > 0 && (
                <p className="understood decline-reason">
                  Instructions aimed at the AI were found in {turn.tampered.map((item) => item.source).join(', ')} and removed
                  before the model read it.
                </p>
              )}
              {turn.stopped && (
                <p className="understood decline-reason">
                  Stopped. The server still finishes this answer, and the full version is in this conversation&apos;s history.
                </p>
              )}
              {(turn.blocked || turn.abstained || turn.quoted || turn.masked) && (
                <div className="answer-flags">
                  {turn.blocked && <Status kind="critical">Blocked: {turn.block_reason.replaceAll('_', ' ')}</Status>}
                  {turn.abstained && <Status kind="warning">Declined to answer</Status>}
                  {turn.quoted && <Status kind="neutral">Quoted from the documents: the model did not write this</Status>}
                  {turn.masked && <Status kind="warning">Sensitive details hidden</Status>}
                </div>
              )}
              {turn.cached && (
                <p className="understood">
                  This question was asked before with the same documents, so the earlier answer was reused.{' '}
                  {onAskAgain && (
                    <button type="button" className="link-button" onClick={onAskAgain}>
                      Answer again
                    </button>
                  )}
                </p>
              )}
              {turn.declineReason && <p className="understood decline-reason">{turn.declineReason}</p>}
              <div className="turn-actions">
                {!turn.searchOnly && (
                  <button type="button" className="icon-button" onClick={copy} title={copied ? 'Copied' : 'Copy answer'} aria-label={copied ? 'Copied' : 'Copy answer'}>
                    {copied ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}
                  </button>
                )}
                {!turn.searchOnly && turn.id && <DownloadMenu messageId={turn.id} />}
                {!turn.searchOnly && turn.id && (
                  <>
                    <button
                      type="button"
                      className="icon-button feedback-button"
                      onClick={() => rate('up')}
                      title="Helpful"
                      aria-label="Helpful"
                      aria-pressed={rating === 'up'}
                    >
                      <ThumbsUp size={16} aria-hidden="true" />
                    </button>
                    <button
                      type="button"
                      className="icon-button feedback-button"
                      onClick={() => rate('down')}
                      title="Not helpful"
                      aria-label="Not helpful"
                      aria-pressed={rating === 'down'}
                    >
                      <ThumbsDown size={16} aria-hidden="true" />
                    </button>
                  </>
                )}
                {(sources.length > 0 || stages.length > 0) && (
                  <button type="button" className="details-toggle" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
                    {sources.length} {sources.length === 1 ? 'source' : 'sources'}
                    <ChevronDown size={16} className={open ? 'chevron-open' : ''} aria-hidden="true" />
                  </button>
                )}
                {!turn.searchOnly && (
                  <span className="turn-meta">
                    {turn.cached
                      ? 'From cache'
                      : ROUTE_LABELS[turn.route] ?? (turn.usage?.model ? turn.usage.model.split('/').pop() : 'No model call')}
                    {turn.route === 'conversation' && turn.usage?.model ? ` · ${turn.usage.model.split('/').pop()}` : ''}
                    {turn.timings?.total ? ` · ${duration(turn.timings.total)}` : ''}
                    {turn.format ? ` · ${turn.format.label}` : ''}
                  </span>
                )}
              </div>
              {askWhy && turn.id && <WhyNotHelpful messageId={turn.id} onDone={() => setAskWhy(false)} />}
              {open && (
                <div className="turn-details" ref={details}>
                  {stages.length > 0 && (
                    <>
                      <h3 className="details-title">How the search narrowed down</h3>
                      <ol className="pipeline">
                        {stages.map(([name, hits]) => (
                          <li className="pipeline-stage" key={name}>
                            <div className="pipeline-name">{STAGE_NAMES[name]}</div>
                            <div className="pipeline-count">{hits.length}</div>
                            <div className="pipeline-note">{turn.timings?.[name] !== undefined ? duration(turn.timings[name]) : 'instant'}</div>
                          </li>
                        ))}
                      </ol>
                    </>
                  )}
                  {!turn.searchOnly && turn.usage?.model && (
                    <dl className="readout">
                      <div>
                        <dt>Retrieval</dt>
                        <dd>{duration(turn.timings?.retrieval)}</dd>
                      </div>
                      <div>
                        <dt>Generation</dt>
                        <dd>{duration(turn.timings?.generation)}</dd>
                      </div>
                      {turn.timings?.first_token > 0 && (
                        <div>
                          <dt>First words</dt>
                          <dd>{duration(turn.timings.first_token)}</dd>
                        </div>
                      )}
                      {turn.usage?.tokens_per_second > 0 && (
                        <div>
                          <dt>Writing speed</dt>
                          <dd>{turn.usage.tokens_per_second.toFixed(1)} tok/s</dd>
                        </div>
                      )}
                      {turn.requestId && (
                        <div>
                          <dt>Request ID</dt>
                          <dd className="request-id" title={turn.requestId}>
                            {turn.requestId.slice(0, 12)}
                          </dd>
                        </div>
                      )}
                      <div>
                        <dt>Tokens in / out</dt>
                        <dd>
                          {turn.usage.prompt_tokens} / {turn.usage.completion_tokens}
                        </dd>
                      </div>
                    </dl>
                  )}
                  {sources.length > 0 && <h3 className="details-title">Sources found</h3>}
                  <ol className="source-list chunk-list">
                    {sources.map((chunk, index) => (
                      <li className="source" data-source={index + 1} key={chunk.chunk_id}>
                        <div className="source-head">
                          <span className="cite cite-static">{index + 1}</span>
                          <span className="source-file">{chunk.filename}</span>
                          {chunk.page_number ? <span>page {chunk.page_number}</span> : null}
                          <span>relevance {score(chunk.score)}</span>
                        </div>
                        <p>{cleanSource(chunk.text)}</p>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </article>
  )
}

export default function Chat() {
  const { sessionId } = useParams()
  const navigate = useNavigate()
  const sessions = useApi('rag/sessions/?limit=200')
  const configs = useApi('rag/configs/')
  const models = useApi('rag/models/')
  // The chosen model is remembered between visits. Empty means the configuration's own.
  const [model, setModel] = useState(() => localStorage.getItem('bench-model') ?? '')
  const [searchMode, setSearchMode] = useState(false)
  const streaming = useRef(null)
  const [turns, setTurns] = useState([])
  const [input, setInput] = useState('')
  const [config, setConfig] = useState('')
  const [busy, setBusy] = useState(false)
  const [loadError, setLoadError] = useState(null)
  const [expanded, setExpanded] = useState(() => !isPhone() && localStorage.getItem(EXPANDED_KEY) === 'true')
  const [historyOpen, setHistoryOpen] = useState(() => !isPhone() && !expanded && historyPreferred())
  const justCreated = useRef(null)
  const composer = useRef(null)
  const end = useRef(null)

  // Load a conversation when the address changes, unless it is the one just started on this screen.
  useEffect(() => {
    setLoadError(null)
    if (!sessionId) {
      setTurns([])
      return undefined
    }
    if (String(justCreated.current) === sessionId) {
      justCreated.current = null
      return undefined
    }
    let active = true
    api(`rag/sessions/${sessionId}/`)
      .then((session) => active && setTurns(session.messages.map(fromServer)))
      .catch((failure) => active && setLoadError(failure))
    return () => {
      active = false
    }
  }, [sessionId])

  useEffect(() => {
    end.current?.scrollIntoView({ block: 'end' })
  }, [turns])

  // The navigation is outside this page, so the page tells the whole document that it should step aside.
  useEffect(() => {
    document.documentElement.dataset.chatExpanded = String(expanded)
    return () => {
      delete document.documentElement.dataset.chatExpanded
    }
  }, [expanded])

  function toggleHistory() {
    const next = !historyOpen
    setHistoryOpen(next)
    if (!isPhone() && !expanded) localStorage.setItem(HISTORY_KEY, next ? 'open' : 'closed')
  }

  function toggleExpanded() {
    const next = !expanded
    setExpanded(next)
    localStorage.setItem(EXPANDED_KEY, String(next))
    setHistoryOpen(!next && historyPreferred())
  }

  function stop() {
    streaming.current?.abort()
  }

  function chooseModel(value) {
    setModel(value)
    localStorage.setItem('bench-model', value)
  }

  const allConfigs = results(configs.data)
  const activeConfigModel = (allConfigs.find((item) => String(item.id) === String(config)) ?? allConfigs.find((item) => item.is_default))?.llm_model

  function resize() {
    const box = composer.current
    if (!box) return
    box.style.height = 'auto'
    box.style.height = `${Math.min(box.scrollHeight, 160)}px`
  }

  useEffect(resize, [input])

  async function send(text, searchOnly = false, fresh = false) {
    const question = text.trim()
    if (!question || busy) return
    const key = crypto.randomUUID()
    setBusy(true)
    setInput('')
    setTurns((current) => [...current, { key, question, pending: true, searchOnly }])
    const body = {
      question,
      ...(config ? { rag_config: Number(config) } : {}),
      ...(model && !searchOnly ? { llm_model: model } : {}),
      ...(sessionId && !searchOnly ? { session: Number(sessionId) } : {}),
      ...(fresh ? { fresh: true } : {}),
    }
    const finish = (update) => setTurns((current) => current.map((turn) => (turn.key === key ? { key, question, ...update } : turn)))
    try {
      let data
      if (searchOnly) {
        data = await api('rag/retrieve/', { method: 'POST', body })
      } else {
        // The answer appears sentence by sentence. The "done" event has the final version, which replaces it.
        streaming.current = new AbortController()
        await streamApi(
          'rag/query/stream/',
          body,
          (name, event) => {
          if (name === 'text') {
            setTurns((current) =>
              current.map((turn) => (turn.key === key ? { ...turn, streamText: (turn.streamText ?? '') + event.text } : turn)),
            )
          } else if (name === 'reset') {
            // The first draft had figures that are in no source; the model is writing it again.
            setTurns((current) => current.map((turn) => (turn.key === key ? { ...turn, streamText: '', rewriting: true } : turn)))
          } else if (name === 'done') {
            data = event
          } else if (name === 'error') {
            throw new ApiError(event.error, 500)
          }
          },
          { signal: streaming.current.signal },
        )
        if (!data) throw new ApiError('The answer stopped before it finished. Try again.', 0)
      }
      finish(searchOnly ? { searchOnly: true, sources: data.chunks, stages: data.stages, timings: data.timings_ms } : fromServer(data))
      if (!searchOnly) {
        if (!sessionId) {
          justCreated.current = data.session
          navigate(`/chat/${data.session}`, { replace: true })
        }
        sessions.reload()
      }
    } catch (failure) {
      if (failure.name === 'AbortError') {
        setTurns((current) =>
          current.map((turn) => (turn.key === key ? { key, question, answer: turn.streamText ?? '', stopped: true } : turn)),
        )
        sessions.reload()
      } else {
        finish({ error: failure, searchOnly })
      }
    }
    streaming.current = null
    setBusy(false)
    composer.current?.focus()
  }

  function retry(turn) {
    setTurns((current) => current.filter((item) => item !== turn))
    send(turn.question, turn.searchOnly)
  }

  const activeTitle = results(sessions.data).find((session) => String(session.id) === String(sessionId))?.title

  return (
    <div className={historyOpen ? 'chat' : 'chat chat-history-closed'}>
      <h1 className="sr-only">Chat</h1>
      {historyOpen && <button type="button" className="history-scrim" aria-label="Close history" onClick={() => setHistoryOpen(false)} />}
      <History
        open={historyOpen}
        activeId={sessionId}
        sessions={sessions}
        onClose={() => {
          if (isPhone()) setHistoryOpen(false)
        }}
        onDeleted={(id) => {
          if (String(id) === String(sessionId)) navigate('/chat')
        }}
      />
      <section className="chat-main">
        <header className="chat-top">
          <button
            type="button"
            className="icon-button history-toggle"
            onClick={toggleHistory}
            title={historyOpen ? 'Hide history' : 'Show history'}
            aria-label={historyOpen ? 'Hide history' : 'Show history'}
            aria-expanded={historyOpen}
          >
            <PanelLeft size={16} aria-hidden="true" />
          </button>
          <div className="chat-title">{sessionId ? activeTitle || 'Conversation' : 'New chat'}</div>
          <button
            type="button"
            className="icon-button chat-expand"
            onClick={toggleExpanded}
            title={expanded ? 'Exit full screen' : 'Expand chat to full screen'}
            aria-label={expanded ? 'Exit full screen' : 'Expand chat to full screen'}
            aria-pressed={expanded}
          >
            {expanded ? <Minimize2 size={16} aria-hidden="true" /> : <Maximize2 size={16} aria-hidden="true" />}
          </button>
        </header>

        <div className="chat-scroll">
          <div className="chat-column">
            <ErrorNote error={loadError} />
            {!turns.length && !sessionId && (
              <div className="chat-empty">
                <span className="chat-empty-mark" aria-hidden="true">
                  <MessageSquare size={24} />
                </span>
                <h2>What would you like to know?</h2>
                <p>Answers come only from documents you are allowed to read, with sources you can check.</p>
                <div className="suggestions">
                  {SUGGESTIONS.map(([label, text]) => (
                    <button type="button" className="suggestion" key={label} onClick={() => send(text)}>
                      <span className="suggestion-label">{label}</span>
                      <span className="suggestion-text">{text}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
            {turns.map((turn) => (
              <Turn key={turn.key ?? turn.id} turn={turn} onRetry={() => retry(turn)} onAskAgain={busy ? null : () => send(turn.question, false, true)} />
            ))}
            <div ref={end} />
          </div>
        </div>

        <form
          className="chat-composer"
          onSubmit={(event) => {
            event.preventDefault()
            send(input, searchMode)
          }}
        >
          <div className="composer-box">
            <textarea
              ref={composer}
              rows={1}
              value={input}
              aria-label="Message"
              placeholder={searchMode ? 'Search your documents for a name or topic' : 'Ask anything about your documents'}
              maxLength={2000}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  send(input, searchMode)
                }
              }}
            />
            <div className="composer-tools">
              {/* Under the message, as in other chat apps: a search mode, the settings, the model and send. */}
              <div className="composer-side">
                <button
                  type="button"
                  className="pill"
                  aria-pressed={searchMode}
                  onClick={() => setSearchMode((value) => !value)}
                  title="Search only: show the matching passages without asking the model"
                >
                  <Search size={16} aria-hidden="true" />
                  <span className="pill-label">Search</span>
                </button>
                <ConfigMenu configs={configs} value={config} onChange={setConfig} model={model} />
              </div>
              <div className="composer-side">
                <ModelMenu value={model} onChange={chooseModel} models={models} configModel={activeConfigModel} />
                {busy && streaming.current ? (
                  <button type="button" className="composer-send composer-stop" onClick={stop} aria-label="Stop the answer" title="Stop">
                    <Square size={12} fill="currentColor" aria-hidden="true" />
                  </button>
                ) : (
                  <button type="submit" className="composer-send" disabled={busy || !input.trim()} aria-label="Send message" title="Send">
                    <ArrowUp size={16} strokeWidth={2.5} aria-hidden="true" />
                  </button>
                )}
              </div>
            </div>
          </div>
          <p className="composer-note">Enter to send, Shift + Enter for a new line. Check important answers against the sources.</p>
        </form>
      </section>
    </div>
  )
}

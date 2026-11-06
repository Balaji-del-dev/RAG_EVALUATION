import { ArrowUp, Check, ChevronDown, Copy, Gauge, Maximize2, MessageSquare, Minimize2, PanelLeft, Plus, Search, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { NavLink, useNavigate, useParams } from 'react-router-dom'
import { api, results, useApi } from '../api.js'
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
// [S1] becomes a link to its source. [EMAIL_ADDRESS] and the like are marked as hidden.
const ANSWER_TOKEN = /(\[S\d+\]|\[[A-Z][A-Z_]+\])/g

function cleanSource(text) {
  return text
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/^\s*[•·▪◦]\s*$\n?/gm, '')
    .replace(/\n{2,}/g, '\n')
    .trim()
}

function fromServer(message) {
  return {
    id: message.id,
    question: message.question,
    understoodAs: message.standalone_question,
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

function AnswerText({ text, onCite }) {
  return (
    <p className="answer">
      {text.split(ANSWER_TOKEN).map((part, index) => {
        const citation = part.match(/^\[S(\d+)\]$/)
        if (citation) {
          return (
            <button type="button" className="cite" key={index} title={`Show source ${citation[1]}`} onClick={() => onCite(citation[1])}>
              {citation[1]}
            </button>
          )
        }
        if (/^\[[A-Z][A-Z_]+\]$/.test(part)) {
          return (
            <span className="masked" key={index} title="Hidden by the security guard">
              {part.slice(1, -1).toLowerCase().replaceAll('_', ' ')} hidden
            </span>
          )
        }
        return part
      })}
    </p>
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

function Turn({ turn, onRetry }) {
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const details = useRef(null)
  const stages = Object.entries(turn.stages ?? {}).filter(([name]) => STAGE_NAMES[name])
  const sources = turn.sources ?? []

  function showSource(number) {
    setOpen(true)
    setTimeout(() => details.current?.querySelector(`[data-source="${number}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 50)
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
          {turn.pending && <Thinking searchOnly={turn.searchOnly} />}
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
                <AnswerText text={turn.answer} onCite={showSource} />
              )}
              {(turn.blocked || turn.abstained || turn.quoted || turn.masked) && (
                <div className="answer-flags">
                  {turn.blocked && <Status kind="critical">Blocked: {turn.block_reason.replaceAll('_', ' ')}</Status>}
                  {turn.abstained && <Status kind="warning">Declined to answer</Status>}
                  {turn.quoted && <Status kind="neutral">Quoted from the documents: the model did not write this</Status>}
                  {turn.masked && <Status kind="warning">Sensitive details hidden</Status>}
                </div>
              )}
              {turn.declineReason && <p className="understood decline-reason">{turn.declineReason}</p>}
              <div className="turn-actions">
                {!turn.searchOnly && (
                  <button type="button" className="icon-button" onClick={copy} title={copied ? 'Copied' : 'Copy answer'} aria-label={copied ? 'Copied' : 'Copy answer'}>
                    {copied ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}
                  </button>
                )}
                {(sources.length > 0 || stages.length > 0) && (
                  <button type="button" className="details-toggle" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
                    {sources.length} {sources.length === 1 ? 'source' : 'sources'}
                    <ChevronDown size={16} className={open ? 'chevron-open' : ''} aria-hidden="true" />
                  </button>
                )}
                {!turn.searchOnly && (
                  <span className="turn-meta">
                    {turn.usage?.model ? turn.usage.model.split('/').pop() : 'No model call'}
                    {turn.timings?.total ? ` · ${duration(turn.timings.total)}` : ''}
                  </span>
                )}
              </div>
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

  function resize() {
    const box = composer.current
    if (!box) return
    box.style.height = 'auto'
    box.style.height = `${Math.min(box.scrollHeight, 160)}px`
  }

  useEffect(resize, [input])

  async function send(text, searchOnly = false) {
    const question = text.trim()
    if (!question || busy) return
    const key = crypto.randomUUID()
    setBusy(true)
    setInput('')
    setTurns((current) => [...current, { key, question, pending: true, searchOnly }])
    const body = {
      question,
      ...(config ? { rag_config: Number(config) } : {}),
      ...(sessionId && !searchOnly ? { session: Number(sessionId) } : {}),
    }
    const finish = (update) => setTurns((current) => current.map((turn) => (turn.key === key ? { key, question, ...update } : turn)))
    try {
      const data = await api(searchOnly ? 'rag/retrieve/' : 'rag/query/', { method: 'POST', body })
      finish(searchOnly ? { searchOnly: true, sources: data.chunks, stages: data.stages, timings: data.timings_ms } : fromServer(data))
      if (!searchOnly) {
        if (!sessionId) {
          justCreated.current = data.session
          navigate(`/chat/${data.session}`, { replace: true })
        }
        sessions.reload()
      }
    } catch (failure) {
      finish({ error: failure, searchOnly })
    }
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
          <label className="chat-config">
            <span className="sr-only">Configuration</span>
            <select value={config} onChange={(e) => setConfig(e.target.value)}>
              <option value="">Default configuration</option>
              {results(configs.data).map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
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
              <Turn key={turn.key ?? turn.id} turn={turn} onRetry={() => retry(turn)} />
            ))}
            <div ref={end} />
          </div>
        </div>

        <form
          className="chat-composer"
          onSubmit={(event) => {
            event.preventDefault()
            send(input)
          }}
        >
          <div className="composer-box">
            <textarea
              ref={composer}
              rows={1}
              value={input}
              aria-label="Message"
              placeholder="Ask about your documents, or search for a name or topic"
              maxLength={2000}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  send(input)
                }
              }}
            />
            <div className="composer-tools">
              <button type="button" className="composer-search" disabled={busy || !input.trim()} onClick={() => send(input, true)}>
                <Search size={16} aria-hidden="true" />
                Search only
              </button>
              <button type="submit" className="composer-send" disabled={busy || !input.trim()} aria-label="Send message" title="Send">
                <ArrowUp size={16} strokeWidth={2.5} aria-hidden="true" />
              </button>
            </div>
          </div>
          <p className="composer-note">Enter to send, Shift + Enter for a new line. Check important answers against the sources.</p>
        </form>
      </section>
    </div>
  )
}

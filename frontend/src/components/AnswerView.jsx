import { Check, ChevronLeft, ChevronRight, Copy, Download, RotateCcw } from 'lucide-react'
import { Children, createContext, isValidElement, useContext, useEffect, useId, useMemo, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import remarkGfm from 'remark-gfm'
import { useTheme } from '../theme.jsx'

// [S1] becomes a link to its source and [EMAIL_ADDRESS] a "hidden" chip. They are turned into Markdown first, so that
// they also work inside tables and lists. Code blocks are left as written.
const CITE_HREF = /^#source-(\d+)$/
const MASK_CODE = /^masked:([A-Z_]+)$/
// Not coloured as code: diagrams are drawn, and these have no grammar worth colouring.
const PLAIN_LANGUAGES = ['mermaid', 'text', 'txt', 'csv', 'plaintext']

// The validated categorical palette (8 slots, fixed order) for each theme. Light slots 3, 4 and 5 are under 3:1 on
// white, so charts always show their values as labels.
const SERIES = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
}

export const DOWNLOADS = [
  ['Documents', [['pdf', 'PDF'], ['docx', 'Word'], ['odt', 'OpenDocument text'], ['rtf', 'RTF'], ['html', 'HTML page'], ['md', 'Markdown'], ['txt', 'Plain text']]],
  ['Data', [['xlsx', 'Excel'], ['ods', 'OpenDocument spreadsheet'], ['csv', 'CSV'], ['json', 'JSON'], ['yaml', 'YAML'], ['xml', 'XML']]],
  ['Slides', [['pptx', 'PowerPoint']]],
]
const FILE_LABELS = Object.fromEntries(DOWNLOADS.flatMap(([, items]) => items))

export const downloadUrl = (messageId, extension) => `/api/rag/messages/${messageId}/download/${extension}/`

function prepare(text) {
  return text
    .split(/(```[\s\S]*?```)/)
    .map((part) =>
      part.startsWith('```') ? part : part.replace(/\[S(\d+)\]/g, '[$1](#source-$1)').replace(/\[([A-Z][A-Z_]+)\]/g, '`masked:$1`'),
    )
    .join('')
}

/** The plain text inside rendered Markdown, for filtering table rows. */
function textOf(node) {
  if (node == null || typeof node === 'boolean') return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(textOf).join(' ')
  if (isValidElement(node)) return textOf(node.props.children)
  return ''
}

const TableFilter = createContext('')

// ---------- Diagrams ----------

let mermaidLoader
const loadMermaid = () => (mermaidLoader ??= import('mermaid').then((module) => module.default))

function mermaidTheme(dark) {
  const series = SERIES[dark ? 'dark' : 'light']
  const ink = dark ? '#f1f5f9' : '#0f172a'
  const surface = dark ? '#111827' : '#ffffff'
  const pies = Object.fromEntries(series.map((color, index) => [`pie${index + 1}`, color]))
  return {
    startOnLoad: false,
    securityLevel: 'strict',
    theme: 'base',
    fontFamily: 'inherit',
    xyChart: { showDataLabel: true },
    themeVariables: {
      background: surface,
      primaryColor: dark ? '#1e293b' : '#eef4ff',
      primaryBorderColor: series[0],
      primaryTextColor: ink,
      secondaryColor: dark ? '#1f2937' : '#f1f5f9',
      tertiaryColor: dark ? '#0b1120' : '#f8fafc',
      lineColor: dark ? '#94a3b8' : '#64748b',
      textColor: ink,
      titleColor: ink,
      edgeLabelBackground: surface,
      fontSize: '14px',
      ...pies,
      pieSectionTextColor: '#0b1120',
      pieStrokeColor: surface,
      pieOuterStrokeColor: surface,
      pieTitleTextColor: ink,
      pieLegendTextColor: ink,
      xyChart: {
        backgroundColor: surface,
        titleColor: ink,
        xAxisLabelColor: ink,
        yAxisLabelColor: ink,
        xAxisTitleColor: ink,
        yAxisTitleColor: ink,
        xAxisLineColor: dark ? '#475569' : '#cbd5e1',
        yAxisLineColor: dark ? '#475569' : '#cbd5e1',
        plotColorPalette: series.join(', '),
      },
    },
  }
}

function MermaidDiagram({ code }) {
  const { resolved } = useTheme()
  const id = useId().replace(/[^a-zA-Z0-9]/g, '')
  const [svg, setSvg] = useState('')
  const [failed, setFailed] = useState(false)
  const [showSource, setShowSource] = useState(false)

  useEffect(() => {
    let active = true
    loadMermaid()
      .then(async (mermaid) => {
        mermaid.initialize(mermaidTheme(resolved === 'dark'))
        const { svg: drawn } = await mermaid.render(`diagram-${id}-${resolved}`, code)
        if (active) {
          setSvg(drawn)
          setFailed(false)
        }
      })
      .catch(() => active && setFailed(true))
    return () => {
      active = false
    }
  }, [code, id, resolved])

  return (
    <figure className="diagram">
      {failed ? (
        <p className="diagram-failed">The model's diagram has a syntax error, so it is shown as written.</p>
      ) : (
        // Mermaid's strict mode sanitises the SVG it returns.
        <div className="diagram-canvas" role="img" aria-label="Diagram" dangerouslySetInnerHTML={{ __html: svg }} />
      )}
      {(failed || showSource) && (
        <pre className="diagram-source">
          <code>{code}</code>
        </pre>
      )}
      {!failed && (
        <button type="button" className="text-button" onClick={() => setShowSource((value) => !value)} aria-expanded={showSource}>
          {showSource ? 'Hide the diagram source' : 'Show the diagram source'}
        </button>
      )}
    </figure>
  )
}

function CodeBlock({ children, ...props }) {
  const [copied, setCopied] = useState(false)
  const block = useRef(null)
  async function copy() {
    await navigator.clipboard.writeText(block.current?.innerText ?? '')
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }
  return (
    <div className="code-block">
      <button type="button" className="icon-button code-copy" onClick={copy} title={copied ? 'Copied' : 'Copy code'} aria-label={copied ? 'Copied' : 'Copy code'}>
        {copied ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}
      </button>
      <pre ref={block} {...props}>
        {children}
      </pre>
    </div>
  )
}

/** An answer, rendered as Markdown: tables, lists, code and diagrams. Raw HTML in it is never rendered. */
export function AnswerText({ text, onCite = () => {}, streaming = false, render = 'markdown', children }) {
  const [filter, setFilter] = useState('')
  const interactive = render === 'checklist'
  const components = {
    a({ href, children: label }) {
      const cite = href?.match(CITE_HREF)
      if (cite) {
        return (
          <button type="button" className="cite" title={`Show source ${cite[1]}`} onClick={() => onCite(cite[1])}>
            {cite[1]}
          </button>
        )
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer">
          {label}
        </a>
      )
    },
    code({ children: content, className }) {
      const raw = String(content ?? '')
      const mask = raw.match(MASK_CODE)
      if (mask) {
        return (
          <span className="masked" title="Hidden by the security guard">
            {mask[1].toLowerCase().replaceAll('_', ' ')} hidden
          </span>
        )
      }
      if (className?.includes('language-mermaid') && !streaming) return <MermaidDiagram code={raw.trim()} />
      return <code className={className}>{content}</code>
    },
    pre({ children: content, node: _node, ...props }) {
      // A diagram must not sit in a <pre>: Mermaid measures labels in one font and the code font then clips them.
      const inner = Children.toArray(content)[0]
      if (!streaming && isValidElement(inner) && String(inner.props.className ?? '').includes('language-mermaid')) {
        return <>{content}</>
      }
      return <CodeBlock {...props}>{content}</CodeBlock>
    },
    table({ children: rows }) {
      return (
        <div className="table-scroll">
          <table>{rows}</table>
        </div>
      )
    },
    tr({ children: cells }) {
      return <FilteredRow>{cells}</FilteredRow>
    },
    input({ type, checked }) {
      if (type !== 'checkbox') return null
      return interactive ? <TaskBox initial={checked} /> : <input type="checkbox" checked={checked} disabled readOnly />
    },
  }
  return (
    <div className={streaming ? 'answer answer-streaming' : 'answer'} aria-busy={streaming || undefined}>
      {render === 'searchable' && !streaming && (
        <label className="table-search">
          <span className="sr-only">Filter the table</span>
          <input type="search" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Filter rows" />
        </label>
      )}
      <TableFilter.Provider value={render === 'searchable' ? filter.trim().toLowerCase() : ''}>
        <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[[rehypeHighlight, { plainText: PLAIN_LANGUAGES }]]} components={components}>
          {prepare(text)}
        </Markdown>
      </TableFilter.Provider>
      {interactive && !streaming && <ChecklistProgress text={text} />}
      {children}
    </div>
  )
}

function FilteredRow({ children }) {
  const filter = useContext(TableFilter)
  const cells = Children.toArray(children)
  const header = cells.some((cell) => isValidElement(cell) && cell.type === 'th')
  if (filter && !header && !textOf(children).toLowerCase().includes(filter)) return null
  return <tr>{children}</tr>
}

function TaskBox({ initial }) {
  const [done, setDone] = useState(Boolean(initial))
  return <input type="checkbox" className="task-box" checked={done} onChange={(event) => setDone(event.target.checked)} aria-label="Done" />
}

function ChecklistProgress({ text }) {
  const total = (text.match(/^\s*[-*] \[[ xX]\]/gm) ?? []).length
  return total ? <p className="answer-hint">Tick the boxes as you go. {total} tasks.</p> : null
}

// ---------- Layouts built from the answer's structure ----------

/** Sections from "## Heading" lines, or from bold lines standing alone. */
export function sectionsOf(text) {
  const lines = text.split('\n')
  const headingAt = (line) => line.match(/^#{2,4}\s+(.+?)\s*#*$/)?.[1] ?? line.match(/^\*\*([^*]+)\*\*:?\s*$/)?.[1]
  const sections = []
  let intro = []
  for (const line of lines) {
    const title = headingAt(line)
    if (title) sections.push({ title: title.replace(/\[S\d+\]/g, '').trim(), body: [] })
    else if (sections.length) sections.at(-1).body.push(line)
    else intro.push(line)
  }
  intro = intro.join('\n').trim()
  return { intro, sections: sections.map((section) => ({ ...section, body: section.body.join('\n').trim() })) }
}

function Accordion({ sections, onCite }) {
  return (
    <div className="layout-accordion">
      {sections.map((section, index) => (
        <details key={section.title + index} open={index === 0}>
          <summary>{section.title}</summary>
          <AnswerText text={section.body} onCite={onCite} />
        </details>
      ))}
    </div>
  )
}

function Tabs({ sections, onCite }) {
  const [active, setActive] = useState(0)
  const id = useId()
  return (
    <div className="layout-tabs">
      <div role="tablist" aria-label="Sections" className="tab-list">
        {sections.map((section, index) => (
          <button
            key={section.title + index}
            type="button"
            role="tab"
            id={`${id}-tab-${index}`}
            aria-selected={active === index}
            aria-controls={`${id}-panel`}
            tabIndex={active === index ? 0 : -1}
            className="tab"
            onClick={() => setActive(index)}
            onKeyDown={(event) => {
              const step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0
              if (step) setActive((active + step + sections.length) % sections.length)
            }}
          >
            {section.title}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`${id}-panel`} aria-labelledby={`${id}-tab-${active}`} className="tab-panel">
        <AnswerText text={sections[active].body} onCite={onCite} />
      </div>
    </div>
  )
}

function Cards({ sections, onCite }) {
  return (
    <div className="layout-cards">
      {sections.map((section, index) => (
        <section className="answer-card" key={section.title + index}>
          <h3>{section.title}</h3>
          <AnswerText text={section.body} onCite={onCite} />
        </section>
      ))}
    </div>
  )
}

function Carousel({ sections, onCite }) {
  const [active, setActive] = useState(0)
  const move = (step) => setActive((active + step + sections.length) % sections.length)
  return (
    <div className="layout-carousel" aria-roledescription="carousel">
      <section className="answer-card carousel-slide" aria-roledescription="slide" aria-label={`${active + 1} of ${sections.length}`}>
        <h3>{sections[active].title}</h3>
        <AnswerText text={sections[active].body} onCite={onCite} />
      </section>
      <div className="carousel-controls">
        <button type="button" className="icon-button" onClick={() => move(-1)} aria-label="Previous" title="Previous">
          <ChevronLeft size={16} aria-hidden="true" />
        </button>
        <span className="carousel-count">
          {active + 1} of {sections.length}
        </span>
        <button type="button" className="icon-button" onClick={() => move(1)} aria-label="Next" title="Next">
          <ChevronRight size={16} aria-hidden="true" />
        </button>
      </div>
    </div>
  )
}

/** "- **when** - what happened [S1]" lines as a vertical timeline. */
export function timelineOf(text) {
  return text
    .split('\n')
    .map((line) => line.match(/^\s*(?:[-*]|\d+[.)])\s+(?:\*\*(.+?)\*\*|([^:–-]{1,40}?))\s*(?:[:–-]|—)\s*(.+)$/))
    .filter(Boolean)
    .map((match) => ({ when: (match[1] ?? match[2]).trim(), what: match[3].trim() }))
}

function Timeline({ items, onCite }) {
  return (
    <ol className="layout-timeline">
      {items.map((item, index) => (
        <li key={item.when + index}>
          <span className="timeline-when">{item.when}</span>
          <AnswerText text={item.what} onCite={onCite} />
        </li>
      ))}
    </ol>
  )
}

/** "**Label:** value" lines as figures, then the rest of the answer. */
export function dashboardOf(text) {
  const tiles = []
  const rest = []
  for (const line of text.split('\n')) {
    const match = line.match(/^\s*(?:[-*]\s+)?\*\*([^*]{1,40}?):?\*\*:?\s*(.+)$/)
    if (match && tiles.length < 8) tiles.push({ label: match[1].replace(/:$/, ''), value: match[2] })
    else rest.push(line)
  }
  return { tiles, rest: rest.join('\n').trim() }
}

function Dashboard({ tiles, rest, onCite }) {
  return (
    <div className="layout-dashboard">
      <div className="dashboard-tiles">
        {tiles.map((tile) => (
          <div className="dashboard-tile" key={tile.label}>
            <div className="dashboard-label">{tile.label}</div>
            <AnswerText text={tile.value} onCite={onCite} />
          </div>
        ))}
      </div>
      {rest && <AnswerText text={rest} onCite={onCite} />}
    </div>
  )
}

/** "Q1. question", "A) option" ... "Answer: B" blocks. */
export function quizOf(text) {
  const questions = []
  for (const line of text.split('\n').map((value) => value.trim().replace(/^\*\*|\*\*$/g, ''))) {
    const question = line.match(/^(?:\*\*)?(?:Q(?:uestion)?\s*\d+|\d+)[.):]\s*(?:\*\*)?\s*(.+)$/i)
    const option = line.match(/^\(?([A-Da-d])[).:]\s+(.+)$/)
    const answer = line.match(/^(?:\*\*)?(?:Correct\s+)?Answer(?:\*\*)?\s*[:-]\s*(?:\*\*)?\(?([A-Da-d])\b/i)
    const source = line.match(/^Source\s*[:-]\s*(.+)$/i)
    if (question && !option) questions.push({ text: question[1], options: [], answer: '', source: '' })
    else if (option && questions.length) questions.at(-1).options.push({ key: option[1].toUpperCase(), text: option[2] })
    else if (answer && questions.length) questions.at(-1).answer = answer[1].toUpperCase()
    else if (source && questions.length) questions.at(-1).source = source[1]
  }
  return questions.filter((question) => question.options.length >= 2 && question.answer)
}

function Quiz({ questions, onCite }) {
  const [picked, setPicked] = useState({})
  const score = questions.filter((question, index) => picked[index] === question.answer).length
  const answered = Object.keys(picked).length
  return (
    <div className="layout-quiz">
      {questions.map((question, index) => (
        <fieldset className="quiz-question" key={question.text + index}>
          <legend>
            {index + 1}. <AnswerText text={question.text} onCite={onCite} />
          </legend>
          {question.options.map((option) => {
            const chosen = picked[index] === option.key
            const reveal = picked[index] !== undefined
            const state = reveal && option.key === question.answer ? 'right' : chosen ? 'wrong' : ''
            return (
              <button
                key={option.key}
                type="button"
                className={`quiz-option ${state ? `quiz-${state}` : ''}`}
                disabled={reveal}
                aria-pressed={chosen}
                onClick={() => setPicked({ ...picked, [index]: option.key })}
              >
                <span className="quiz-key">{option.key}</span>
                <span>{option.text.replace(/\[S\d+\]/g, '')}</span>
                {reveal && option.key === question.answer && <span className="quiz-mark">Correct</span>}
                {chosen && option.key !== question.answer && <span className="quiz-mark">Your answer</span>}
              </button>
            )
          })}
          {picked[index] !== undefined && question.source && <AnswerText text={`Source: ${question.source}`} onCite={onCite} />}
        </fieldset>
      ))}
      <div className="quiz-score" role="status">
        {answered < questions.length ? `${answered} of ${questions.length} answered` : `Score: ${score} of ${questions.length}`}
        {answered > 0 && (
          <button type="button" className="text-button" onClick={() => setPicked({})}>
            <RotateCcw size={16} aria-hidden="true" /> Start again
          </button>
        )}
      </div>
    </div>
  )
}

/** "Q: question" then "A: answer" lines. */
export function flashcardsOf(text) {
  const cards = []
  for (const line of text.split('\n').map((value) => value.trim().replace(/\*\*/g, ''))) {
    const question = line.match(/^(?:[-*]\s+)?Q(?:uestion)?\s*\d*\s*[:.]\s*(.+)$/i)
    const answer = line.match(/^(?:[-*]\s+)?A(?:nswer)?\s*\d*\s*[:.]\s*(.+)$/i)
    if (question) cards.push({ front: question[1], back: '' })
    else if (answer && cards.length && !cards.at(-1).back) cards.at(-1).back = answer[1]
  }
  return cards.filter((card) => card.back)
}

function Flashcards({ cards }) {
  const [flipped, setFlipped] = useState({})
  return (
    <div className="layout-flashcards">
      {cards.map((card, index) => (
        <button
          key={card.front + index}
          type="button"
          className={`flashcard ${flipped[index] ? 'flashcard-back' : ''}`}
          onClick={() => setFlipped({ ...flipped, [index]: !flipped[index] })}
          aria-pressed={Boolean(flipped[index])}
        >
          <span className="flashcard-side">{flipped[index] ? 'Answer' : 'Question'}</span>
          <span className="flashcard-text">{(flipped[index] ? card.back : card.front).replace(/\[S(\d+)\]/g, '(source $1)')}</span>
        </button>
      ))}
      <p className="answer-hint">Click a card to turn it over.</p>
    </div>
  )
}

function HtmlPreview({ messageId }) {
  const [page, setPage] = useState('')
  const [view, setView] = useState('preview')
  useEffect(() => {
    let active = true
    fetch(`${downloadUrl(messageId, 'html')}?inline=1`, { credentials: 'same-origin' })
      .then((response) => (response.ok ? response.text() : ''))
      .then((html) => active && setPage(html))
    return () => {
      active = false
    }
  }, [messageId])
  if (!page) return null
  return (
    <div className="html-preview">
      <div className="view-switch" role="group" aria-label="HTML view">
        <button type="button" aria-pressed={view === 'preview'} onClick={() => setView('preview')}>
          Preview
        </button>
        <button type="button" aria-pressed={view === 'source'} onClick={() => setView('source')}>
          HTML source
        </button>
      </div>
      {view === 'preview' ? (
        // No sandbox permissions: the page cannot run scripts, submit forms or reach this app.
        <iframe title="HTML preview" sandbox="" srcDoc={page} className="html-frame" />
      ) : (
        <AnswerText text={`\`\`\`html\n${page}\n\`\`\``} />
      )}
      <a className="button" href={downloadUrl(messageId, 'html')} download>
        <Download size={16} aria-hidden="true" /> Download HTML
      </a>
    </div>
  )
}

/** The answer in the layout its question asked for. Falls back to plain Markdown when the model's structure is off. */
export function FormattedAnswer({ turn, onCite }) {
  const render = turn.format?.render ?? 'markdown'
  const text = turn.answer
  const structure = useMemo(() => {
    if (['accordion', 'tabs', 'cards', 'carousel'].includes(render)) {
      const { intro, sections } = sectionsOf(text)
      return sections.length >= 2 ? { intro, sections } : null
    }
    if (render === 'timeline') return timelineOf(text).length >= 2 ? timelineOf(text) : null
    if (render === 'dashboard') return dashboardOf(text).tiles.length >= 2 ? dashboardOf(text) : null
    if (render === 'quiz') return quizOf(text).length ? quizOf(text) : null
    if (render === 'flashcards') return flashcardsOf(text).length ? flashcardsOf(text) : null
    return null
  }, [render, text])

  let body = <AnswerText text={text} onCite={onCite} render={render} />
  if (structure) {
    const sectioned = { accordion: Accordion, tabs: Tabs, cards: Cards, carousel: Carousel }[render]
    if (sectioned) {
      const Layout = sectioned
      body = (
        <>
          {structure.intro && <AnswerText text={structure.intro} onCite={onCite} />}
          <Layout sections={structure.sections} onCite={onCite} />
        </>
      )
    } else if (render === 'timeline') body = <Timeline items={structure} onCite={onCite} />
    else if (render === 'dashboard') body = <Dashboard {...structure} onCite={onCite} />
    else if (render === 'quiz') body = <Quiz questions={structure} onCite={onCite} />
    else if (render === 'flashcards') body = <Flashcards cards={structure} />
  }

  return (
    <div className={`formatted formatted-${render}`}>
      {body}
      {turn.format?.note && <p className="understood">{turn.format.note}</p>}
      {render === 'html' && turn.id && <HtmlPreview messageId={turn.id} />}
      {render === 'file' && turn.id && turn.format?.target && (
        <a className="button button-primary file-download" href={downloadUrl(turn.id, turn.format.target)} download>
          <Download size={16} aria-hidden="true" /> Download {FILE_LABELS[turn.format.target] ?? turn.format.target.toUpperCase()}
        </a>
      )}
    </div>
  )
}

/** Every answer can be saved as a file. */
export function DownloadMenu({ messageId }) {
  const [open, setOpen] = useState(false)
  const box = useRef(null)
  useEffect(() => {
    if (!open) return undefined
    const close = (event) => {
      if (event.type === 'keydown' ? event.key === 'Escape' : !box.current?.contains(event.target)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    document.addEventListener('keydown', close)
    return () => {
      document.removeEventListener('mousedown', close)
      document.removeEventListener('keydown', close)
    }
  }, [open])
  return (
    <div className="download-menu" ref={box}>
      <button type="button" className="icon-button" onClick={() => setOpen((value) => !value)} aria-expanded={open} aria-haspopup="true" title="Download" aria-label="Download this answer">
        <Download size={16} aria-hidden="true" />
      </button>
      {open && (
        <div className="download-list" role="group" aria-label="Download as">
          {DOWNLOADS.map(([group, items]) => (
            <div key={group}>
              <div className="download-group">{group}</div>
              {items.map(([extension, label]) => (
                <a key={extension} className="download-item" href={downloadUrl(messageId, extension)} download onClick={() => setOpen(false)}>
                  {label}
                  <span className="download-ext">.{extension}</span>
                </a>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

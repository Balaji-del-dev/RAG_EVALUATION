import { useId, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, useApi } from '../api.js'
import { AnswerText } from '../components/AnswerView.jsx'
import { Empty, ErrorNote, Field, Page, Panel, Status } from '../components/common.jsx'
import { useToast } from '../components/toast.jsx'
import { dateTime } from '../format.js'

const PAGE = 25
const KINDS = [
  ['', 'All'],
  ['thumbs_down', 'Not helpful'],
  ['flagged', 'Flagged'],
  ['declined', 'Declined'],
  ['failed', 'Failed'],
]
const TAG_KIND = { thumbs_down: 'critical', flagged: 'warning', declined: 'neutral', failed: 'critical' }
const sentences = (count) => `${count} ${count === 1 ? 'sentence' : 'sentences'}`
const CHECK_TEXT = {
  numbers: (value) => `Figures in no source: ${value.join(', ')}`,
  missing_sources: (value) => `Cites sources it was not given: ${value.join(', ')}`,
  layout: (value) => value,
  unsupported: (value) => `${sentences(value.length)} the sources do not support`,
  contradicted: (value) => `${sentences(value.length)} the sources contradict`,
}

/** One chat answer to look at again. Selecting it opens the fields its test case needs. */
function ReviewItem({ item, labels, choice, onChoose }) {
  const id = useId()
  const selected = Boolean(choice)
  const needsAnswer = selected && choice.answerable && !choice.reference_answer.trim()
  return (
    <li className={selected ? 'review-item review-item-selected' : 'review-item'}>
      <div className="review-head">
        <label className="review-select">
          <input
            type="checkbox"
            checked={selected}
            onChange={(event) => onChoose(event.target.checked ? { reference_answer: '', answerable: true } : null)}
          />
          Add to dataset
        </label>
        <div className="review-tags">
          {item.tags.map((tag) => (
            <Status key={tag} kind={TAG_KIND[tag]}>
              {labels[tag]}
            </Status>
          ))}
        </div>
        <span className="review-when">
          {dateTime(item.created_at)}
          {item.model ? ` · ${item.model.split('/').pop()}` : ''}
        </span>
      </div>
      <p className="review-question">{item.question}</p>
      {item.standalone_question && <p className="understood">Understood as: {item.standalone_question}</p>}
      {item.feedback?.comment && (
        <blockquote className="review-comment">
          <span className="sr-only">The asker said: </span>
          {item.feedback.comment}
        </blockquote>
      )}
      {item.answer ? (
        <details className="review-answer">
          <summary>Answer given</summary>
          <AnswerText text={item.answer} />
        </details>
      ) : (
        <p className="panel-note">No answer: the request failed.</p>
      )}
      {(Object.keys(item.checks).length > 0 || item.decline_reason) && (
        <ul className="review-checks">
          {Object.entries(item.checks).map(([key, value]) => (
            <li key={key}>{CHECK_TEXT[key]?.(value) ?? key}</li>
          ))}
          {item.decline_reason && <li>{item.decline_reason}</li>}
        </ul>
      )}
      {item.sources.length > 0 && <p className="understood">Sources: {item.sources.join(', ')}</p>}
      {selected && (
        <div className="review-case">
          <Field label="Correct answer" hint="What the answer should have said, from the documents. The test compares against it.">
            <textarea
              id={`${id}-answer`}
              rows={3}
              value={choice.reference_answer}
              disabled={!choice.answerable}
              aria-invalid={needsAnswer || undefined}
              onChange={(event) => onChoose({ ...choice, reference_answer: event.target.value })}
            />
          </Field>
          <label className="review-select">
            <input
              type="checkbox"
              checked={!choice.answerable}
              onChange={(event) => onChoose({ ...choice, answerable: !event.target.checked })}
            />
            The documents do not answer this: the right reply is to decline
          </label>
        </div>
      )}
    </li>
  )
}

export default function Review() {
  const [kind, setKind] = useState('')
  const [offset, setOffset] = useState(0)
  const [choices, setChoices] = useState({})
  const [name, setName] = useState('Chat review')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(null)
  const toast = useToast()
  const tabsId = useId()
  const list = useApi(`rag/review/?days=30&kind=${kind}&limit=${PAGE}&offset=${offset}`)
  const data = list.data

  const chosen = Object.entries(choices).filter(([, choice]) => choice)
  const incomplete = chosen.filter(([, choice]) => choice.answerable && !choice.reference_answer.trim()).length

  function choose(id, choice) {
    setChoices((current) => ({ ...current, [id]: choice }))
  }

  function show(next) {
    setKind(next)
    setOffset(0)
  }

  async function save() {
    setSaving(true)
    try {
      const items = chosen.map(([id, choice]) => ({ message_id: Number(id), ...choice, reference_answer: choice.reference_answer.trim() }))
      const dataset = await api('rag/review/dataset/', { method: 'POST', body: { name: name.trim(), items } })
      setSaved(dataset)
      setChoices({})
      toast(`Saved ${dataset.case_count} cases as ${dataset.name} v${dataset.version}`)
    } catch (failure) {
      toast(failure.message, 'error')
    }
    setSaving(false)
  }

  return (
    <Page
      title="Review"
      intro="Chat answers from the last 30 days that someone rated not helpful, that the checks flagged, that declined, or that failed. Write the correct answer for the ones that matter and save them as test cases."
    >
      <div role="tablist" aria-label="Which answers" className="tab-list review-tabs">
        {KINDS.map(([key, label], index) => (
          <button
            key={key || 'all'}
            type="button"
            role="tab"
            id={`${tabsId}-${index}`}
            aria-selected={kind === key}
            aria-controls={`${tabsId}-panel`}
            tabIndex={kind === key ? 0 : -1}
            className="tab"
            onClick={() => show(key)}
            onKeyDown={(event) => {
              const step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0
              if (!step) return
              const next = (KINDS.findIndex(([value]) => value === kind) + step + KINDS.length) % KINDS.length
              show(KINDS[next][0])
              document.getElementById(`${tabsId}-${next}`)?.focus()
            }}
          >
            {label}
            {key && data?.counts ? <span className="tab-count">{data.counts[key]}</span> : null}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`${tabsId}-panel`} aria-labelledby={`${tabsId}-${KINDS.findIndex(([value]) => value === kind)}`}>
        <ErrorNote error={list.error} />
        {saved && (
          <p className="review-saved" role="status">
            Saved as {saved.name} v{saved.version} with {saved.case_count} cases. Start a run on it from <Link to="/runs">Runs</Link>.
          </p>
        )}
        {data && data.count === 0 && (
          <Empty title="Nothing to review">
            No answers of this kind in the last 30 days. Thumbs-down ratings in the Chat appear here.
          </Empty>
        )}
        {data && data.count > 0 && (
          <Panel note={`${data.count} ${data.count === 1 ? 'answer' : 'answers'}, newest first. Who asked is not shown.`}>
            <ul className="review-list">
              {data.results.map((item) => (
                <ReviewItem key={item.id} item={item} labels={data.labels} choice={choices[item.id]} onChoose={(choice) => choose(item.id, choice)} />
              ))}
            </ul>
            {(data.previous || data.next) && (
              <div className="review-pages">
                <button type="button" className="button" disabled={!data.previous} onClick={() => setOffset(Math.max(offset - PAGE, 0))}>
                  Newer
                </button>
                <span className="panel-note">
                  {offset + 1}–{Math.min(offset + PAGE, data.count)} of {data.count}
                </span>
                <button type="button" className="button" disabled={!data.next} onClick={() => setOffset(offset + PAGE)}>
                  Older
                </button>
              </div>
            )}
          </Panel>
        )}
      </div>
      {chosen.length > 0 && (
        <div className="review-bar" role="region" aria-label="New dataset">
          <span className="review-bar-count">
            {chosen.length} selected
            {incomplete > 0 && <span className="review-bar-hint"> · {incomplete} still need a correct answer</span>}
          </span>
          <Field label="Dataset name">
            <input value={name} maxLength={255} onChange={(event) => setName(event.target.value)} />
          </Field>
          <button type="button" className="button" onClick={() => setChoices({})}>
            Clear
          </button>
          <button type="button" className="button button-primary" disabled={saving || incomplete > 0 || !name.trim()} onClick={save}>
            Save as new version
          </button>
        </div>
      )}
    </Page>
  )
}

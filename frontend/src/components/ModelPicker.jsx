import { Check, ChevronDown, Cpu, SlidersHorizontal } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'
import { results } from '../api.js'

const GB = 1000 ** 3 // decimal, as `ollama ls` shows sizes
const gb = (bytes) => `${(bytes / GB).toFixed(1)} GB`
// What a model needs rounds up and what is free rounds down, so "needs 2.8 GB, 2.8 GB free" never appears.
const gbUp = (bytes) => `${(Math.ceil((bytes / GB) * 10) / 10).toFixed(1)} GB`
const gbDown = (bytes) => `${(Math.floor((bytes / GB) * 10) / 10).toFixed(1)} GB`
export const shortName = (model) => (model ?? '').replace(/^ollama(_chat)?\//, '')

/** A pill in the message box that opens a list above it, as in other chat apps. Arrow keys move, Enter picks,
 * Escape closes. Items that cannot be picked stay visible, greyed out, with the reason. */
function MenuPill({ icon: Icon, label, title, items, value, onSelect, onOpen, footer, align = 'left', className = '' }) {
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const box = useRef(null)
  const list = useRef(null)
  const id = useId()
  const enabled = items.map((item, index) => (item.disabled ? -1 : index)).filter((index) => index >= 0)

  useEffect(() => {
    if (!open) return undefined
    const close = (event) => {
      if (!box.current?.contains(event.target)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    list.current?.focus()
    return () => document.removeEventListener('mousedown', close)
  }, [open])

  function show() {
    const chosen = items.findIndex((item) => item.value === value)
    setActive(chosen >= 0 ? chosen : (enabled[0] ?? 0))
    setOpen(true)
    onOpen?.()
  }

  function pick(item) {
    if (item.disabled) return
    onSelect(item.value)
    setOpen(false)
    box.current?.querySelector('button')?.focus()
  }

  function onKeyDown(event) {
    const position = enabled.indexOf(active)
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      const step = event.key === 'ArrowDown' ? 1 : -1
      setActive(enabled[(position + step + enabled.length) % enabled.length])
    } else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      if (items[active]) pick(items[active])
    } else if (event.key === 'Escape' || event.key === 'Tab') {
      setOpen(false)
      if (event.key === 'Escape') box.current?.querySelector('button')?.focus()
    }
  }

  return (
    <div className={`menu-pill ${className}`} ref={box}>
      <button
        type="button"
        className="pill"
        onClick={() => (open ? setOpen(false) : show())}
        aria-haspopup="listbox"
        aria-expanded={open}
        title={title}
      >
        <Icon size={16} aria-hidden="true" />
        <span className="pill-label">{label}</span>
        <ChevronDown size={16} aria-hidden="true" className={open ? 'chevron-open' : ''} />
      </button>
      {open && (
        <div className={`menu-popover menu-${align}`}>
          <ul
            ref={list}
            role="listbox"
            tabIndex={-1}
            aria-label={title}
            aria-activedescendant={`${id}-${active}`}
            className="menu-list"
            onKeyDown={onKeyDown}
          >
            {items.map((item, index) => (
              <li
                key={item.value || 'default'}
                id={`${id}-${index}`}
                role="option"
                aria-selected={item.value === value}
                aria-disabled={item.disabled || undefined}
                className={`menu-item ${index === active ? 'menu-item-active' : ''} ${item.disabled ? 'menu-item-disabled' : ''}`}
                onMouseEnter={() => !item.disabled && setActive(index)}
                onClick={() => pick(item)}
              >
                <span className="menu-item-text">
                  <span className="menu-item-title">{item.title}</span>
                  {item.detail && <span className="menu-item-detail">{item.detail}</span>}
                </span>
                {item.value === value && <Check size={16} aria-hidden="true" className="menu-check" />}
              </li>
            ))}
          </ul>
          {footer}
        </div>
      )}
    </div>
  )
}

/** The installed Ollama chat models. Embedding-only ones are left out; those too big for free memory are greyed out. */
export function ModelMenu({ value, onChange, models, configModel }) {
  const list = (models.data?.models ?? []).filter((model) => model.can_chat)
  const free = models.data?.available_bytes ?? 0
  const items = [
    { value: '', title: `${shortName(configModel) || 'Configuration model'}`, detail: "The configuration's own model" },
    ...list.map((model) => ({
      value: model.model,
      title: model.name,
      detail: model.fits
        ? `${model.parameters} · ${gb(model.size_bytes)}${model.loaded ? ' · loaded' : ''}`
        : `${model.parameters} · needs ${gbUp(model.needed_bytes)}, ${gbDown(free)} free`,
      disabled: !model.fits,
    })),
  ]
  return (
    <MenuPill
      icon={Cpu}
      className="chat-model"
      label={shortName(value) || shortName(configModel) || 'Model'}
      title="Model"
      items={items}
      value={value}
      onSelect={onChange}
      onOpen={models.reload}
      align="right"
      footer={
        <p className="menu-footer" role={models.error && !list.length ? 'alert' : undefined}>
          {!list.length && models.error
            ? `Could not list the installed models: ${models.error.message}`
            : !list.length
              ? 'Finding the installed models…'
              : 'Models too big for the memory free right now are greyed out. Close other apps to use them.'}
        </p>
      }
    />
  )
}

function configSummary(config) {
  if (!config) return ''
  return [config.retrieval_mode, config.reranker ? 'rerank' : '', config.use_mmr ? 'MMR' : '', `top ${config.top_k}`]
    .filter(Boolean)
    .join(' · ')
}

/** The configurations, and below them what the chosen one does. */
export function ConfigMenu({ configs, value, onChange, model }) {
  const all = results(configs.data)
  const current = all.find((item) => String(item.id) === String(value)) ?? all.find((item) => item.is_default) ?? all[0]
  const items = [
    { value: '', title: 'Default configuration', detail: configSummary(all.find((item) => item.is_default)) },
    ...all.map((item) => ({ value: String(item.id), title: item.name, detail: configSummary(item) })),
  ]
  const chosen = model && current && model !== current.llm_model
  const rows = current
    ? [
        ['Model', chosen ? `${shortName(model)} (chosen for this chat)` : shortName(current.llm_model)],
        ['Prompt', `${current.prompt_version}, temperature ${current.temperature}, up to ${current.max_tokens} tokens`],
        ['Search', `${current.retrieval_mode}, ${current.candidate_k} candidates, ${current.top_k} given to the model`],
        ['Reranker', current.reranker ? `${shortName(current.reranker)}, top ${current.rerank_top_n}` : 'none'],
        ['Diversity', current.use_mmr ? `MMR, λ ${current.mmr_lambda}` : 'off'],
        ['Query expansion', { multi_query: 'Multi-query', hyde: 'HyDE' }[current.query_expansion] ?? 'off'],
        ['Chunks', `${current.chunk_strategy}, ${current.chunk_size} tokens, ${current.chunk_overlap} overlap`],
        ['Embeddings', shortName(current.embedding_model).replace('sentence-transformers/', '')],
      ]
    : []
  return (
    <MenuPill
      icon={SlidersHorizontal}
      className="chat-settings"
      label={current && value ? current.name : 'Default configuration'}
      title="Configuration"
      items={items}
      value={value}
      onSelect={onChange}
      footer={
        current && (
          <section className="config-panel" aria-label="About this configuration">
            <h2>{current.name}</h2>
            <dl>
              {rows.map(([label, text]) => (
                <div key={label}>
                  <dt>{label}</dt>
                  <dd>{text}</dd>
                </div>
              ))}
            </dl>
            <p className="menu-footer">A model chosen in the chat answers there only. Evaluation runs use the configuration's own model.</p>
          </section>
        )
      }
    />
  )
}

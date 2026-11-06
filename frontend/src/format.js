export const score = (value) => (value === null || value === undefined ? '–' : Number(value).toFixed(2))

export function percent(value) {
  return value === null || value === undefined ? '–' : `${Math.round(Number(value) * 100)}%`
}

export function duration(ms) {
  if (ms === null || ms === undefined) return '–'
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`
}

export function bytes(size) {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

export function dateTime(value) {
  if (!value) return '–'
  return new Date(value).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function shortDate(value) {
  if (!value) return '–'
  return new Date(value).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
}

/** Format a metric value by its unit: 0–1 scores, milliseconds, tokens, rates. */
export function metricValue(metric, value) {
  if (value === null || value === undefined) return '–'
  switch (metric?.unit) {
    case 'ms':
      return duration(value)
    case 'tokens':
      return `${Math.round(value)} tokens`
    case 'req/s':
      return `${Number(value).toFixed(3)} req/s`
    case 'currency':
      return `₹${Number(value).toFixed(4)}`
    default:
      return score(value)
  }
}

export const runLabel = (run) => run?.name || (run ? `Run ${run.id}` : '')

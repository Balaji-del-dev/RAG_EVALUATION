import { useCallback, useEffect, useRef, useState } from 'react'

function csrfToken() {
  const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/)
  return match ? decodeURIComponent(match[1]) : ''
}

/** Turn a DRF error body into one readable sentence. */
function errorMessage(body, status) {
  if (body?.error) return body.error
  if (body?.detail) return body.detail
  if (body && typeof body === 'object') {
    const [field, problem] = Object.entries(body)[0] ?? []
    if (field) return `${field}: ${Array.isArray(problem) ? problem[0] : problem}`
  }
  return `The server answered with status ${status}.`
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

export async function api(path, { method = 'GET', body, form } = {}) {
  const headers = {}
  if (method !== 'GET') headers['X-CSRFToken'] = csrfToken()
  if (body) headers['Content-Type'] = 'application/json'
  let response
  try {
    response = await fetch(`/api/${path}`, {
      method,
      headers,
      credentials: 'same-origin',
      body: form ?? (body ? JSON.stringify(body) : undefined),
    })
  } catch {
    throw new ApiError('Could not reach the server. Check that it is running.', 0)
  }
  if (response.status === 204) return null
  const isJson = response.headers.get('content-type')?.includes('json')
  const data = isJson ? await response.json() : null
  if (!response.ok) {
    // A session that has ended: the whole app goes back to the sign-in screen, rather than each panel failing.
    if ((response.status === 401 || response.status === 403) && /credentials were not provided/i.test(data?.detail ?? '')) {
      window.dispatchEvent(new Event('bench:session-ended'))
    }
    throw new ApiError(errorMessage(data, response.status), response.status)
  }
  return data
}

/** POST, then read the server-sent events in the response, calling onEvent(name, data) for each. */
export async function streamApi(path, body, onEvent, { signal } = {}) {
  let response
  try {
    response = await fetch(`/api/${path}`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken(), Accept: 'text/event-stream' },
      body: JSON.stringify(body),
      signal,
    })
  } catch (failure) {
    if (failure.name === 'AbortError') throw failure
    throw new ApiError('Could not reach the server. Check that it is running.', 0)
  }
  if (!response.ok) {
    // Errors are JSON, whichever content type they arrive with.
    let data = null
    try {
      data = JSON.parse(await response.text())
    } catch {
      data = null
    }
    throw new ApiError(errorMessage(data, response.status), response.status)
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let end
    while ((end = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, end)
      buffer = buffer.slice(end + 2)
      const name = block.match(/^event: (.*)$/m)?.[1] ?? 'message'
      const data = block.match(/^data: (.*)$/m)?.[1]
      if (data) onEvent(name, JSON.parse(data))
    }
  }
}

export const results = (data) => data?.results ?? data ?? []

/**
 * Fetch on mount and whenever `path` changes. While `pollWhile(data)` is true it refetches every 3 s.
 * The previous data stays on screen during a refetch, so nothing flashes or jumps.
 * `reload` also restarts polling, so a run started after the last one finished is followed too.
 */
export function useApi(path, { pollWhile, skip = false } = {}) {
  const [state, setState] = useState({ data: null, error: null, loading: !skip })
  const [attempt, setAttempt] = useState(0)
  const pollWhileRef = useRef(pollWhile)

  useEffect(() => {
    pollWhileRef.current = pollWhile
  })

  useEffect(() => {
    if (skip || !path) return undefined
    let active = true
    let timer
    const tick = async () => {
      setState((previous) => ({ ...previous, loading: true }))
      try {
        const data = await api(path)
        if (!active) return
        setState({ data, error: null, loading: false })
        if (pollWhileRef.current?.(data)) timer = setTimeout(tick, 3000)
      } catch (error) {
        if (active) setState((previous) => ({ ...previous, error, loading: false }))
      }
    }
    tick()
    return () => {
      active = false
      clearTimeout(timer)
    }
  }, [path, skip, attempt])

  const reload = useCallback(() => setAttempt((count) => count + 1), [])
  return { ...state, reload }
}

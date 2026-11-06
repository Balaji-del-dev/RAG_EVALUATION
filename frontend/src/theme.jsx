import { Check, Monitor, Moon, Sun } from 'lucide-react'
import { createContext, useContext, useEffect, useMemo, useState } from 'react'

const MODE_KEY = 'bench-theme'
const ACCENT_KEY = 'bench-accent'

export const MODES = [
  { id: 'light', name: 'Light', Icon: Sun },
  { id: 'dark', name: 'Dark', Icon: Moon },
  { id: 'system', name: 'System', Icon: Monitor },
]

export const ACCENTS = [
  { id: 'blue', name: 'Blue' },
  { id: 'violet', name: 'Violet' },
  { id: 'emerald', name: 'Emerald' },
  { id: 'rose', name: 'Rose' },
  { id: 'amber', name: 'Amber' },
]

function stored(key, fallback, allowed) {
  try {
    const value = localStorage.getItem(key)
    return allowed.some((item) => item.id === value) ? value : fallback
  } catch {
    return fallback
  }
}

function remember(key, value) {
  try {
    localStorage.setItem(key, value)
  } catch {
    // Private browsing can refuse storage. The choice still applies until the page reloads.
  }
}

const ThemeContext = createContext(null)

export function ThemeProvider({ children }) {
  const [mode, setMode] = useState(() => stored(MODE_KEY, 'light', MODES))
  const [accent, setAccent] = useState(() => stored(ACCENT_KEY, 'blue', ACCENTS))
  const [systemDark, setSystemDark] = useState(() => matchMedia('(prefers-color-scheme: dark)').matches)

  useEffect(() => {
    const query = matchMedia('(prefers-color-scheme: dark)')
    const update = (event) => setSystemDark(event.matches)
    query.addEventListener('change', update)
    return () => query.removeEventListener('change', update)
  }, [])

  const resolved = mode === 'system' ? (systemDark ? 'dark' : 'light') : mode

  useEffect(() => {
    document.documentElement.dataset.theme = resolved
    document.documentElement.dataset.accent = accent
  }, [resolved, accent])

  const value = useMemo(
    () => ({
      mode,
      accent,
      resolved,
      setMode: (next) => {
        setMode(next)
        remember(MODE_KEY, next)
      },
      setAccent: (next) => {
        setAccent(next)
        remember(ACCENT_KEY, next)
      },
    }),
    [mode, accent, resolved],
  )
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export const useTheme = () => useContext(ThemeContext)

/** Choice of light, dark or system, and of the colour used for actions and highlights. */
export function ThemeControls() {
  const { mode, setMode, accent, setAccent } = useTheme()
  const accentName = ACCENTS.find((item) => item.id === accent)?.name
  return (
    <div className="theme-controls">
      <fieldset className="theme-group">
        <legend className="field-label">Theme</legend>
        <div className="segmented">
          {MODES.map(({ id, name, Icon }) => (
            <label className={`segmented-option ${mode === id ? 'segmented-on' : ''}`} data-mode={id} key={id}>
              <input type="radio" className="sr-only" name="theme-mode" checked={mode === id} onChange={() => setMode(id)} />
              <Icon size={16} aria-hidden="true" />
              {name}
            </label>
          ))}
        </div>
      </fieldset>
      <fieldset className="theme-group">
        <legend className="field-label">Colour: {accentName}</legend>
        <div className="swatches">
          {ACCENTS.map(({ id, name }) => (
            <label className={`swatch swatch-${id} ${accent === id ? 'swatch-on' : ''}`} data-accent={id} title={name} key={id}>
              <input
                type="radio"
                className="sr-only"
                name="theme-accent"
                aria-label={name}
                checked={accent === id}
                onChange={() => setAccent(id)}
              />
              {accent === id && <Check size={16} strokeWidth={3} aria-hidden="true" />}
            </label>
          ))}
        </div>
      </fieldset>
    </div>
  )
}

/** One-click light and dark switch, for the sign-in page. */
export function ThemeToggle() {
  const { resolved, setMode } = useTheme()
  const dark = resolved === 'dark'
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={() => setMode(dark ? 'light' : 'dark')}
      aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
      title={dark ? 'Light theme' : 'Dark theme'}
    >
      {dark ? <Sun size={16} aria-hidden="true" /> : <Moon size={16} aria-hidden="true" />}
    </button>
  )
}

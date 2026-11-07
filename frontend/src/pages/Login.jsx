import { Check, Gauge } from 'lucide-react'
import { useState } from 'react'
import { api } from '../api.js'
import { ErrorNote, Field } from '../components/common.jsx'
import { ThemeToggle } from '../theme.jsx'

export default function Login({ onSignedIn, signedOut }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  async function submit(event) {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      onSignedIn(await api('accounts/login/', { method: 'POST', body: { username, password } }))
    } catch (failure) {
      setError(failure)
      setBusy(false)
    }
  }

  return (
    <main className="login">
      <ThemeToggle />
      <div className="login-box">
        <div className="login-brand">
          <span className="rail-mark">
            <Gauge size={16} aria-hidden="true" />
          </span>
          RAG Evaluation Bench
        </div>
        <form className="login-card" onSubmit={submit}>
          {signedOut && (
            <p className="login-notice" role="status">
              <Check size={16} aria-hidden="true" />
              {signedOut === 'expired' ? 'Your session ended. Sign in again to carry on.' : 'You have been signed out.'}
            </p>
          )}
          <h1>Sign in</h1>
          <p className="page-intro">Measure how your RAG pipeline retrieves, answers and holds up under attack.</p>
          <Field label="Username">
            <input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" autoFocus required />
          </Field>
          <Field label="Password">
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
            />
          </Field>
          <ErrorNote error={error} />
          <button className="button button-primary" disabled={busy}>
            {busy ? 'Signing in' : 'Sign in'}
          </button>
        </form>
      </div>
    </main>
  )
}

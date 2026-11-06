import { ChevronsUpDown, LogOut } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { ThemeControls } from '../theme.jsx'

const ROLES = {
  admin: { name: 'Admin', can: 'Can read every document, change settings and see the audit log.' },
  analyst: { name: 'Analyst', can: 'Can upload documents, start runs and read answers.' },
  viewer: { name: 'Viewer', can: 'Can ask questions and see scores.' },
}

const initials = (username) => username.slice(0, 2).toUpperCase()

/** The signed-in user: a button in the sidebar that opens a card with their details, theme and sign out. */
export function ProfileCard({ user, onSignOut }) {
  const [open, setOpen] = useState(false)
  const box = useRef(null)
  const trigger = useRef(null)
  const role = ROLES[user.role] ?? { name: user.role, can: '' }

  useEffect(() => {
    if (!open) return undefined
    const onClick = (event) => {
      if (!box.current.contains(event.target)) setOpen(false)
    }
    const onKey = (event) => {
      if (event.key === 'Escape') {
        setOpen(false)
        trigger.current.focus()
      }
    }
    document.addEventListener('mousedown', onClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div className="profile" ref={box}>
      {open && (
        <section className="profile-card" role="dialog" aria-label="Your profile">
          <div className="profile-head">
            <span className="avatar avatar-large" aria-hidden="true">
              {initials(user.username)}
            </span>
            <div className="profile-who">
              <div className="profile-name">{user.username}</div>
              {user.email && <div className="profile-email">{user.email}</div>}
              <span className="profile-role">{role.name}</span>
            </div>
          </div>
          <p className="profile-can">{role.can}</p>
          <ThemeControls />
          <button type="button" className="button rail-signout" onClick={onSignOut}>
            <LogOut size={16} aria-hidden="true" />
            Sign out
          </button>
        </section>
      )}
      <button
        type="button"
        className="profile-trigger"
        ref={trigger}
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-haspopup="dialog"
      >
        <span className="avatar" aria-hidden="true">
          {initials(user.username)}
        </span>
        <span className="profile-trigger-text">
          <span className="rail-username">{user.username}</span>{' '}
          <span className="rail-role">{role.name}</span>
        </span>
        <ChevronsUpDown className="profile-chevron" size={16} aria-hidden="true" />
      </button>
    </div>
  )
}

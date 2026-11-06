import { AlertTriangle, Check } from 'lucide-react'
import { createContext, useCallback, useContext, useState } from 'react'

const ToastContext = createContext(() => {})
const TOAST_MS = 4000

/** Brief confirmation that an action worked, so that a click is never met with silence. */
export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const show = useCallback((message, kind = 'good') => {
    const id = crypto.randomUUID()
    setToasts((current) => [...current, { id, message, kind }])
    setTimeout(() => setToasts((current) => current.filter((toast) => toast.id !== id)), TOAST_MS)
  }, [])

  return (
    <ToastContext.Provider value={show}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((toast) => (
          <div className="toast" key={toast.id}>
            {toast.kind === 'good' ? <Check size={16} aria-hidden="true" /> : <AlertTriangle size={16} aria-hidden="true" />}
            {toast.message}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export const useToast = () => useContext(ToastContext)

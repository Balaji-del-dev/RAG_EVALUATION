import { FileUp } from 'lucide-react'
import { useState } from 'react'

/** A file picker that also accepts a dropped file. The real input stays, so the keyboard still works.
 * A dropped file skips the picker's own type filter, so the type is checked here too. */
export function FileDrop({ inputRef, accept, hint, file, onFile, wrongTypeHint = '' }) {
  const [over, setOver] = useState(false)
  const [refused, setRefused] = useState('')
  const allowed = accept.split(',').map((extension) => extension.trim().toLowerCase())

  function choose(chosen) {
    if (chosen && !allowed.some((extension) => chosen.name.toLowerCase().endsWith(extension))) {
      setRefused(`${chosen.name} is not a ${hint} file. ${wrongTypeHint}`.trim())
      inputRef.current.value = ''
      onFile(null)
      return
    }
    setRefused('')
    onFile(chosen ?? null)
  }

  function drop(event) {
    event.preventDefault()
    setOver(false)
    const dropped = event.dataTransfer.files
    if (dropped.length) {
      inputRef.current.files = dropped
      choose(dropped[0])
    }
  }

  return (
    <>
      <label
        className={`file-drop ${over ? 'file-drop-over' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={drop}
      >
        <input
          type="file"
          className="sr-only"
          ref={inputRef}
          accept={accept}
          aria-describedby={refused ? 'file-drop-refused' : undefined}
          onChange={(event) => choose(event.target.files[0])}
        />
        <FileUp size={24} aria-hidden="true" />
        <span>
          <span className="file-drop-title">{file ? file.name : 'Drop a file here, or click to choose one'}</span>
          <br />
          <span className="file-drop-hint">{hint}</span>
        </span>
      </label>
      {refused && (
        <p className="file-drop-refused" id="file-drop-refused" role="alert">
          {refused}
        </p>
      )}
    </>
  )
}

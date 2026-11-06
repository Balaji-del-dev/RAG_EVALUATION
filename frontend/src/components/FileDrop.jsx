import { FileUp } from 'lucide-react'
import { useState } from 'react'

/** A file picker that also accepts a dropped file. The real input stays, so the keyboard still works. */
export function FileDrop({ inputRef, accept, hint, file, onFile }) {
  const [over, setOver] = useState(false)

  function drop(event) {
    event.preventDefault()
    setOver(false)
    const dropped = event.dataTransfer.files
    if (dropped.length) {
      inputRef.current.files = dropped
      onFile(dropped[0])
    }
  }

  return (
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
        onChange={(event) => onFile(event.target.files[0] ?? null)}
      />
      <FileUp size={24} aria-hidden="true" />
      <span>
        <span className="file-drop-title">{file ? file.name : 'Drop a file here, or click to choose one'}</span>
        <br />
        <span className="file-drop-hint">{hint}</span>
      </span>
    </label>
  )
}

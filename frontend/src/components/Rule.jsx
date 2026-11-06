import { useState } from 'react'

/**
 * The calibrated rule: every score in the app sits on one of these. An engraved scale with a thin bar
 * rising from zero. With `marks` it carries several dots instead, one per run being compared.
 * The scale is never reversed: "lower is better" is stated in words, because a flipped axis is easy to misread.
 * Each value is always printed beside the rule, so the hover tooltip only adds convenience.
 */
export function Rule({ value, max = 1, marks, label }) {
  const position = (v) => Math.max(0, Math.min(100, (Number(v) / max) * 100))
  const [active, setActive] = useState(null)

  return (
    <div className="rule" role="img" aria-label={label}>
      <div className="rule-track" aria-hidden="true">
        {marks
          ? marks.map((mark) => (
              <span
                key={mark.key}
                className="rule-needle"
                style={{ left: `${position(mark.value)}%`, background: mark.color }}
                onMouseEnter={() => setActive(mark)}
                onMouseLeave={() => setActive(null)}
              />
            ))
          : value !== null &&
            value !== undefined && <span className="rule-fill" style={{ width: `${position(value)}%` }} />}
        {active && (
          <span className="rule-tip" style={{ left: `${position(active.value)}%` }}>
            {active.name}: {active.display}
          </span>
        )}
      </div>
    </div>
  )
}

/** The labels for a group of rules. Shown once above the group rather than on every row. */
export function RuleScale({ max = 1, format = (v) => v }) {
  return (
    <div className="rule-scale" aria-hidden="true">
      {[0, 0.25, 0.5, 0.75, 1].map((step) => (
        <span key={step} style={{ left: `${step * 100}%` }}>
          {format(step * max)}
        </span>
      ))}
    </div>
  )
}

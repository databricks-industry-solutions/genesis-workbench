// Flashing orange dot + "N run(s) in progress" badge.
// Mirrors the Streamlit `run-dot-in-progress` banner used above every
// search-past-runs table (modules/core/app/views/disease_biology.py).

type Props = {
  count: number
  /** Override the noun — defaults to "run". */
  label?: string
}

export function InProgressBadge({ count, label = 'run' }: Props) {
  if (count <= 0) return null
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-amber-500/40 bg-amber-500/10 px-3 py-1 text-xs text-amber-200">
      <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-amber-400" />
      {count} {label}
      {count > 1 ? 's' : ''} in progress
    </span>
  )
}

export function InProgressDot() {
  return (
    <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-amber-400" />
  )
}

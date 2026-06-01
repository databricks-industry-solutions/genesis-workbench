import { cn } from '@/lib/utils'

/**
 * Determinate progress bar driven by real server-side (pct, msg) updates
 * streamed via SSE. Pair with `useSseMutation`.
 *
 * When `stages` is provided, also renders a checklist beside/below the bar.
 * Each stage's `pctEnd` is the upper bound where that stage finishes — the
 * stage with `pct < pctEnd` becomes the active one; earlier stages tick.
 * The boundaries should mirror the pct markers the matching backend
 * service emits (see `progress_callback(...)` call sites).
 */
export type ProgressStage = { label: string; pctEnd: number }

export function RealtimeProgress({
  pct,
  msg,
  title,
  stages,
}: {
  pct: number
  msg: string
  title?: string
  stages?: ProgressStage[]
}) {
  const clamped = Math.max(0, Math.min(100, pct))
  const currentIdx = stages
    ? stages.findIndex((s) => clamped < s.pctEnd)
    : -1
  // If pct >= last stage's pctEnd, all stages are considered done.
  const effectiveIdx = currentIdx === -1 ? (stages?.length ?? 0) : currentIdx

  return (
    <div className="space-y-3 rounded-md border border-border bg-muted/30 p-4">
      {title && <div className="text-sm font-medium">{title}</div>}

      <div
        className="h-2 w-full overflow-hidden rounded-full bg-background"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={clamped}
      >
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-200 ease-out"
          style={{ width: `${clamped}%` }}
        />
      </div>

      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="truncate text-foreground" title={msg}>
          {msg}
        </span>
        <span className="shrink-0 text-muted-foreground">{clamped}%</span>
      </div>

      {stages && stages.length > 1 && (
        <ul className="space-y-1 pt-1 text-xs">
          {stages.map((stage, i) => {
            const state: 'done' | 'active' | 'pending' =
              i < effectiveIdx ? 'done' : i === effectiveIdx ? 'active' : 'pending'
            return (
              <li
                key={i}
                className={cn(
                  'flex items-center gap-2',
                  state === 'done' && 'text-muted-foreground',
                  state === 'active' && 'font-medium text-foreground',
                  state === 'pending' && 'text-muted-foreground/60',
                )}
              >
                <span className="flex h-4 w-4 items-center justify-center">
                  {state === 'done' ? (
                    <span className="text-success">✓</span>
                  ) : state === 'active' ? (
                    <Spinner />
                  ) : (
                    <span className="text-muted-foreground/40">○</span>
                  )}
                </span>
                {stage.label}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

function Spinner() {
  return (
    <span
      aria-hidden
      className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-primary/30 border-t-primary"
    />
  )
}

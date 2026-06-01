import { useEffect, useState } from 'react'

import { cn } from '@/lib/utils'

export type ProgressStage = {
  label: string
  /** Expected wall-clock seconds for this stage. The bar uses the cumulative
   * estimate to advance the indicator; we cap at 95% so the final 5% only
   * fills when the mutation actually resolves and we unmount. */
  estSeconds: number
}

type Props = {
  /** When true, the bar is shown and animates. Toggle off to hide. */
  active: boolean
  /** Ordered stages with per-stage time estimates. Omit for a single
   * indeterminate spinner with an elapsed timer. */
  stages?: ProgressStage[]
  /** Optional header above the stage list. */
  title?: string
  /** Optional one-line note rendered below the elapsed timer (e.g. "Cold-start
   * can take a few minutes"). */
  note?: string
}

/**
 * Staged progress indicator for synchronous workflows.
 * Time-based: advances through the listed stages on a clock, not from real
 * server-side events — stages transition on a timer with a determinate bar
 * so the user sees visible motion during the wait.
 */
export function WorkflowProgress({ active, stages, title, note }: Props) {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    if (!active) {
      setElapsed(0)
      return
    }
    const start = Date.now()
    const id = setInterval(() => setElapsed((Date.now() - start) / 1000), 200)
    return () => clearInterval(id)
  }, [active])

  if (!active) return null

  if (!stages || stages.length === 0) {
    return (
      <div className="flex items-center gap-3 rounded-md border border-border bg-muted/30 p-4 text-sm">
        <Spinner />
        <span className="text-foreground">{title ?? 'Working…'}</span>
        <span className="ml-auto text-xs text-muted-foreground">{formatElapsed(elapsed)}</span>
      </div>
    )
  }

  const totalEst = stages.reduce((s, x) => s + x.estSeconds, 0)
  const rawPct = Math.min((elapsed / totalEst) * 100, 95)

  let accum = 0
  let currentIdx = stages.length // past-the-end means "still finalising"
  for (let i = 0; i < stages.length; i++) {
    if (elapsed < accum + stages[i].estSeconds) {
      currentIdx = i
      break
    }
    accum += stages[i].estSeconds
  }
  const exhausted = currentIdx >= stages.length

  return (
    <div className="space-y-3 rounded-md border border-border bg-muted/30 p-4">
      {title && <div className="text-sm font-medium">{title}</div>}

      <div
        className={cn(
          'h-2 w-full overflow-hidden rounded-full bg-background',
          exhausted && 'relative',
        )}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={exhausted ? undefined : Math.round(rawPct)}
      >
        {exhausted ? (
          <div className="h-full w-1/3 animate-progress-indeterminate rounded-full bg-primary" />
        ) : (
          <div
            className="h-full rounded-full bg-primary transition-[width] duration-300 ease-linear"
            style={{ width: `${rawPct}%` }}
          />
        )}
      </div>

      <ul className="space-y-1 text-xs">
        {stages.map((s, i) => {
          const state: 'done' | 'active' | 'pending' =
            i < currentIdx ? 'done' : i === currentIdx ? 'active' : 'pending'
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
                  <Spinner small />
                ) : (
                  <span className="text-muted-foreground/40">○</span>
                )}
              </span>
              {s.label}
            </li>
          )
        })}
        {exhausted && (
          <li className="flex items-center gap-2 font-medium text-foreground">
            <span className="flex h-4 w-4 items-center justify-center">
              <Spinner small />
            </span>
            Waiting for server response…
          </li>
        )}
      </ul>

      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{note ?? ''}</span>
        <span>{formatElapsed(elapsed)}</span>
      </div>
    </div>
  )
}

function Spinner({ small = false }: { small?: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        'inline-block animate-spin rounded-full border-2 border-primary/30 border-t-primary',
        small ? 'h-3 w-3' : 'h-4 w-4',
      )}
    />
  )
}

function formatElapsed(seconds: number): string {
  const s = Math.floor(seconds)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const r = s % 60
  return `${m}m ${r.toString().padStart(2, '0')}s`
}

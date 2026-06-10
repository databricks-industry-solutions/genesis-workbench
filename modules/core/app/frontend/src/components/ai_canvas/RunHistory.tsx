// Vortex (ai_canvas) — browse past runs and inspect their results.
import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { Dialog } from '@/components/Dialog'
import { cn } from '@/lib/utils'

const STATUS_BADGE: Record<string, string> = {
  submitted: 'bg-muted text-muted-foreground',
  started: 'bg-amber-500/15 text-amber-600',
  running: 'bg-amber-500/15 text-amber-600',
  complete: 'bg-emerald-500/15 text-emerald-600',
  failed: 'bg-destructive/15 text-destructive',
}

export function RunHistory() {
  const [open, setOpen] = useState(false)
  const [text, setText] = useState('')
  const [page, setPage] = useState(1)
  const [resultRunId, setResultRunId] = useState<string | null>(null)

  const runs = useQuery({
    queryKey: ['ai_canvas', 'runs', text, page],
    queryFn: () => api.aiCanvasRuns(text, page),
    enabled: open,
  })

  const result = useMutation({
    mutationFn: (runId: string) => api.aiCanvasRunResult(runId),
  })

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-accent"
      >
        Past runs
      </button>

      <Dialog open={open} onClose={() => setOpen(false)} title="Past runs" width="max-w-2xl">
        <input
          value={text}
          onChange={(e) => {
            setText(e.target.value)
            setPage(1) // a new filter restarts from the first page
          }}
          placeholder="Filter by run name…"
          className="mb-3 w-full rounded-md border border-border bg-background px-2 py-1 text-sm"
        />

        {/* Status is read fresh from MLflow on each fetch — show a spinner while
            that's in flight (initial load, paging, or refetch). */}
        {runs.isFetching && (
          <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
            <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-blue-500/30 border-t-blue-500" />
            Fetching latest status…
          </div>
        )}
        {runs.isLoading ? (
          <p className="text-sm text-muted-foreground">Loading…</p>
        ) : !runs.data || runs.data.runs.length === 0 ? (
          <p className="text-sm text-muted-foreground">No runs yet.</p>
        ) : (
          <ul className="divide-y divide-border">
            {runs.data.runs.map((r) => (
              <li key={r.run_id} className="flex items-center gap-3 py-2">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium">{r.run_name || r.run_id}</div>
                  <div className="text-[10px] text-muted-foreground">
                    {r.node_count ?? '?'} nodes · {r.start_time}
                  </div>
                </div>
                <span
                  className={cn(
                    'rounded px-1.5 py-0.5 text-[10px] font-medium',
                    STATUS_BADGE[r.job_status] ?? 'bg-muted text-muted-foreground',
                  )}
                >
                  {r.job_status || 'unknown'}
                </span>
                {r.run_url && (
                  <a
                    href={r.run_url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-[11px] text-primary hover:underline"
                  >
                    Job ↗
                  </a>
                )}
                <button
                  onClick={() => {
                    setResultRunId(r.run_id)
                    result.mutate(r.run_id)
                  }}
                  disabled={!['complete', 'failed'].includes(r.job_status)}
                  className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-accent disabled:opacity-40"
                >
                  Result
                </button>
              </li>
            ))}
          </ul>
        )}

        {/* Pager — 20 most-recent per page. */}
        {(page > 1 || runs.data?.has_more) && (
          <div className="mt-3 flex items-center justify-between border-t border-border pt-2 text-xs">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1 || runs.isFetching}
              className="rounded-md border border-border px-2.5 py-1 hover:bg-accent disabled:opacity-40"
            >
              ← Newer
            </button>
            <span className="text-muted-foreground">Page {page}</span>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={!runs.data?.has_more || runs.isFetching}
              className="rounded-md border border-border px-2.5 py-1 hover:bg-accent disabled:opacity-40"
            >
              Older →
            </button>
          </div>
        )}
      </Dialog>

      <Dialog
        open={resultRunId !== null}
        onClose={() => setResultRunId(null)}
        title="Workflow result"
        width="max-w-2xl"
      >
        {result.isPending ? (
          <p className="text-sm text-muted-foreground">Loading result…</p>
        ) : result.data && Object.keys(result.data.result).length > 0 ? (
          <pre className="max-h-[60vh] overflow-auto rounded-md border border-border bg-muted/30 p-3 text-xs">
            {JSON.stringify(result.data.result, null, 2)}
          </pre>
        ) : (
          <p className="text-sm text-muted-foreground">
            No result artifact yet — the run may still be in progress or produced no output.
          </p>
        )}
      </Dialog>
    </>
  )
}

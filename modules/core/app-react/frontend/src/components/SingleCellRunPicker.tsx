import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import type { SingleCellRun } from '@/types/api'

type Props = {
  /** Selected run_id (controlled). */
  value: string | null
  onChange: (run: SingleCellRun | null) => void
  /** Statuses to include. Defaults to ['complete', 'finished'] — annotation
   * needs a successful run; in-flight runs have no markers_flat artifact. */
  acceptStatuses?: string[]
}

const DEFAULT_ACCEPT = ['complete', 'finished']

export function SingleCellRunPicker({ value, onChange, acceptStatuses = DEFAULT_ACCEPT }: Props) {
  const q = useQuery({
    queryKey: ['single_cell', 'runs'],
    queryFn: api.singleCellRuns,
    staleTime: 30_000,
  })

  const completed = useMemo(() => {
    if (!q.data) return []
    return q.data.runs.filter((r) =>
      acceptStatuses.includes(r.status.toLowerCase()),
    )
  }, [q.data, acceptStatuses])

  if (q.isLoading) {
    return <div className="text-sm text-muted-foreground">Loading runs…</div>
  }
  if (q.error) {
    return (
      <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
        {String(q.error)}
      </div>
    )
  }
  if (completed.length === 0) {
    return (
      <div className="rounded-md border border-border bg-muted/30 p-3 text-sm text-muted-foreground">
        No completed single-cell processing runs found for your account. Run an analysis in the
        Streamlit app or once the Raw Single Cell Processing tab ports here.
      </div>
    )
  }

  return (
    <label className="block text-xs">
      <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
        Completed processing run
      </span>
      <select
        value={value ?? ''}
        onChange={(e) => {
          const sel = completed.find((r) => r.run_id === e.target.value) ?? null
          onChange(sel)
        }}
        className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
      >
        <option value="">— pick a run —</option>
        {completed.map((r) => (
          <option key={r.run_id} value={r.run_id}>
            {r.run_name} ({r.experiment_name}, {r.processing_mode}) —{' '}
            {r.start_time_ms ? new Date(r.start_time_ms).toLocaleString() : '?'}
            {r.cells ? ` · ${r.cells.toLocaleString()} cells` : ''}
          </option>
        ))}
      </select>
    </label>
  )
}

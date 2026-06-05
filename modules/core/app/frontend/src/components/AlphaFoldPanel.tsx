// AlphaFold "Search Past Runs" + result viewer — rendered as the right panel of
// the Structure Prediction tab when the AlphaFold model is selected (AlphaFold
// is async, so you fold via the shared input panel then find the result here).
import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { Dialog } from '@/components/Dialog'
import { InProgressBadge } from '@/components/InProgressBadge'
import { MolstarViewer } from '@/components/MolstarViewer'
import type { AlphaFoldRun } from '@/types/api'
import { cn } from '@/lib/utils'

type SearchMode = 'experiment_name' | 'run_name'

function statusBadge(status: string): string {
  if (status === 'fold_complete') return '🟢 fold_complete'
  if (status === 'started' || status === 'running') return '🟡 ' + status
  if (status === 'failed' || status.startsWith('error')) return '🔴 ' + status
  return '⚪ ' + status
}

export function AlphaFoldSearchResults() {
  const [searchMode, setSearchMode] = useState<SearchMode>('experiment_name')
  const [searchText, setSearchText] = useState('alphafold')
  const [searchedAt, setSearchedAt] = useState<number>(0)
  const [viewing, setViewing] = useState<AlphaFoldRun | null>(null)

  const search = useQuery({
    queryKey: ['alphafold', 'search', searchMode, searchText, searchedAt],
    queryFn: () => api.alphafoldSearch(searchMode, searchText),
    enabled: searchedAt > 0,
  })

  const columns = useMemo<ColumnDef<AlphaFoldRun, unknown>[]>(
    () => [
      {
        id: 'run_name',
        header: 'Run Name',
        cell: (ctx) => {
          const r = ctx.row.original
          return r.run_url ? (
            <a
              href={r.run_url}
              target="_blank"
              rel="noreferrer"
              className="text-primary hover:underline"
              title="Open Databricks run page"
            >
              {r.run_name}
            </a>
          ) : (
            r.run_name
          )
        },
      },
      { id: 'experiment_name', header: 'Experiment', accessorKey: 'experiment_name' },
      {
        id: 'protein_sequence',
        header: 'Sequence',
        cell: (ctx) => {
          const s = ctx.row.original.protein_sequence
          return s.length > 40 ? `${s.slice(0, 40)}…` : s
        },
      },
      {
        id: 'start',
        header: 'Start',
        cell: (ctx) =>
          ctx.row.original.start_time_ms
            ? new Date(ctx.row.original.start_time_ms).toLocaleString()
            : '',
      },
      {
        id: 'status',
        header: 'Status',
        cell: (ctx) => statusBadge(ctx.row.original.status),
      },
      {
        id: 'view',
        header: '',
        cell: (ctx) => {
          const r = ctx.row.original
          const ready = r.status === 'fold_complete'
          return (
            <button
              onClick={() => setViewing(r)}
              disabled={!ready}
              title={ready ? 'View predicted structure' : 'Available once status is fold_complete'}
              className={cn(
                'rounded-md border px-3 py-1 text-xs transition-colors',
                ready
                  ? 'border-primary bg-primary/10 text-primary hover:bg-primary/20'
                  : 'cursor-not-allowed border-border text-muted-foreground opacity-50',
              )}
            >
              View
            </button>
          )
        },
      },
    ],
    [],
  )

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <h4 className="text-sm font-medium">Search Past Runs</h4>
        <InProgressBadge
          count={
            (search.data?.runs ?? []).filter(
              (r) => r.status === 'started' || r.status === 'running',
            ).length
          }
        />
      </div>
      <p className="text-xs text-muted-foreground">
        AlphaFold runs asynchronously (minutes–hours). Start a job from the panel on the left, then
        find it here once it completes.
      </p>
      <div className="flex flex-wrap items-end gap-3">
        <div className="flex gap-1">
          {(['experiment_name', 'run_name'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setSearchMode(m)}
              className={cn(
                'rounded-md border px-3 py-2 text-sm transition-colors',
                m === searchMode
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border text-muted-foreground hover:bg-accent',
              )}
            >
              {m === 'experiment_name' ? 'Experiment' : 'Run name'}
            </button>
          ))}
        </div>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            {searchMode === 'experiment_name' ? 'Experiment contains' : 'Run name contains'}
          </span>
          <input
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            className="w-56 rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
        </label>
        <button
          onClick={() => setSearchedAt(Date.now())}
          disabled={searchText.trim().length === 0}
          className="rounded-md border border-border px-3 py-2 text-sm hover:bg-accent disabled:opacity-50"
        >
          Search
        </button>
      </div>

      {search.isLoading && <div className="text-sm text-muted-foreground">Searching…</div>}
      {search.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(search.error)}
        </div>
      )}
      {search.data &&
        (search.data.runs.length === 0 ? (
          <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
            No results.
          </div>
        ) : (
          <DataTable columns={columns} data={search.data.runs} />
        ))}

      <Dialog
        open={!!viewing}
        onClose={() => setViewing(null)}
        title={`AlphaFold2 result — ${viewing?.run_name ?? ''}`}
        width="max-w-4xl"
      >
        {viewing && <ViewerLoader runId={viewing.run_id} runName={viewing.run_name} />}
      </Dialog>
    </div>
  )
}

function ViewerLoader({ runId, runName }: { runId: string; runName: string }) {
  const q = useQuery({
    queryKey: ['alphafold', 'result', runId, runName],
    queryFn: () => api.alphafoldResult(runId, runName),
    staleTime: 60_000,
  })

  if (q.isLoading) return <div className="text-sm text-muted-foreground">Fetching result…</div>
  if (q.error)
    return (
      <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
        {String(q.error)}
      </div>
    )
  return <MolstarViewer viewerHtml={q.data?.viewer_html ?? null} height={620} />
}

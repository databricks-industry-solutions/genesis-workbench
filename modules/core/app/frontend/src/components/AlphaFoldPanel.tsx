import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { Dialog } from '@/components/Dialog'
import { InProgressBadge } from '@/components/InProgressBadge'
import { MolstarViewer } from '@/components/MolstarViewer'
import type { AlphaFoldRun } from '@/types/api'
import { cn } from '@/lib/utils'

const DEFAULT_SEQUENCE =
  'QVQLVESGGGLVQAGGSLRLACIASGRTFHSYVMAWFRQAPGKEREFVAAISWSSTPTYYGESVKGRFTISRDNAKNTVYLQMNRLKPEDTAVYFCAADRGESYYYTRPTEYEFWGQGTQVTVSS'

type SearchMode = 'experiment_name' | 'run_name'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${(d.getMonth() + 1).toString().padStart(2, '0')}${d.getDate().toString().padStart(2, '0')}_${d.getHours().toString().padStart(2, '0')}${d.getMinutes().toString().padStart(2, '0')}`
}

function statusBadge(status: string): string {
  if (status === 'fold_complete') return '🟢 fold_complete'
  if (status === 'started' || status === 'running') return '🟡 ' + status
  if (status === 'failed' || status.startsWith('error')) return '🔴 ' + status
  return '⚪ ' + status
}

export function AlphaFoldPanel() {
  const qc = useQueryClient()
  const [sequence, setSequence] = useState(DEFAULT_SEQUENCE)
  const [expName, setExpName] = useState('alphafold_structure_prediction')
  const [runName, setRunName] = useState(`alphafold_${ts()}`)

  const [searchMode, setSearchMode] = useState<SearchMode>('experiment_name')
  const [searchText, setSearchText] = useState('alphafold')
  const [searchedAt, setSearchedAt] = useState<number>(0)

  const [viewing, setViewing] = useState<AlphaFoldRun | null>(null)

  const start = useMutation({
    mutationFn: () =>
      api.alphafoldStart({
        sequence,
        experiment_name: expName,
        run_name: runName,
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['alphafold', 'search'] }),
  })

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
    <div className="space-y-6">
      <section className="space-y-3">
        <h4 className="text-sm font-medium">Start AlphaFold2 job</h4>
        <p className="text-xs text-muted-foreground">
          Runs the MSA + template search, then folds. Takes minutes to hours depending on sequence
          and infrastructure. Results show up in Search Past Runs below.
        </p>
        <textarea
          rows={3}
          value={sequence}
          onChange={(e) => setSequence(e.target.value)}
          placeholder="Protein sequence"
          className="w-full rounded-md border border-border bg-background p-3 font-mono text-xs"
        />
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              MLflow Experiment
            </span>
            <input
              value={expName}
              onChange={(e) => setExpName(e.target.value)}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
          </label>
          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Run Name
            </span>
            <input
              value={runName}
              onChange={(e) => setRunName(e.target.value)}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
          </label>
          <div className="flex items-end">
            <button
              onClick={() => start.mutate()}
              disabled={
                start.isPending ||
                sequence.trim().length === 0 ||
                expName.trim().length === 0 ||
                runName.trim().length === 0
              }
              className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
            >
              {start.isPending ? 'Starting…' : 'Start Job'}
            </button>
          </div>
        </div>
        {start.data && (
          <div className="rounded-md border border-success/40 bg-success/10 p-3 text-sm">
            Job started — run id <code>{start.data.job_run_id}</code>.
          </div>
        )}
        {start.error && (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            {String(start.error)}
          </div>
        )}
      </section>

      <section className="space-y-3 border-t border-border pt-6">
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
              className="w-72 rounded-md border border-border bg-background px-3 py-2 text-sm"
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

        {search.isLoading && (
          <div className="text-sm text-muted-foreground">Searching…</div>
        )}
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
      </section>

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

import { useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { DataTable } from '@/components/DataTable'
import { Dialog } from '@/components/Dialog'
import { InProgressBadge } from '@/components/InProgressBadge'
import type { DBRunRow, DBSearchResponse } from '@/types/api'
import { cn } from '@/lib/utils'

// Disease Biology workflow statuses that mean "the job is still running"
// (mirrors _IN_PROGRESS in backend/app/services/disease_biology.py).
const DB_IN_PROGRESS_STATUSES = new Set(['started', 'phenotype_prepared'])

/**
 * Generic "Search Past Runs" block for the Disease Biology tabs. Each tab
 * supplies its own searchFn + detail-column label + result-dialog renderer.
 * View button is disabled until the run reaches one of the `viewable_statuses`.
 */
type Props = {
  searchKey: readonly unknown[]
  searchFn: (by: 'run_name' | 'experiment_name', text: string) => Promise<DBSearchResponse>
  detailLabel: string
  initialText?: string
  viewableStatuses: string[]
  renderDialog: (run: DBRunRow) => React.ReactNode
  runUrlFor?: (jobRunId: string) => string
}

export function RunSearchSection({
  searchKey,
  searchFn,
  detailLabel,
  initialText = '',
  viewableStatuses,
  renderDialog,
}: Props) {
  const qc = useQueryClient()
  const [mode, setMode] = useState<'run_name' | 'experiment_name'>('run_name')
  const [text, setText] = useState(initialText)
  const [rows, setRows] = useState<DBRunRow[]>([])
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [viewing, setViewing] = useState<DBRunRow | null>(null)

  const runSearch = async () => {
    setSearching(true)
    setError(null)
    try {
      const data = await qc.fetchQuery({
        queryKey: [...searchKey, mode, text],
        queryFn: () => searchFn(mode, text),
      })
      setRows(data.runs)
    } catch (e) {
      setError(e as Error)
    } finally {
      setSearching(false)
    }
  }

  const columns = useMemo<ColumnDef<DBRunRow, unknown>[]>(
    () => [
      {
        id: 'run_name',
        header: 'Run',
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
        id: 'detail',
        header: detailLabel,
        cell: (ctx) => <PathCell value={ctx.row.original.detail} />,
        meta: { thClass: 'min-w-[280px]', tdClass: 'whitespace-normal' },
      },
      {
        id: 'start_time',
        header: 'Started',
        cell: (ctx) =>
          ctx.row.original.start_time_ms
            ? new Date(ctx.row.original.start_time_ms).toLocaleString()
            : '',
      },
      { id: 'status', header: 'Status', accessorKey: 'status' },
      { id: 'progress', header: 'Progress', accessorKey: 'progress' },
      {
        id: 'view',
        header: '',
        cell: (ctx) => {
          const r = ctx.row.original
          const viewable = viewableStatuses.includes(r.status)
          return (
            <button
              type="button"
              onClick={() => setViewing(r)}
              disabled={!viewable}
              title={
                viewable
                  ? 'View this run'
                  : `View enabled once status is one of: ${viewableStatuses.join(', ')}`
              }
              className={cn(
                'rounded-md border px-3 py-1 text-xs',
                viewable
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
    [detailLabel, viewableStatuses],
  )

  const inProgressCount = rows.filter((r) =>
    DB_IN_PROGRESS_STATUSES.has(r.status),
  ).length

  return (
    <section className="space-y-3 border-t border-border pt-4">
      <div className="flex items-baseline justify-between">
        <h4 className="text-sm font-medium">Search Past Runs</h4>
        <InProgressBadge count={inProgressCount} />
      </div>
      <div className="flex flex-wrap items-end gap-3">
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Search by
          </span>
          <div className="flex gap-1">
            {(['run_name', 'experiment_name'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={cn(
                  'rounded-md border px-3 py-2 text-sm transition-colors',
                  mode === m
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border text-muted-foreground hover:bg-accent',
                )}
              >
                {m === 'run_name' ? 'Run name' : 'Experiment name'}
              </button>
            ))}
          </div>
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Contains
          </span>
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="(e.g. 2026)"
            className="w-64 rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
        </label>
        <button
          type="button"
          onClick={runSearch}
          disabled={searching || !text.trim()}
          className="rounded-md border border-border px-3 py-2 text-sm hover:bg-accent disabled:opacity-50"
        >
          {searching ? 'Searching…' : 'Search'}
        </button>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(error)}
        </div>
      )}
      {!searching && !error && rows.length === 0 && (
        <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
          No runs to show. Hit Search after typing a substring.
        </div>
      )}
      {rows.length > 0 && <DataTable columns={columns} data={rows} />}

      <Dialog
        open={!!viewing}
        onClose={() => setViewing(null)}
        title={viewing ? `Run: ${viewing.run_name}` : ''}
        width="max-w-5xl"
      >
        {viewing && renderDialog(viewing)}
      </Dialog>
    </section>
  )
}

/**
 * Renders a path-like value across multiple labelled lines so long UC volume
 * paths or fully-qualified tables don't blow out the table column width.
 *
 *   /Volumes/<catalog>/<schema>/<volume>/...filename  →
 *     catalog:  <catalog>
 *     schema:   <schema>
 *     volume:   <volume>
 *     filename: <filename>
 *
 *   <catalog>.<schema>.<table>  →
 *     catalog: <catalog>
 *     schema:  <schema>
 *     table:   <table>
 *
 * Non-path strings fall back to a plain line.
 */
function PathCell({ value }: { value: string }) {
  if (!value) return <span className="text-muted-foreground">—</span>

  // UC volume path
  if (value.startsWith('/Volumes/')) {
    const segs = value.slice('/Volumes/'.length).split('/').filter(Boolean)
    if (segs.length >= 4) {
      const [catalog, schema, volume, ...rest] = segs
      const filename = rest.join('/')
      return (
        <PathLines
          rows={[
            ['catalog', catalog],
            ['schema', schema],
            ['volume', volume],
            ['filename', filename],
          ]}
        />
      )
    }
  }

  // catalog.schema.table — three-part identifier (no slashes, two dots).
  if (!value.includes('/') && value.split('.').length === 3) {
    const [catalog, schema, table] = value.split('.')
    return (
      <PathLines
        rows={[
          ['catalog', catalog],
          ['schema', schema],
          ['table', table],
        ]}
      />
    )
  }

  return <span className="break-all font-mono text-[10px]">{value}</span>
}

function PathLines({ rows }: { rows: [string, string][] }) {
  return (
    <div className="space-y-0.5 font-mono text-[10px] leading-tight">
      {rows.map(([k, v]) => (
        <div key={k} className="flex gap-2">
          <span className="w-16 shrink-0 text-muted-foreground">{k}:</span>
          <span className="break-all">{v}</span>
        </div>
      ))}
    </div>
  )
}

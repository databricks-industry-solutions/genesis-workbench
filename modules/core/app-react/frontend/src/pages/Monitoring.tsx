import { useMemo, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { Tabs } from '@/components/Tabs'
import type { WorkflowRun } from '@/types/api'
import { cn } from '@/lib/utils'

const LOOKBACK_OPTIONS = [7, 15, 30]

function statusLabel(r: WorkflowRun): string {
  if (r.lifecycle_state === 'RUNNING') return '🟡 Running'
  if (r.result_state === 'SUCCESS') return '🟢 Success'
  if (r.result_state === 'FAILED') return '🔴 Failed'
  if (r.lifecycle_state === 'PENDING') return '🟡 Pending'
  if (r.lifecycle_state === 'TERMINATED' && !r.result_state) return '⚪ Terminated'
  return `⚪ ${r.result_state ?? r.lifecycle_state ?? ''}`
}

function formatDuration(startMs: number | null, endMs: number | null): string {
  if (!startMs) return ''
  if (!endMs) return 'In Progress'
  const seconds = Math.floor((endMs - startMs) / 1000)
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  return `${h}h ${m}m`
}

function formatTime(ms: number | null): string {
  return ms ? new Date(ms).toLocaleString() : ''
}

function WorkflowRunsTab() {
  const qc = useQueryClient()
  const [daysBack, setDaysBack] = useState<number>(7)
  const [selectedJob, setSelectedJob] = useState<string | null>(null)

  const runs = useQuery({
    queryKey: ['monitoring', 'runs', daysBack],
    queryFn: () => api.monitoringRuns(daysBack),
  })

  const jobNames = useMemo(() => {
    if (!runs.data) return []
    return Array.from(new Set(runs.data.runs.map((r) => r.job_name)))
  }, [runs.data])

  const activeJob = selectedJob ?? jobNames[0] ?? null

  const filtered = useMemo(() => {
    if (!runs.data || !activeJob) return []
    return runs.data.runs.filter((r) => r.job_name === activeJob)
  }, [runs.data, activeJob])

  const columns = useMemo<ColumnDef<WorkflowRun, unknown>[]>(
    () => [
      {
        id: 'run_id',
        header: 'Run ID',
        cell: (ctx) => (
          <a
            href={ctx.row.original.run_url}
            target="_blank"
            rel="noreferrer"
            className="text-primary hover:underline"
          >
            {ctx.row.original.run_id}
          </a>
        ),
      },
      { id: 'job_name', header: 'Job', accessorKey: 'job_name' },
      {
        id: 'status',
        header: 'Status',
        cell: (ctx) => statusLabel(ctx.row.original),
      },
      {
        id: 'start',
        header: 'Start Time',
        cell: (ctx) => formatTime(ctx.row.original.start_time_ms),
      },
      {
        id: 'end',
        header: 'End Time',
        cell: (ctx) => formatTime(ctx.row.original.end_time_ms),
      },
      {
        id: 'duration',
        header: 'Duration',
        cell: (ctx) => formatDuration(ctx.row.original.start_time_ms, ctx.row.original.end_time_ms),
      },
      { id: 'creator', header: 'Created By', accessorKey: 'creator_user_name' },
    ],
    [],
  )

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className="flex gap-1">
          {LOOKBACK_OPTIONS.map((d) => (
            <button
              key={d}
              onClick={() => setDaysBack(d)}
              className={cn(
                'rounded-md border px-3 py-1.5 text-xs transition-colors',
                d === daysBack
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border text-muted-foreground hover:bg-accent',
              )}
            >
              {d} Days
            </button>
          ))}
        </div>
        <button
          onClick={() => qc.invalidateQueries({ queryKey: ['monitoring', 'runs'] })}
          className="rounded-md border border-border px-3 py-1.5 text-xs hover:bg-accent"
        >
          Refresh
        </button>
      </div>

      {runs.isLoading ? (
        <div className="text-sm text-muted-foreground">Fetching workflow runs…</div>
      ) : runs.error ? (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(runs.error)}
        </div>
      ) : jobNames.length === 0 ? (
        <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
          No workflow runs found.
        </div>
      ) : (
        <>
          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Filter by job
            </span>
            <select
              value={activeJob ?? ''}
              onChange={(e) => setSelectedJob(e.target.value)}
              className="w-full max-w-md rounded-md border border-border bg-background px-3 py-2 text-sm"
            >
              {jobNames.map((j) => (
                <option key={j} value={j}>
                  {j}
                </option>
              ))}
            </select>
          </label>
          <DataTable columns={columns} data={filtered} emptyText="No runs for the selected job." />
        </>
      )}
    </div>
  )
}

function DashboardTab() {
  const q = useQuery({ queryKey: ['monitoring', 'admin-dashboard'], queryFn: api.adminDashboard })

  if (q.isLoading) return <div className="text-sm text-muted-foreground">Loading…</div>
  if (q.error || !q.data?.embed_url)
    return (
      <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
        Admin usage dashboard not configured (ADMIN_USAGE_DASHBOARD_ID env var not set).
      </div>
    )

  return (
    <iframe
      title="Admin usage dashboard"
      src={q.data.embed_url}
      className="h-[1200px] w-full rounded-md border border-border bg-background"
    />
  )
}

export function MonitoringPage() {
  return (
    <div className="space-y-6 px-8 py-8">
      <header>
        <h1 className="text-2xl font-semibold">Monitoring and Alerts</h1>
      </header>
      <Tabs
        tabs={[
          { id: 'runs', label: 'Workflow Runs', content: <WorkflowRunsTab /> },
          { id: 'dashboard', label: 'Dashboard', content: <DashboardTab /> },
          {
            id: 'alerts',
            label: 'Alerts',
            content: (
              <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
                Alerts functionality coming soon.
              </div>
            ),
          },
        ]}
      />
    </div>
  )
}

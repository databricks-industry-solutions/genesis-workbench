import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { InProgressDot } from '@/components/InProgressBadge'
import { Tabs } from '@/components/Tabs'
import { useThemeStore, type Theme } from '@/stores/theme'
import { useUserStore } from '@/stores/user'
import type {
  BatchModelRow,
  EndpointRow,
  SettingRow,
} from '@/types/api'
import { cn } from '@/lib/utils'

function titleCase(s: string): string {
  return s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function GeneralTab() {
  const q = useQuery({ queryKey: ['settings', 'system'], queryFn: api.systemSettings })

  const settingColumns = useMemo<ColumnDef<SettingRow, unknown>[]>(
    () => [
      { id: 'setting', header: 'Setting', accessorFn: (r) => titleCase(r.key) },
      { id: 'value', header: 'Value', accessorKey: 'value' },
      { id: 'module', header: 'Module', accessorFn: (r) => titleCase(r.module) },
    ],
    [],
  )

  const workflowColumns = useMemo<ColumnDef<SettingRow, unknown>[]>(
    () => [
      {
        id: 'workflow',
        header: 'Workflow',
        accessorFn: (r) => titleCase(r.key.replace(/_job_id$/, '')),
      },
      { id: 'value', header: 'Job ID', accessorKey: 'value' },
      { id: 'module', header: 'Module', accessorFn: (r) => titleCase(r.module) },
    ],
    [],
  )

  if (q.isLoading) return <Loading />
  if (q.error || !q.data) return <ErrorBlock error={q.error} />

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <ReadField label="Catalog" value={q.data.catalog} />
        <ReadField label="Schema" value={q.data.schema_name} />
        <ReadField label="Warehouse" value={q.data.warehouse_id} />
      </div>

      <AppearanceSection />

      <Tabs
        tabs={[
          {
            id: 'settings',
            label: 'Settings',
            content: (
              <DataTable
                columns={settingColumns}
                data={q.data.settings}
                emptyText="No additional settings found."
              />
            ),
          },
          {
            id: 'workflows',
            label: 'Registered Workflows',
            content: (
              <DataTable
                columns={workflowColumns}
                data={q.data.workflows}
                emptyText="No workflows registered yet. Deploy modules to register workflows."
              />
            ),
          },
        ]}
      />
    </div>
  )
}

function EndpointTab() {
  const qc = useQueryClient()
  const list = useQuery({ queryKey: ['settings', 'endpoints'], queryFn: api.endpointStatuses })
  const status = useQuery({
    queryKey: ['settings', 'start-endpoints', 'status'],
    queryFn: api.startEndpointsStatus,
    refetchInterval: 15_000,
  })

  const [numHours, setNumHours] = useState(4)
  const trigger = useMutation({
    mutationFn: () => api.startEndpointsTrigger(numHours),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['settings', 'start-endpoints', 'status'] })
    },
  })

  const columns = useMemo<ColumnDef<EndpointRow, unknown>[]>(
    () => [
      { id: 'deployment', header: 'Deployment', accessorKey: 'deployment' },
      { id: 'endpoint', header: 'Endpoint', accessorKey: 'endpoint' },
      { id: 'model', header: 'Model', accessorKey: 'model' },
      { id: 'status', header: 'Status', accessorKey: 'status' },
    ],
    [],
  )

  return (
    <div className="space-y-6">
      <section>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-semibold">
            Deployed Endpoints
            {list.data && (
              <span className="ml-2 rounded-full border border-border bg-muted px-2 py-0.5 text-[11px] font-normal text-muted-foreground">
                {list.data.endpoints.length}
              </span>
            )}
          </h3>
          <button
            onClick={() => qc.invalidateQueries({ queryKey: ['settings', 'endpoints'] })}
            className="rounded-md border border-border px-3 py-1.5 text-xs hover:bg-accent"
          >
            Refresh
          </button>
        </div>
        {list.isLoading ? (
          <Loading />
        ) : list.error || !list.data ? (
          <ErrorBlock error={list.error} />
        ) : (
          <DataTable
            columns={columns}
            data={list.data.endpoints}
            emptyText="No active endpoints deployed yet."
          />
        )}
      </section>

      <section className="border-t border-border pt-6">
        <h3 className="text-sm font-semibold">Start All Endpoints</h3>
        <p className="mt-1 text-xs text-muted-foreground">
          Launches a background job that pings each endpoint with sample data, preventing
          scale-to-zero for the selected duration.
        </p>

        {status.data?.active ? (
          <div className="mt-4 rounded-md border border-warning bg-warning/10 p-4 text-sm">
            <div className="flex items-center gap-2 font-semibold">
              <InProgressDot />
              Keep-alive job is running (Run ID: {status.data.run_id})
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-muted-foreground">Started:</span>{' '}
                {status.data.start_time_iso
                  ? new Date(status.data.start_time_iso).toLocaleString()
                  : 'unknown'}
              </div>
              <div>
                <span className="text-muted-foreground">Duration:</span> {status.data.duration_hours}h
              </div>
              <div>
                <span className="text-muted-foreground">Remaining:</span>{' '}
                {status.data.remaining_minutes != null
                  ? `${Math.floor(status.data.remaining_minutes / 60)}h ${status.data.remaining_minutes % 60}m`
                  : 'unknown'}
              </div>
            </div>
          </div>
        ) : (
          <div className="mt-4 flex items-center gap-3">
            <label className="text-xs text-muted-foreground">
              Keep alive (hours):{' '}
              <select
                value={numHours}
                onChange={(e) => setNumHours(parseInt(e.target.value))}
                className="ml-2 rounded-md border border-border bg-background px-2 py-1 text-sm"
              >
                {Array.from({ length: 12 }, (_, i) => i + 1).map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <button
              onClick={() => trigger.mutate()}
              disabled={trigger.isPending}
              className="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground hover:opacity-90 disabled:opacity-50"
            >
              {trigger.isPending ? 'Starting…' : 'Start All Endpoints'}
            </button>
            {trigger.error && (
              <span className="text-xs text-destructive">{String(trigger.error)}</span>
            )}
          </div>
        )}
      </section>
    </div>
  )
}

function BatchModelsTab() {
  const q = useQuery({ queryKey: ['settings', 'batch-models'], queryFn: api.batchModels })

  const columns = useMemo<ColumnDef<BatchModelRow, unknown>[]>(
    () => [
      { id: 'model', header: 'Model', accessorKey: 'model_display_name' },
      {
        id: 'category',
        header: 'Category',
        accessorFn: (r) => titleCase(r.model_category),
      },
      { id: 'module', header: 'Module', accessorFn: (r) => titleCase(r.module) },
      { id: 'cluster', header: 'Cluster', accessorKey: 'cluster_type' },
      { id: 'job', header: 'Job Name', accessorKey: 'job_name' },
      { id: 'job_id', header: 'Job ID', accessorKey: 'job_id' },
    ],
    [],
  )

  if (q.isLoading) return <Loading />
  if (q.error || !q.data) return <ErrorBlock error={q.error} />

  return (
    <section>
      <div className="mb-3 flex items-center">
        <h3 className="text-sm font-semibold">
          Batch Models/Packages
          <span className="ml-2 rounded-full border border-border bg-muted px-2 py-0.5 text-[11px] font-normal text-muted-foreground">
            {q.data.batch_models.length}
          </span>
        </h3>
      </div>
      <DataTable
        columns={columns}
        data={q.data.batch_models}
        emptyText="No batch models registered yet. Deploy Parabricks, AlphaFold2, or BioNeMo to register batch models."
      />
    </section>
  )
}

export function SettingsPage() {
  return (
    <div className="space-y-6 px-8 py-8">
      <header>
        <h1 className="text-2xl font-semibold">Settings</h1>
      </header>
      <Tabs
        tabs={[
          { id: 'general', label: 'General', content: <GeneralTab /> },
          { id: 'endpoints', label: 'Endpoint Management', content: <EndpointTab /> },
          { id: 'batch', label: 'Batch Models/Packages', content: <BatchModelsTab /> },
        ]}
      />
    </div>
  )
}

function AppearanceSection() {
  const theme = useThemeStore((s) => s.theme)
  const setTheme = useThemeStore((s) => s.setTheme)
  const setUserSettings = useUserStore((s) => s.setUserSettings)
  const qc = useQueryClient()

  const save = useMutation({
    mutationFn: (t: Theme) => api.saveTheme(t),
    onSuccess: (resp) => {
      setUserSettings(resp.user_settings)
      // Bootstrap caches user_settings — keep them in sync.
      qc.invalidateQueries({ queryKey: ['bootstrap'] })
    },
  })

  const pickTheme = (t: Theme) => {
    // Optimistic local update so the UI flips instantly; server save races
    // in the background. localStorage persistence still acts as a fallback
    // if the request fails (the user_settings table just won't update).
    setTheme(t)
    save.mutate(t)
  }

  return (
    <div className="space-y-3 rounded-md border border-border bg-card px-4 py-3">
      <div>
        <h3 className="text-sm font-semibold">Appearance</h3>
      </div>
      <div className="flex items-center gap-3">
        <div className="flex gap-1">
          {(['dark', 'light'] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => pickTheme(t)}
              disabled={save.isPending}
              className={cn(
                'rounded-md border px-4 py-2 text-sm transition-colors',
                theme === t
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border text-muted-foreground hover:bg-accent',
                save.isPending && 'opacity-70',
              )}
            >
              {t === 'dark' ? 'Dark' : 'Light'}
            </button>
          ))}
        </div>
        {save.isPending && (
          <span className="text-xs text-muted-foreground">Saving…</span>
        )}
        {save.isSuccess && (
          <span className="text-xs text-success">Saved.</span>
        )}
        {save.error && (
          <span className="text-xs text-destructive">{String(save.error)}</span>
        )}
      </div>
    </div>
  )
}

function ReadField({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-muted/30 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="truncate text-sm text-foreground">{value}</div>
    </div>
  )
}

function Loading() {
  return <div className="text-sm text-muted-foreground">Loading…</div>
}

function ErrorBlock({ error }: { error: unknown }) {
  return (
    <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
      {String(error)}
    </div>
  )
}

import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { Dialog } from '@/components/Dialog'
import { QcSubTab, RawDataSubTab } from '@/components/RunViewSubTabs'
import { Tabs } from '@/components/Tabs'
import { WorkflowProgress } from '@/components/WorkflowProgress'
import { useUserStore } from '@/stores/user'
import type { SingleCellRun, StartProcessingResponse } from '@/types/api'
import { cn } from '@/lib/utils'

const FormSchema = z.object({
  mode: z.enum(['scanpy', 'rapids-singlecell']),
  data_path: z
    .string()
    .min(1, 'Required')
    .startsWith('/Volumes', 'Must be a /Volumes path')
    .endsWith('.h5ad', 'Must point to an .h5ad file'),
  mlflow_experiment: z.string().min(1, 'Required'),
  mlflow_run_name: z.string().min(1, 'Required'),
  gene_name_column: z.string(),
  species: z.enum(['hsapiens', 'mmusculus', 'rnorvegicus']),
  min_genes: z.number().int().min(0),
  min_cells: z.number().int().min(0),
  pct_counts_mt: z.number().min(0).max(100),
  n_genes_by_counts: z.number().int().min(0),
  target_sum: z.number().int().min(0),
  n_top_genes: z.number().int().min(0),
  n_pcs: z.number().int().min(0),
  cluster_resolution: z.number().min(0).max(2),
  compute_pseudotime: z.boolean(),
})

type FormValues = z.infer<typeof FormSchema>

function tsTag(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

function defaultExperiment(mode: 'scanpy' | 'rapids-singlecell'): string {
  return mode === 'rapids-singlecell'
    ? 'rapidssinglecell_genesis_workbench'
    : 'scanpy_genesis_workbench'
}

function defaultRunName(mode: 'scanpy' | 'rapids-singlecell'): string {
  return mode === 'rapids-singlecell' ? `rapidssinglecell_${tsTag()}` : `scanpy_${tsTag()}`
}

type SearchMode = 'Run Name' | 'Experiment Name'
type ModeFilter = 'All' | 'scanpy' | 'rapids-singlecell'

export function RawProcessingTab() {
  // View opens a modal showing the run's metadata + QC/Raw Data only. The
  // per-run analysis sub-tabs (UMAP, markers, DE, …) moved to the top-level
  // Analysis tab where one shared run picker drives the whole stack.
  const [viewing, setViewing] = useState<SingleCellRun | null>(null)

  return (
    <div className="space-y-6">
      <RunNewAnalysisForm />
      <div className="border-t border-border pt-6">
        <SearchPastRuns onView={setViewing} />
      </div>
      <Dialog
        open={Boolean(viewing)}
        onClose={() => setViewing(null)}
        title={viewing ? `Run: ${viewing.run_name}` : ''}
        width="max-w-5xl"
      >
        {viewing && <RunDetailsBody run={viewing} />}
      </Dialog>
    </div>
  )
}

function RunNewAnalysisForm() {
  const bootstrap = useUserStore((s) => s.bootstrap)
  const defaultH5ad =
    bootstrap?.env
      ? `/Volumes/${bootstrap.env.catalog}/${bootstrap.env.schema_name}/raw_h5ad/0ae6f031-2f9c-4247-8b26-db320d6efd32.h5ad`
      : ''

  const form = useForm<FormValues>({
    resolver: zodResolver(FormSchema),
    defaultValues: {
      mode: 'scanpy',
      data_path: defaultH5ad,
      mlflow_experiment: defaultExperiment('scanpy'),
      mlflow_run_name: defaultRunName('scanpy'),
      gene_name_column: '',
      species: 'hsapiens',
      min_genes: 200,
      min_cells: 3,
      pct_counts_mt: 5,
      n_genes_by_counts: 2500,
      target_sum: 10000,
      n_top_genes: 2000,
      n_pcs: 50,
      cluster_resolution: 0.15,
      compute_pseudotime: false,
    },
  })

  // Keep experiment + run defaults in sync with mode switches.
  const mode = form.watch('mode')
  useEffect(() => {
    form.setValue('mlflow_experiment', defaultExperiment(mode))
    form.setValue('mlflow_run_name', defaultRunName(mode))
  }, [mode, form])

  const qc = useQueryClient()
  const start = useMutation({
    mutationFn: (values: FormValues) =>
      api.singleCellStart({
        ...values,
        // Empty gene_name_column → species fills in the gap server-side.
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['single_cell', 'runs'] })
    },
  })

  const geneNameCol = form.watch('gene_name_column')
  const errors = form.formState.errors

  return (
    <form
      onSubmit={form.handleSubmit((values) => start.mutate(values))}
      className="space-y-6"
    >
      <header>
        <h3 className="text-sm font-semibold">Start a new processing run</h3>
        <p className="text-xs text-muted-foreground">
          Dispatches the scanpy or rapids-singlecell job. Pre-creates the MLflow run so it
          shows up in Search Past Runs immediately.
        </p>
      </header>

      <Field label="Analysis mode">
        <div className="flex gap-1">
          {(['scanpy', 'rapids-singlecell'] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => form.setValue('mode', m)}
              className={cn(
                'rounded-md border px-3 py-2 text-xs transition-colors',
                form.watch('mode') === m
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border text-muted-foreground hover:bg-accent',
              )}
            >
              {m === 'rapids-singlecell' ? 'rapids-singlecell (GPU)' : 'scanpy'}
            </button>
          ))}
        </div>
      </Field>

      <Section title="Data configuration">
        <Field label="Data path (h5ad file)" error={errors.data_path?.message}>
          <input
            {...form.register('data_path')}
            className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
          />
        </Field>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <Field label="Gene name column (optional)" error={errors.gene_name_column?.message}>
            <input
              {...form.register('gene_name_column')}
              placeholder="e.g. gene_name, feature_name"
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
          </Field>
          <Field label="Species (used when gene column is empty)">
            <select
              {...form.register('species')}
              disabled={Boolean(geneNameCol?.trim())}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm disabled:opacity-50"
            >
              <option value="hsapiens">Human (Homo sapiens)</option>
              <option value="mmusculus">Mouse (Mus musculus)</option>
              <option value="rnorvegicus">Rat (Rattus norvegicus)</option>
            </select>
          </Field>
        </div>
      </Section>

      <Section title="MLflow tracking">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <Field label="MLflow experiment name" error={errors.mlflow_experiment?.message}>
            <input
              {...form.register('mlflow_experiment')}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
          </Field>
          <Field label="MLflow run name" error={errors.mlflow_run_name?.message}>
            <input
              {...form.register('mlflow_run_name')}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
          </Field>
        </div>
      </Section>

      <Section title="Filtering">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Field label="Min genes per cell">
            <input type="number" min={0} step={10} {...form.register('min_genes', { valueAsNumber: true })} className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
          </Field>
          <Field label="Min cells per gene">
            <input type="number" min={0} step={1} {...form.register('min_cells', { valueAsNumber: true })} className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
          </Field>
          <Field label="Max % MT counts">
            <input type="number" min={0} max={100} step={0.1} {...form.register('pct_counts_mt', { valueAsNumber: true })} className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
          </Field>
          <Field label="Max genes by counts">
            <input type="number" min={0} step={100} {...form.register('n_genes_by_counts', { valueAsNumber: true })} className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
          </Field>
        </div>
      </Section>

      <Section title="Normalisation & HVG">
        <div className="grid grid-cols-2 gap-3">
          <Field label="Target sum for normalization">
            <input type="number" min={0} step={1000} {...form.register('target_sum', { valueAsNumber: true })} className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
          </Field>
          <Field label="Number of highly-variable genes">
            <input type="number" min={0} step={100} {...form.register('n_top_genes', { valueAsNumber: true })} className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
          </Field>
        </div>
      </Section>

      <Section title="Dimensionality reduction & clustering">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
          <Field label="Principal components">
            <input type="number" min={0} step={5} {...form.register('n_pcs', { valueAsNumber: true })} className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
          </Field>
          <Field label="Cluster resolution">
            <input type="number" min={0} max={2} step={0.05} {...form.register('cluster_resolution', { valueAsNumber: true })} className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
          </Field>
          <Field label="">
            <label className="flex items-center gap-2 pt-2 text-xs">
              <input type="checkbox" {...form.register('compute_pseudotime')} />
              <span>Compute pseudotime</span>
            </label>
          </Field>
        </div>
      </Section>

      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={start.isPending}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
        >
          {start.isPending ? 'Starting…' : 'Start analysis'}
        </button>
        {start.data && <StartSuccessBanner data={start.data} />}
        {start.error && (
          <span className="text-xs text-destructive">{String(start.error)}</span>
        )}
      </div>

      <WorkflowProgress
        active={start.isPending}
        title="Dispatching processing job"
        stages={[
          { label: 'Creating MLflow experiment + run', estSeconds: 3 },
          { label: 'Logging parameters', estSeconds: 1 },
          { label: 'Triggering Databricks job', estSeconds: 4 },
        ]}
      />
    </form>
  )
}

function StartSuccessBanner({ data }: { data: StartProcessingResponse }) {
  return (
    <span className="text-xs">
      <span className="text-success">✓ Job started.</span>{' '}
      Run ID <code className="rounded bg-muted px-1">{data.job_run_id}</code>{' '}
      <a className="text-primary hover:underline" href={data.run_url} target="_blank" rel="noreferrer">
        View in Databricks ↗
      </a>
    </span>
  )
}

function SearchPastRuns({ onView }: { onView: (run: SingleCellRun) => void }) {
  const qc = useQueryClient()
  const [searchMode, setSearchMode] = useState<SearchMode>('Run Name')
  const [searchText, setSearchText] = useState('')
  const [modeFilter, setModeFilter] = useState<ModeFilter>('All')

  const q = useQuery({
    queryKey: ['single_cell', 'runs'],
    queryFn: api.singleCellRuns,
    staleTime: 15_000,
  })

  const filtered = useMemo(() => {
    if (!q.data) return []
    let rows = q.data.runs
    if (modeFilter !== 'All') {
      rows = rows.filter((r) => r.processing_mode === modeFilter)
    }
    if (searchText.trim()) {
      const needle = searchText.trim().toLowerCase()
      const col = searchMode === 'Run Name' ? 'run_name' : 'experiment_name'
      rows = rows.filter((r) => r[col]?.toLowerCase().includes(needle))
    }
    return rows
  }, [q.data, modeFilter, searchText, searchMode])

  const inProgressCount = useMemo(
    () =>
      filtered.filter((r) => ['started', 'processing', 'running'].includes(r.status.toLowerCase()))
        .length,
    [filtered],
  )

  const columns = useMemo<ColumnDef<SingleCellRun, unknown>[]>(
    () => [
      { id: 'run_name', header: 'Run Name', accessorKey: 'run_name' },
      { id: 'experiment_name', header: 'Experiment', accessorKey: 'experiment_name' },
      { id: 'processing_mode', header: 'Mode', accessorKey: 'processing_mode' },
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
          const ready = ['complete', 'finished'].includes(r.status.toLowerCase())
          return (
            <button
              onClick={() => onView(r)}
              disabled={!ready}
              title={
                ready
                  ? 'Load this run into the View Loaded Run tab'
                  : `Wait for status complete — currently ${r.status}`
              }
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
    [onView],
  )

  return (
    <section className="space-y-3">
      <div className="flex items-baseline justify-between">
        <h4 className="text-sm font-medium">Search past runs</h4>
        {inProgressCount > 0 && (
          <span className="inline-flex items-center gap-2 rounded-full border border-amber-500/40 bg-amber-500/10 px-3 py-1 text-xs text-amber-200">
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-amber-400" />
            {inProgressCount} run{inProgressCount > 1 ? 's' : ''} in progress
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <div className="flex gap-1">
          {(['Run Name', 'Experiment Name'] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setSearchMode(m)}
              className={cn(
                'rounded-md border px-3 py-2 text-sm transition-colors',
                m === searchMode
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border text-muted-foreground hover:bg-accent',
              )}
            >
              {m}
            </button>
          ))}
        </div>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            {searchMode} contains
          </span>
          <input
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            placeholder="e.g. scanpy_2026"
            className="w-64 rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Mode</span>
          <select
            value={modeFilter}
            onChange={(e) => setModeFilter(e.target.value as ModeFilter)}
            className="rounded-md border border-border bg-background px-3 py-2 text-sm"
          >
            <option value="All">All</option>
            <option value="scanpy">scanpy</option>
            <option value="rapids-singlecell">rapids-singlecell</option>
          </select>
        </label>
        <button
          type="button"
          onClick={() => qc.invalidateQueries({ queryKey: ['single_cell', 'runs'] })}
          className="rounded-md border border-border px-3 py-2 text-sm hover:bg-accent"
        >
          Refresh
        </button>
      </div>

      {q.isLoading && (
        <div className="text-sm text-muted-foreground">Loading runs…</div>
      )}
      {q.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(q.error)}
        </div>
      )}
      {q.data && filtered.length === 0 && (
        <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
          No runs match the current filters.
        </div>
      )}
      {filtered.length > 0 && <DataTable columns={columns} data={filtered} />}
    </section>
  )
}

function RunDetailsBody({ run }: { run: SingleCellRun }) {
  const summary = useQuery({
    queryKey: ['single_cell', 'run-summary', run.run_id],
    queryFn: () => api.singleCellRunSummary(run.run_id),
    staleTime: 5 * 60_000,
  })

  return (
    <div className="space-y-4">
      <div>
        <p className="text-xs text-muted-foreground">
          Experiment <code>{run.experiment_name}</code> · Mode <code>{run.processing_mode}</code>{' '}
          · Status <code>{run.status}</code>
        </p>
      </div>

      <WorkflowProgress
        active={summary.isLoading}
        title="Loading results"
        stages={[{ label: 'Downloading markers_flat.parquet from MLflow', estSeconds: 5 }]}
      />

      {summary.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(summary.error)}
        </div>
      )}

      {summary.data && (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Metric
              label="Total cells"
              value={
                summary.data.cells_total != null
                  ? summary.data.cells_total.toLocaleString()
                  : `${summary.data.cells_subsample.toLocaleString()}*`
              }
            />
            <Metric
              label={summary.data.cells_total != null ? 'Subsample loaded' : 'Subsample loaded*'}
              value={summary.data.cells_subsample.toLocaleString()}
            />
            <Metric label="Clusters" value={String(summary.data.clusters_count)} />
            <Metric label="Marker genes" value={String(summary.data.markers_count)} />
          </div>

          {summary.data.key_metrics.length > 0 && (
            <div className="rounded-md border border-border bg-card p-3">
              <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Key MLflow metrics
              </div>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs md:grid-cols-3">
                {summary.data.key_metrics.map((m) => (
                  <div key={m.label} className="flex justify-between gap-3">
                    <dt className="text-muted-foreground">{m.label}</dt>
                    <dd className="font-medium">{m.value}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}

          <Tabs
            tabs={[
              {
                id: 'qc',
                label: 'QC & Outputs',
                content: <QcSubTab summary={summary.data} />,
              },
              {
                id: 'raw',
                label: 'Raw Data',
                content: <RawDataSubTab runId={run.run_id} summary={summary.data} />,
              },
            ]}
          />
        </>
      )}
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <fieldset className="space-y-3 rounded-md border border-border bg-card p-4">
      <legend className="px-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {title}
      </legend>
      {children}
    </fieldset>
  )
}

function Field({
  label,
  error,
  children,
}: {
  label: string
  error?: string
  children: React.ReactNode
}) {
  return (
    <label className="block">
      {label && (
        <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      )}
      {children}
      {error && <div className="mt-1 text-xs text-destructive">{error}</div>}
    </label>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-muted/30 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-sm font-medium">{value}</div>
    </div>
  )
}

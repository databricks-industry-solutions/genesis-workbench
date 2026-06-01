import { useMemo, useState } from 'react'
import { useQueries } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { Dialog } from '@/components/Dialog'
import type { ModuleName } from '@/types/api'

type Kind = 'endpoint' | 'workflow_model' | 'workflow_package'

type Row = {
  kind: Kind
  model_id: string
  deploy_id: string
  name: string
  description: string
  model_name: string
  source_version: string
  uc_name: string
  endpoint_name: string
  cluster: string
}

// Batch entries whose display_name matches one of these are pipelines /
// libraries rather than ML models. Keep this list in sync as new modules
// register batch entries (see `batch_models` Delta table).
const WORKFLOW_PACKAGES = new Set<string>([
  'NVIDIA Parabricks',
  'Glow Genomics',
  'Rapids-SingleCell Analysis (GPU)',
  'Scanpy Single Cell Analysis',
])

function classifyBatch(displayName: string): Kind {
  return WORKFLOW_PACKAGES.has(displayName) ? 'workflow_package' : 'workflow_model'
}

const SECTION_META: Record<Kind, { label: string; description: string; empty: string }> = {
  endpoint: {
    label: 'Serving Endpoints',
    description:
      'Models deployed as real-time inference endpoints in Mosaic AI Model Serving.',
    empty: 'No serving endpoints registered for this module.',
  },
  workflow_model: {
    label: 'Models in Workflows',
    description: 'ML Models loaded directly without a model serving endpoint',
    empty: 'No workflow models registered for this module.',
  },
  workflow_package: {
    label: 'Packages in Workflows',
    description:
      'Pipelines and libraries used inside batch jobs (e.g. RAPIDS-singleCell, Parabricks, Glow, Scanpy).',
    empty: 'No workflow packages registered for this module.',
  },
}

export function DeployedModelsTab({ module }: { module: ModuleName }) {
  const includeRealTime = module !== 'genomics'
  const [info, setInfo] = useState<Row | null>(null)

  const results = useQueries({
    queries: [
      {
        queryKey: ['models', 'deployed', module],
        queryFn: () => api.deployedModelsByModule(module),
        enabled: includeRealTime,
      },
      {
        queryKey: ['models', 'batch', module],
        queryFn: () => api.batchModelsByModule(module),
      },
    ],
  })

  const rtQuery = results[0]
  const batchQuery = results[1]
  const isLoading =
    (includeRealTime && rtQuery.isLoading) || batchQuery.isLoading
  const error = rtQuery.error || batchQuery.error

  const grouped: Record<Kind, Row[]> = useMemo(() => {
    const out: Record<Kind, Row[]> = {
      endpoint: [],
      workflow_model: [],
      workflow_package: [],
    }
    if (rtQuery.data && includeRealTime) {
      for (const m of rtQuery.data.models) {
        out.endpoint.push({
          kind: 'endpoint',
          model_id: String(m.model_id),
          deploy_id: String(m.deployment_id),
          name: m.deployment_name,
          description: m.deployment_description ?? '',
          model_name: m.model_display_name,
          source_version: m.model_source_version ?? '',
          uc_name: m.uc_name,
          endpoint_name: m.model_endpoint_name,
          cluster: '',
        })
      }
    }
    if (batchQuery.data) {
      for (const m of batchQuery.data.models) {
        const kind = classifyBatch(m.model_display_name)
        out[kind].push({
          kind,
          model_id: '',
          deploy_id: '',
          name: m.model_display_name,
          description: m.model_description ?? '',
          model_name: '',
          source_version: '',
          uc_name: '',
          endpoint_name: m.job_name,
          cluster: m.cluster_type ?? '',
        })
      }
    }
    return out
  }, [rtQuery.data, batchQuery.data, includeRealTime])

  if (isLoading) return <div className="text-sm text-muted-foreground">Loading…</div>
  if (error)
    return (
      <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
        {String(error)}
      </div>
    )

  const sectionsToRender: Kind[] = includeRealTime
    ? ['endpoint', 'workflow_model', 'workflow_package']
    : ['workflow_model', 'workflow_package']

  return (
    <>
      <div className="space-y-6">
        {sectionsToRender.map((k) => (
          <Section key={k} kind={k} rows={grouped[k]} onInfo={setInfo} />
        ))}
      </div>
      <Dialog
        open={Boolean(info)}
        onClose={() => setInfo(null)}
        title={info ? `${SECTION_META[info.kind].label}: ${info.name}` : ''}
        width="max-w-2xl"
      >
        {info && <DeployedModelDetails row={info} />}
      </Dialog>
    </>
  )
}

function Section({
  kind,
  rows,
  onInfo,
}: {
  kind: Kind
  rows: Row[]
  onInfo: (r: Row) => void
}) {
  const meta = SECTION_META[kind]

  const columns = useMemo<ColumnDef<Row, unknown>[]>(() => {
    const cols: ColumnDef<Row, unknown>[] = [
      { id: 'name', header: 'Name', accessorKey: 'name' },
      {
        id: 'description',
        header: 'Description',
        accessorKey: 'description',
        meta: { thClass: 'min-w-[360px]', tdClass: 'whitespace-normal' },
      },
    ]
    if (kind !== 'endpoint') {
      cols.push({
        id: 'cluster',
        header: 'Cluster',
        accessorKey: 'cluster',
        meta: { tdClass: 'whitespace-nowrap' },
      })
    } else {
      cols.push({
        id: 'endpoint_name',
        header: 'Endpoint',
        accessorKey: 'endpoint_name',
        meta: { tdClass: 'whitespace-nowrap font-mono text-xs' },
      })
    }
    cols.push({
      id: 'info',
      header: '',
      meta: { thClass: 'w-10', tdClass: 'w-10 text-center' },
      cell: (ctx) => (
        <button
          type="button"
          onClick={() => onInfo(ctx.row.original)}
          className="inline-flex h-6 w-6 items-center justify-center rounded-full border border-border text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          aria-label={`More details about ${ctx.row.original.name}`}
          title="More details"
        >
          i
        </button>
      ),
    })
    return cols
  }, [kind, onInfo])

  return (
    <section>
      <div className="mb-2">
        <h3 className="text-sm font-semibold">{meta.label}</h3>
        <p className="text-xs text-muted-foreground">{meta.description}</p>
      </div>
      <DataTable columns={columns} data={rows} emptyText={meta.empty} />
    </section>
  )
}

function DeployedModelDetails({ row }: { row: Row }) {
  // Skip empty fields so a batch row doesn't show blank endpoint-only rows.
  const entries: { label: string; value: string }[] = [
    { label: 'Type', value: SECTION_META[row.kind].label },
    { label: 'Name', value: row.name },
    { label: 'Description', value: row.description },
    { label: 'Model', value: row.model_name },
    { label: 'UC Name', value: row.uc_name },
    { label: 'Source version', value: row.source_version },
    {
      label: row.kind === 'endpoint' ? 'Endpoint' : 'Job',
      value: row.endpoint_name,
    },
    { label: 'Cluster', value: row.cluster },
    { label: 'Model ID', value: row.model_id },
    { label: 'Deployment ID', value: row.deploy_id },
  ].filter((e) => e.value)

  return (
    <dl className="grid grid-cols-1 gap-x-6 gap-y-2 text-sm md:grid-cols-[140px,1fr]">
      {entries.map((e) => (
        <div key={e.label} className="contents">
          <dt className="text-xs uppercase tracking-wide text-muted-foreground">
            {e.label}
          </dt>
          <dd className="break-words font-mono text-xs">{e.value}</dd>
        </div>
      ))}
    </dl>
  )
}

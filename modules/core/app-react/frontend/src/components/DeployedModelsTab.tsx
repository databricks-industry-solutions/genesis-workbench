import { useMemo, useState } from 'react'
import { useQueries } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { Dialog } from '@/components/Dialog'
import type { ModuleName } from '@/types/api'

type Row = {
  model_id: string
  deploy_id: string
  name: string
  description: string
  model_name: string
  source_version: string
  uc_name: string
  endpoint_name: string
  type: 'Real-time' | 'Batch'
  cluster: string
}

export function DeployedModelsTab({ module }: { module: ModuleName }) {
  const includeRealTime = module !== 'disease_biology'
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

  const rows: Row[] = useMemo(() => {
    const out: Row[] = []
    if (rtQuery.data && includeRealTime) {
      for (const m of rtQuery.data.models) {
        out.push({
          model_id: String(m.model_id),
          deploy_id: String(m.deployment_id),
          name: m.deployment_name,
          description: m.deployment_description ?? '',
          model_name: m.model_display_name,
          source_version: m.model_source_version ?? '',
          uc_name: m.uc_name,
          endpoint_name: m.model_endpoint_name,
          type: 'Real-time',
          cluster: '',
        })
      }
    }
    if (batchQuery.data) {
      for (const m of batchQuery.data.models) {
        out.push({
          model_id: '',
          deploy_id: '',
          name: m.model_display_name,
          description: m.model_description ?? '',
          model_name: '',
          source_version: '',
          uc_name: '',
          endpoint_name: m.job_name,
          type: 'Batch',
          cluster: m.cluster_type ?? '',
        })
      }
    }
    return out
  }, [rtQuery.data, batchQuery.data, includeRealTime])

  // Table shows the four headline columns; Info icon opens the rest.
  const columns = useMemo<ColumnDef<Row, unknown>[]>(
    () => [
      {
        id: 'type',
        header: 'Type',
        accessorKey: 'type',
        meta: { thClass: 'min-w-[110px]', tdClass: 'whitespace-nowrap' },
      },
      { id: 'name', header: 'Name', accessorKey: 'name' },
      {
        id: 'description',
        header: 'Description',
        accessorKey: 'description',
        meta: { thClass: 'min-w-[360px]', tdClass: 'whitespace-normal' },
      },
      {
        id: 'cluster',
        header: 'Cluster',
        accessorKey: 'cluster',
        meta: { tdClass: 'whitespace-nowrap' },
      },
      {
        id: 'info',
        header: '',
        meta: { thClass: 'w-10', tdClass: 'w-10 text-center' },
        cell: (ctx) => (
          <button
            type="button"
            onClick={() => setInfo(ctx.row.original)}
            className="inline-flex h-6 w-6 items-center justify-center rounded-full border border-border text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            aria-label={`More details about ${ctx.row.original.name}`}
            title="More details"
          >
            i
          </button>
        ),
      },
    ],
    [],
  )

  if (isLoading) return <div className="text-sm text-muted-foreground">Loading…</div>
  if (error)
    return (
      <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
        {String(error)}
      </div>
    )

  return (
    <>
      <DataTable
        columns={columns}
        data={rows}
        emptyText="No models deployed yet for this module."
      />
      <Dialog
        open={Boolean(info)}
        onClose={() => setInfo(null)}
        title={info ? `${info.type}: ${info.name}` : ''}
        width="max-w-2xl"
      >
        {info && <DeployedModelDetails row={info} />}
      </Dialog>
    </>
  )
}

function DeployedModelDetails({ row }: { row: Row }) {
  // Skip empty fields so a Batch row doesn't show blank Real-time-only rows.
  const entries: { label: string; value: string }[] = [
    { label: 'Type', value: row.type },
    { label: 'Name', value: row.name },
    { label: 'Description', value: row.description },
    { label: 'Model', value: row.model_name },
    { label: 'UC Name', value: row.uc_name },
    { label: 'Source version', value: row.source_version },
    {
      label: row.type === 'Batch' ? 'Job' : 'Endpoint',
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

import { useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { RunSearchSection } from '@/components/RunSearchSection'
import type { DBRunRow } from '@/types/api'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

export function VcfIngestionTab() {
  const defaults = useQuery({
    queryKey: ['disease_biology', 'defaults'],
    queryFn: api.diseaseBiologyDefaults,
    staleTime: Infinity,
  })

  const [vcfPath, setVcfPath] = useState('')
  const [tableName, setTableName] = useState(`variants_${ts()}`)
  const [experiment, setExperiment] = useState('gwb_vcf_ingestion')
  const [runName, setRunName] = useState(`vcf_ingestion_${ts()}`)

  // Seed the VCF path with the BRCA pathogenic sample shipped during the
  // disease_biology deploy — same default the Streamlit page uses for
  // one-click demo runs.
  useEffect(() => {
    if (!defaults.data) return
    setVcfPath((cur) => cur || defaults.data!.vcf_ingestion.vcf_path)
  }, [defaults.data])

  const start = useMutation({ mutationFn: api.vcfIngestionStart })
  const canStart =
    !start.isPending && vcfPath.trim() && tableName.trim() && runName.trim()

  const runStart = () =>
    start.mutate({
      vcf_path: vcfPath,
      output_table_name: tableName,
      mlflow_experiment: experiment,
      mlflow_run_name: runName,
    })

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Convert a VCF File into a Delta Table</h3>
        <p className="text-xs text-muted-foreground">
          Loads a VCF (single- or multi-sample) into a queryable Delta table via Glow. The
          ingested table becomes available to the Variant Annotation workflow's input picker.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">VCF path</span>
          <input
            value={vcfPath}
            onChange={(e) => setVcfPath(e.target.value)}
            placeholder="/Volumes/.../cohort.vcf.gz"
            className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
          />
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Output table name
          </span>
          <input
            value={tableName}
            onChange={(e) => setTableName(e.target.value)}
            className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
          />
        </label>
      </div>

      <div className="rounded-md border border-border bg-card p-3 text-xs">
        <div className="mb-2 font-medium uppercase tracking-wide text-muted-foreground">
          MLflow tracking
        </div>
        <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
          <label className="block">
            <span className="mb-1 block text-muted-foreground">Experiment</span>
            <input
              value={experiment}
              onChange={(e) => setExperiment(e.target.value)}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
          </label>
          <label className="block">
            <span className="mb-1 block text-muted-foreground">Run name</span>
            <input
              value={runName}
              onChange={(e) => setRunName(e.target.value)}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
          </label>
        </div>
      </div>

      <div>
        <button
          onClick={runStart}
          disabled={!canStart}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
        >
          {start.isPending ? 'Dispatching…' : 'Run VCF ingestion'}
        </button>
      </div>

      {start.data && (
        <div className="rounded-md border border-success/40 bg-success/10 p-3 text-xs">
          <span className="text-success">✓ Job dispatched.</span> Run ID{' '}
          <code className="rounded bg-muted px-1">{start.data.job_run_id}</code>{' '}
          {start.data.run_url && (
            <a
              href={start.data.run_url}
              target="_blank"
              rel="noreferrer"
              className="text-primary hover:underline"
            >
              View in Databricks ↗
            </a>
          )}
        </div>
      )}
      {start.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
          {String(start.error)}
        </div>
      )}

      <RunSearchSection
        searchKey={['disease_biology', 'vcf_ingestion', 'search'] as const}
        searchFn={api.vcfIngestionSearch}
        detailLabel="VCF path"
        initialText="vcf_ingestion"
        viewableStatuses={['ingestion_complete']}
        renderDialog={(run) => <VcfIngestionDetailsBody run={run} />}
      />
    </div>
  )
}

function VcfIngestionDetailsBody({ run }: { run: DBRunRow }) {
  const details = useQuery({
    queryKey: ['db', 'run_details', run.run_id],
    queryFn: () => api.diseaseBiologyRunDetails(run.run_id),
    staleTime: 30_000,
  })
  if (details.isLoading) return <div className="text-sm text-muted-foreground">Loading…</div>
  if (details.error)
    return (
      <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
        {String(details.error)}
      </div>
    )
  if (!details.data) return null
  const d = details.data
  return (
    <div className="space-y-3 text-xs">
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-md border border-border bg-muted/30 px-3 py-2">
          <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Status</div>
          <div className="text-sm font-medium">{d.job_status || d.status}</div>
        </div>
        <div className="rounded-md border border-border bg-muted/30 px-3 py-2">
          <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
            Job Run ID
          </div>
          <div className="text-sm font-medium">{d.job_run_id || '—'}</div>
        </div>
      </div>
      <div className="rounded-md border border-border bg-card p-3">
        <div className="mb-2 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          Inputs / outputs
        </div>
        <dl className="space-y-1">
          <KV label="VCF path" value={d.params.vcf_path} />
          <KV label="Output table" value={d.tags.output_table || d.params.output_table_name} />
        </dl>
      </div>
    </div>
  )
}

function KV({ label, value }: { label: string; value: string | undefined }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-mono break-all text-right">{value || '—'}</dd>
    </div>
  )
}

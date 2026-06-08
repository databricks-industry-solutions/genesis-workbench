import { useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { RunSearchSection } from '@/components/RunSearchSection'
import type { DBRunRow } from '@/types/api'
import { cn } from '@/lib/utils'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

type RefMode = 'preset' | 'custom'

export function VariantCallingTab() {
  const defaults = useQuery({
    queryKey: ['genomics', 'defaults'],
    queryFn: api.diseaseBiologyDefaults,
    staleTime: Infinity,
  })

  const [r1, setR1] = useState('')
  const [r2, setR2] = useState('')
  const [refMode, setRefMode] = useState<RefMode>('preset')
  const [customRef, setCustomRef] = useState('')
  const [outVol, setOutVol] = useState('')
  const [experiment, setExperiment] = useState('gwb_variant_calling')
  const [runName, setRunName] = useState(`variant_calling_${ts()}`)

  const presetRef = defaults.data?.variant_calling.reference_genome_path ?? ''
  const ref = refMode === 'preset' ? presetRef : customRef

  // Seed FASTQ + output volume paths from the server-provided defaults once
  // they land. Don't overwrite anything the user has already typed.
  useEffect(() => {
    if (!defaults.data) return
    const d = defaults.data.variant_calling
    setR1((cur) => cur || d.fastq_r1)
    setR2((cur) => cur || d.fastq_r2)
    setOutVol((cur) => cur || d.output_volume_path)
  }, [defaults.data])

  const [searchToken, setSearchToken] = useState(0)
  const start = useMutation({
    mutationFn: api.variantCallingStart,
    onSuccess: () => setSearchToken((t) => t + 1),
  })
  const canStart =
    !start.isPending &&
    r1.trim() &&
    r2.trim() &&
    ref.trim() &&
    outVol.trim() &&
    runName.trim()

  const runStart = () =>
    start.mutate({
      fastq_r1: r1,
      fastq_r2: r2,
      reference_genome_path: ref,
      output_volume_path: outVol,
      mlflow_experiment: experiment,
      mlflow_run_name: runName,
    })

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Align Paired-End Reads to a Reference Genome</h3>
        <p className="text-xs text-muted-foreground">
          GPU-accelerated germline alignment + variant calling with NVIDIA Parabricks. Provide
          paired FASTQ inputs and a reference genome path on a UC Volume; the orchestrator job
          writes BAM + VCF outputs to <code>output_volume_path</code>.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">FASTQ R1</span>
          <input
            value={r1}
            onChange={(e) => setR1(e.target.value)}
            placeholder="/Volumes/.../sample_R1.fastq.gz"
            className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
          />
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">FASTQ R2</span>
          <input
            value={r2}
            onChange={(e) => setR2(e.target.value)}
            placeholder="/Volumes/.../sample_R2.fastq.gz"
            className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
          />
        </label>
        <div className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Reference genome
          </span>
          <div className="flex gap-1">
            {(['preset', 'custom'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setRefMode(m)}
                className={cn(
                  'rounded-md border px-3 py-2 text-xs transition-colors',
                  refMode === m
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border text-muted-foreground hover:bg-accent',
                )}
              >
                {m === 'preset' ? 'GRCh38 (pre-staged)' : 'Custom path'}
              </button>
            ))}
          </div>
          {refMode === 'preset' ? (
            <div className="mt-1 break-all font-mono text-[10px] text-muted-foreground">
              {presetRef || '(no preset configured — set GWAS_REFERENCE_GENOME_PATH on the app)'}
            </div>
          ) : (
            <input
              value={customRef}
              onChange={(e) => setCustomRef(e.target.value)}
              placeholder="/Volumes/.../GRCh38.fa"
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
            />
          )}
        </div>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Output volume path
          </span>
          <input
            value={outVol}
            onChange={(e) => setOutVol(e.target.value)}
            placeholder="/Volumes/.../parabricks_outputs"
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
          {start.isPending ? 'Dispatching…' : 'Run alignment job'}
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
        searchKey={['genomics', 'variant_calling', 'search'] as const}
        searchFn={api.variantCallingSearch}
        detailLabel="FASTQ R1"
        initialText="variant_calling"
        viewableStatuses={['alignment_complete']}
        searchToken={searchToken}
        renderDialog={(run) => <VariantCallingDetailsBody run={run} />}
      />
    </div>
  )
}

function VariantCallingDetailsBody({ run }: { run: DBRunRow }) {
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
        <Metric label="Status" value={d.job_status || d.status} />
        <Metric label="Job Run ID" value={d.job_run_id || '—'} />
      </div>
      <Section title="Input files">
        <KV label="FASTQ R1" value={d.params.fastq_r1} />
        <KV label="FASTQ R2" value={d.params.fastq_r2} />
        <KV label="Reference genome" value={d.params.reference_genome} />
      </Section>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-muted/30 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-sm font-medium">{value || '—'}</div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-border bg-card p-3">
      <div className="mb-2 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        {title}
      </div>
      <dl className="space-y-1">{children}</dl>
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

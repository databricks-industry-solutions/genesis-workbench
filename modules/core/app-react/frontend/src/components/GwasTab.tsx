import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { PlotlyChart as Plot } from '@/components/PlotlyChart'
import { RunSearchSection } from '@/components/RunSearchSection'
import type { DBRunRow, GwasHit, VariantCallingPickerRow } from '@/types/api'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

export function GwasTab() {
  // Pick a completed variant-calling run to seed the VCF path.
  const picker = useQuery({
    queryKey: ['genomics', 'variant_calling', 'successful'],
    queryFn: api.variantCallingSuccessful,
    staleTime: 60_000,
  })

  const defaults = useQuery({
    queryKey: ['genomics', 'defaults'],
    queryFn: api.diseaseBiologyDefaults,
    staleTime: Infinity,
  })

  const [selectedVcRun, setSelectedVcRun] = useState<string>('')
  const [vcfPath, setVcfPath] = useState('')
  const [phenoPath, setPhenoPath] = useState('')
  const [phenoCol, setPhenoCol] = useState('phenotype')
  const [contigs, setContigs] = useState('6')
  const [hwe, setHwe] = useState('0.01')
  const [pvalueThreshold, setPvalueThreshold] = useState('0.01')
  const [experiment, setExperiment] = useState('gwb_gwas')
  const [runName, setRunName] = useState(`gwas_${ts()}`)

  // Seed VCF + phenotype paths from the server-staged sample data once it
  // loads — same env vars (`GWAS_SAMPLE_VCF_PATH`, `GWAS_SAMPLE_PHENOTYPE_PATH`)
  // the Streamlit page reads.
  useEffect(() => {
    if (!defaults.data) return
    const d = defaults.data.gwas
    setVcfPath((cur) => cur || d.vcf_path)
    setPhenoPath((cur) => cur || d.phenotype_path)
  }, [defaults.data])

  // When the user picks a VC run, slot its output VCF into the path field.
  useEffect(() => {
    if (!selectedVcRun || !picker.data) return
    const row = picker.data.runs.find((r) => r.run_id === selectedVcRun)
    if (row?.output_vcf) setVcfPath(row.output_vcf)
  }, [selectedVcRun, picker.data])

  const start = useMutation({ mutationFn: api.gwasStart })
  const canStart =
    !start.isPending &&
    vcfPath.trim() &&
    phenoPath.trim() &&
    phenoCol.trim() &&
    runName.trim()

  const runStart = () =>
    start.mutate({
      vcf_path: vcfPath,
      phenotype_path: phenoPath,
      phenotype_column: phenoCol,
      contigs,
      hwe_cutoff: hwe,
      pvalue_threshold: pvalueThreshold,
      mlflow_experiment: experiment,
      mlflow_run_name: runName,
    })

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Run Genome-Wide Association Analysis</h3>
        <p className="text-xs text-muted-foreground">
          Glow-based GWAS on a VCF + phenotype CSV. Pick a completed Variant Calling run to
          auto-fill the VCF path, or type a custom path. Results land in a per-run Delta table
          you can browse in the View dialog (Manhattan plot + top hits).
        </p>
      </div>

      {picker.data && picker.data.runs.length > 0 && (
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Use VCF from a completed Variant Calling run
          </span>
          <select
            value={selectedVcRun}
            onChange={(e) => setSelectedVcRun(e.target.value)}
            className="w-full max-w-xl rounded-md border border-border bg-background px-3 py-2 text-sm"
          >
            <option value="">— pick a run —</option>
            {picker.data.runs.map((r: VariantCallingPickerRow) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_name} · {r.experiment_name}
              </option>
            ))}
          </select>
        </label>
      )}

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">VCF path</span>
          <input
            value={vcfPath}
            onChange={(e) => setVcfPath(e.target.value)}
            placeholder="/Volumes/.../sample.vcf.gz"
            className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
          />
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Phenotype CSV path
          </span>
          <input
            value={phenoPath}
            onChange={(e) => setPhenoPath(e.target.value)}
            placeholder="/Volumes/.../phenotypes.csv"
            className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
          />
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Phenotype column
          </span>
          <input
            value={phenoCol}
            onChange={(e) => setPhenoCol(e.target.value)}
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Contigs (comma-separated, empty = all)
          </span>
          <input
            value={contigs}
            onChange={(e) => setContigs(e.target.value)}
            placeholder="chr1,chr2,…"
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            HWE cutoff
          </span>
          <input
            value={hwe}
            onChange={(e) => setHwe(e.target.value)}
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            p-value threshold
          </span>
          <input
            value={pvalueThreshold}
            onChange={(e) => setPvalueThreshold(e.target.value)}
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
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
          {start.isPending ? 'Dispatching…' : 'Run GWAS analysis'}
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
        searchKey={['genomics', 'gwas', 'search'] as const}
        searchFn={api.gwasSearch}
        detailLabel="VCF path"
        initialText="gwas"
        viewableStatuses={['gwas_complete']}
        renderDialog={(run) => <GwasResultsBody run={run} />}
      />
    </div>
  )
}

function GwasResultsBody({ run }: { run: DBRunRow }) {
  const results = useQuery({
    queryKey: ['db', 'gwas_results', run.run_id],
    queryFn: () => api.gwasResults(run.run_id),
    staleTime: 60_000,
  })

  const columns = useMemo<ColumnDef<GwasHit, unknown>[]>(
    () => [
      { id: 'contig', header: 'Contig', accessorKey: 'contig' },
      { id: 'position', header: 'Position', accessorKey: 'position' },
      {
        id: 'pvalue',
        header: 'p-value',
        accessorFn: (r) => r.pvalue.toExponential(2),
      },
      {
        id: 'neg_log_pval',
        header: '-log10(p)',
        accessorFn: (r) => (r.neg_log_pval == null ? '—' : r.neg_log_pval.toFixed(2)),
      },
      { id: 'reference_allele', header: 'Ref', accessorKey: 'reference_allele' },
      { id: 'alternate_alleles', header: 'Alt', accessorKey: 'alternate_alleles' },
    ],
    [],
  )

  if (results.isLoading) return <div className="text-sm text-muted-foreground">Loading…</div>
  if (results.error)
    return (
      <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
        {String(results.error)}
      </div>
    )
  if (!results.data) return null
  const r = results.data
  if (r.total_variants === 0) {
    return (
      <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-200">
        No significant results found for this run. All p-values may be NULL — this happens
        when the sample size is too small or the phenotype has insufficient variation.
      </div>
    )
  }
  return (
    <div className="space-y-3 text-xs">
      <div className="grid grid-cols-3 gap-3">
        <Metric label="Total variants" value={r.total_variants.toLocaleString()} />
        <Metric label="Significant (p<5e-8)" value={r.significant_count.toLocaleString()} />
        <Metric
          label="Min p-value"
          value={r.min_pvalue != null ? r.min_pvalue.toExponential(2) : '—'}
        />
      </div>

      {r.manhattan_points.length > 0 && (
        <div className="rounded-md border border-border bg-card p-2">
          <Plot
            data={[
              {
                type: 'scattergl',
                mode: 'markers',
                x: r.manhattan_points.map((p) => p.x),
                y: r.manhattan_points.map((p) => p.y),
                marker: { size: 3, opacity: 0.5, color: '#60a5fa' },
                hovertemplate: '%{x}<br>-log10(p)=%{y:.2f}<extra></extra>',
              } as never,
            ]}
            layout={{
              title: { text: 'GWAS Manhattan plot' },
              height: 320,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { size: 11 },
              xaxis: { title: { text: 'genomic position' }, gridcolor: '#333' },
              yaxis: { title: { text: '-log10(p)' }, gridcolor: '#333' },
              shapes: [
                {
                  type: 'line',
                  xref: 'paper',
                  x0: 0,
                  x1: 1,
                  yref: 'y',
                  y0: -Math.log10(5e-8),
                  y1: -Math.log10(5e-8),
                  line: { color: '#ef4444', dash: 'dash', width: 1 },
                },
                {
                  type: 'line',
                  xref: 'paper',
                  x0: 0,
                  x1: 1,
                  yref: 'y',
                  y0: 5,
                  y1: 5,
                  line: { color: '#3b82f6', dash: 'dash', width: 1 },
                },
              ],
              margin: { l: 50, r: 20, t: 40, b: 40 },
            }}
            config={{ displaylogo: false, responsive: true }}
            style={{ width: '100%' }}
            useResizeHandler
          />
        </div>
      )}

      <details className="rounded-md border border-border">
        <summary className="cursor-pointer px-4 py-2 text-sm">
          Top hits ({r.top_hits.length})
        </summary>
        <div className="p-3">
          <DataTable columns={columns} data={r.top_hits} />
        </div>
      </details>
    </div>
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

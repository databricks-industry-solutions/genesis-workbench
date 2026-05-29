import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { RunSearchSection } from '@/components/RunSearchSection'
import type { AnnotationVariant, DBRunRow, VcfIngestionPickerRow } from '@/types/api'
import { cn } from '@/lib/utils'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

export function VariantAnnotationTab() {
  const picker = useQuery({
    queryKey: ['disease_biology', 'vcf_ingestion', 'successful'],
    queryFn: api.vcfIngestionSuccessful,
    staleTime: 60_000,
  })

  const [selectedIngRun, setSelectedIngRun] = useState('')
  const [variantsTable, setVariantsTable] = useState('')
  const [geneRegions, setGeneRegions] = useState('')
  // UI-only preset; collapses to acmg|custom for the backend.
  const [genePanelPreset, setGenePanelPreset] = useState<'acmg' | 'brca' | 'custom'>('acmg')
  const [experiment, setExperiment] = useState('gwb_variant_annotation')
  const [runName, setRunName] = useState(`variant_annotation_${ts()}`)

  const BRCA1_BRCA2_JSON =
    '[{"name":"BRCA1","contig":"chr17","start":43044292,"end":43170327},{"name":"BRCA2","contig":"chr13","start":32315086,"end":32400268}]'

  const genePanelMode: 'acmg' | 'custom' =
    genePanelPreset === 'acmg' ? 'acmg' : 'custom'
  const effectiveGeneRegions =
    genePanelPreset === 'brca'
      ? BRCA1_BRCA2_JSON
      : genePanelPreset === 'custom'
        ? geneRegions
        : ''

  useEffect(() => {
    if (!selectedIngRun || !picker.data) return
    const row = picker.data.runs.find((r) => r.run_id === selectedIngRun)
    if (row?.output_table) setVariantsTable(row.output_table)
  }, [selectedIngRun, picker.data])

  const start = useMutation({ mutationFn: api.variantAnnotationStart })

  const canStart =
    !start.isPending &&
    variantsTable.trim() &&
    runName.trim() &&
    (genePanelPreset !== 'custom' || geneRegions.trim())

  const runStart = () =>
    start.mutate({
      variants_table: variantsTable,
      gene_regions: effectiveGeneRegions,
      pathogenic_vcf_path: '',
      gene_panel_mode: genePanelMode,
      mlflow_experiment: experiment,
      mlflow_run_name: runName,
    })

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Annotate Variants with Clinical Significance</h3>
        <p className="text-xs text-muted-foreground">
          Annotate an ingested variants table against a pathogenic-variant VCF (e.g. ClinVar)
          and a gene panel. ACMG SF v3.2 panel is selectable as a built-in; switch to{' '}
          <code>custom</code> to supply a JSON list of gene regions.
        </p>
      </div>

      {picker.data && picker.data.runs.length > 0 && (
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Use table from a completed VCF Ingestion run
          </span>
          <select
            value={selectedIngRun}
            onChange={(e) => setSelectedIngRun(e.target.value)}
            className="w-full max-w-xl rounded-md border border-border bg-background px-3 py-2 text-sm"
          >
            <option value="">— pick a run —</option>
            {picker.data.runs.map((r: VcfIngestionPickerRow) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_name} · {r.experiment_name}
              </option>
            ))}
          </select>
        </label>
      )}

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Variants table (catalog.schema.table)
          </span>
          <input
            value={variantsTable}
            onChange={(e) => setVariantsTable(e.target.value)}
            placeholder="catalog.schema.cohort_variants"
            className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
          />
        </label>
      </div>

      <label className="block text-xs">
        <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
          Gene panel
        </span>
        <div className="flex flex-wrap gap-1">
          {(
            [
              ['acmg', 'ACMG SF v3.2 (81 genes)'],
              ['brca', 'BRCA1 + BRCA2'],
              ['custom', 'Custom JSON'],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setGenePanelPreset(id)}
              className={cn(
                'rounded-md border px-3 py-2 text-xs transition-colors',
                genePanelPreset === id
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border text-muted-foreground hover:bg-accent',
              )}
            >
              {label}
            </button>
          ))}
        </div>
        {genePanelPreset === 'acmg' && (
          <div className="mt-1 text-[10px] text-muted-foreground">
            81 medically-actionable genes — Cancer (29), Cardiovascular (44), Metabolic (4),
            Miscellaneous (4).
          </div>
        )}
        {genePanelPreset === 'brca' && (
          <div className="mt-1 text-[10px] text-muted-foreground">
            Using BRCA1 (chr17:43,044,292-43,170,327) and BRCA2 (chr13:32,315,086-32,400,268).
          </div>
        )}
      </label>

      {genePanelPreset === 'custom' && (
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Gene regions (JSON)
          </span>
          <textarea
            rows={5}
            value={geneRegions}
            onChange={(e) => setGeneRegions(e.target.value)}
            placeholder='[{"gene": "TP53", "chromosome": "chr17", "start": 7565097, "end": 7590856}]'
            className="w-full rounded-md border border-border bg-background p-3 font-mono text-[11px]"
          />
        </label>
      )}

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
          {start.isPending ? 'Dispatching…' : 'Run annotation'}
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
        searchKey={['disease_biology', 'variant_annotation', 'search'] as const}
        searchFn={api.variantAnnotationSearch}
        detailLabel="Variants table"
        initialText="annotation"
        viewableStatuses={['annotation_complete']}
        renderDialog={(run) => <AnnotationResultsBody run={run} />}
      />
    </div>
  )
}

function AnnotationResultsBody({ run }: { run: DBRunRow }) {
  const results = useQuery({
    queryKey: ['db', 'annotation_results', run.run_id],
    queryFn: () => api.variantAnnotationResults(run.run_id),
    staleTime: 60_000,
  })

  const cols = useMemo<ColumnDef<AnnotationVariant, unknown>[]>(
    () => [
      { id: 'gene', header: 'Gene', accessorKey: 'gene' },
      { id: 'category', header: 'Category', accessorKey: 'category' },
      { id: 'condition', header: 'Condition', accessorKey: 'condition' },
      { id: 'chromosome', header: 'Chr', accessorKey: 'chromosome' },
      { id: 'position', header: 'Position', accessorKey: 'position' },
      { id: 'ref', header: 'Ref', accessorKey: 'ref' },
      { id: 'alt', header: 'Alt', accessorKey: 'alt' },
      { id: 'zygosity', header: 'Zyg', accessorKey: 'zygosity' },
      {
        id: 'clinical_significance',
        header: 'Clinical significance',
        accessorKey: 'clinical_significance',
        meta: { tdClass: 'whitespace-normal break-words' },
      },
      {
        id: 'disease_name',
        header: 'Disease',
        accessorKey: 'disease_name',
        meta: { tdClass: 'whitespace-normal break-words' },
      },
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

  if (results.data.total === 0) {
    return (
      <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
        No pathogenic variants were logged for this run.
      </div>
    )
  }

  return (
    <div className="space-y-3 text-xs">
      <div className="text-sm">
        <strong>{results.data.total.toLocaleString()}</strong> pathogenic variants
      </div>
      <DataTable columns={cols} data={results.data.variants} />
    </div>
  )
}

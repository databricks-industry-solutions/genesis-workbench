/**
 * Seven sub-tabs for the View Analysis Results dialog (UMAP, Marker Genes, DE, Pathway Enrichment, Trajectory, QC, Outputs).
 * modules/core/app/views/single_cell_workflows/processing.py:
 *
 *   UMAP, Marker Genes, Differential Expression, Pathway Enrichment,
 *   Trajectory, QC & Outputs, Raw Data
 *
 * Each tab is a thin component over the matching backend route
 * (/api/single_cell/run-*) — scanpy/scipy run server-side, results plotted by Plotly client-side.
 * here the computation lives in FastAPI because the OBO user token cannot
 * carry that workload.
 */
import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { GeneHighlightPicker, type Highlight } from '@/components/GeneHighlightPicker'
import { PlotlyChart as Plot } from '@/components/PlotlyChart'
import { RealtimeProgress } from '@/components/RealtimeProgress'
import { Tabs } from '@/components/Tabs'
import { WorkflowProgress } from '@/components/WorkflowProgress'
import { useSseMutation } from '@/hooks/useSseMutation'
import { useUserStore } from '@/stores/user'
import type {
  AnnotateResponse,
  ClusterAnnotation,
  ColorPointsResponse,
  DEGene,
  DotplotResponse,
  EnrichmentTerm,
  RunSummaryResponse,
  TeddyAnnotateResponse,
  TeddyClusterAnnotation,
} from '@/types/api'
import { cn } from '@/lib/utils'

export function RunViewSubTabs({
  runId,
  summary,
}: {
  runId: string
  summary: RunSummaryResponse
}) {
  return (
    <Tabs
      tabs={[
        {
          id: 'umap',
          label: 'UMAP',
          content: <UmapSubTab runId={runId} summary={summary} />,
        },
        {
          id: 'markers',
          label: 'Marker Genes',
          content: <MarkerDotplotSubTab runId={runId} summary={summary} />,
        },
        {
          id: 'de',
          label: 'Differential Expression',
          content: <DESubTab runId={runId} summary={summary} />,
        },
        {
          id: 'enrich',
          label: 'Pathway Enrichment',
          content: <EnrichmentSubTab runId={runId} summary={summary} />,
        },
        {
          id: 'traj',
          label: 'Trajectory',
          content: <TrajectorySubTab runId={runId} summary={summary} />,
        },
        {
          id: 'qc',
          label: 'QC & Outputs',
          content: <QcSubTab summary={summary} />,
        },
        {
          id: 'raw',
          label: 'Raw Data',
          content: <RawDataSubTab runId={runId} summary={summary} />,
        },
      ]}
    />
  )
}

// ─── UMAP sub-tab ──────────────────────────────────────────────────────────

type ColorType = 'cluster' | 'predicted_cell_type' | 'predicted_disease' | 'gene' | 'metric'
type OverlaySource = 'scimilarity' | 'teddy'

export function UmapSubTab({ runId, summary }: { runId: string; summary: RunSummaryResponse }) {
  // Annotation controls — both models run from the same Annotate button.
  const [useScim, setUseScim] = useState(true)
  const [useTeddy, setUseTeddy] = useState(true)
  const [biasCorrect, setBiasCorrect] = useState(true)
  const [annotationCells, setAnnotationCells] = useState(10)
  const [annotationK, setAnnotationK] = useState(20)

  // SSE-backed mutations so the progress bars below show real server-side
  // pct + step text instead of time-based estimates.
  const scim = useSseMutation<
    { run_id: string; cells_per_cluster: number; k_neighbors: number },
    AnnotateResponse
  >('/api/single_cell/annotate/stream')
  const teddy = useSseMutation<
    {
      run_id: string
      cells_per_cluster: number
      k_neighbors: number
      bias_correct: boolean
    },
    TeddyAnnotateResponse
  >('/api/single_cell/annotate-teddy/stream')

  const runAnnotate = () => {
    if (useScim)
      scim.start({
        run_id: runId,
        cells_per_cluster: annotationCells,
        k_neighbors: annotationK,
      })
    if (useTeddy)
      teddy.start({
        run_id: runId,
        cells_per_cluster: annotationCells,
        k_neighbors: annotationK,
        bias_correct: biasCorrect,
      })
  }

  // Restore previously-saved annotations (if any) on tab open / runId change.
  // The MLflow artifacts persist across sessions so the table+UMAP overlay
  // come back without re-running the pipeline.
  useEffect(() => {
    let cancelled = false
    api
      .singleCellSavedAnnotations(runId)
      .then((saved) => {
        if (cancelled) return
        if (saved.scimilarity) scim.setData(saved.scimilarity)
        if (saved.teddy) teddy.setData(saved.teddy)
      })
      .catch(() => {
        // Treat any failure as "no saved annotations" — fresh state is fine.
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId])

  const hasScim = Boolean(scim.data)
  const hasTeddy = Boolean(teddy.data)

  // Cluster → label maps for each model. The overlay-source selector picks
  // which one drives the UMAP "Predicted cell type" colouring.
  const scimCellTypeMap = useMemo(() => {
    const m = new Map<string, string>()
    scim.data?.annotations.forEach((a) => m.set(a.cluster, a.predicted_cell_type))
    return m
  }, [scim.data])
  const teddyCellTypeMap = useMemo(() => {
    const m = new Map<string, string>()
    Object.entries(teddy.data?.cluster_to_cell_type ?? {}).forEach(([k, v]) =>
      m.set(k, v),
    )
    return m
  }, [teddy.data])
  const teddyDiseaseMap = useMemo(() => {
    const m = new Map<string, string>()
    Object.entries(teddy.data?.cluster_to_disease ?? {}).forEach(([k, v]) => m.set(k, v))
    return m
  }, [teddy.data])

  const [overlaySource, setOverlaySource] = useState<OverlaySource>('scimilarity')
  // Reset overlay source if the currently-selected model loses its data.
  if (overlaySource === 'scimilarity' && !hasScim && hasTeddy) {
    setOverlaySource('teddy')
  }

  const [colorType, setColorType] = useState<ColorType>('cluster')
  const [selectedGene, setSelectedGene] = useState<string>(summary.expr_genes[0] ?? '')
  const [selectedMetric, setSelectedMetric] = useState<string>(
    summary.obs_numerical.find((c) =>
      ['n_genes', 'n_counts', 'pct_counts_mt', 'n_genes_by_counts'].includes(c),
    ) ??
      summary.obs_numerical[0] ??
      '',
  )

  const activeCellTypeMap =
    overlaySource === 'scimilarity' ? scimCellTypeMap : teddyCellTypeMap

  const colorColumn = useMemo<string | null>(() => {
    if (colorType === 'cluster') return summary.cluster_col
    if (colorType === 'predicted_cell_type') return null
    if (colorType === 'predicted_disease') return null
    if (colorType === 'gene') return selectedGene ? `expr_${selectedGene}` : null
    if (colorType === 'metric') return selectedMetric || null
    return null
  }, [colorType, selectedGene, selectedMetric, summary.cluster_col])

  const colorPoints = useQuery({
    queryKey: ['sc', 'color-points', runId, colorColumn],
    queryFn: () => api.singleCellColorPoints({ run_id: runId, color_column: colorColumn! }),
    enabled: Boolean(colorColumn),
    staleTime: 60_000,
  })

  const traces = useMemo(() => {
    if (colorType === 'predicted_cell_type' || colorType === 'predicted_disease') {
      const labelMap =
        colorType === 'predicted_cell_type' ? activeCellTypeMap : teddyDiseaseMap
      const byLabel = new Map<string, { x: number[]; y: number[] }>()
      for (const p of summary.umap_points) {
        const label = labelMap.get(p.cluster) ?? 'Unknown'
        const bucket = byLabel.get(label) ?? { x: [], y: [] }
        bucket.x.push(p.umap_0)
        bucket.y.push(p.umap_1)
        byLabel.set(label, bucket)
      }
      return [...byLabel.entries()].map(([label, b]) => ({
        type: 'scattergl' as const,
        mode: 'markers' as const,
        name: label,
        x: b.x,
        y: b.y,
        marker: { size: 3, opacity: 0.75 },
        hovertemplate: `<b>${label}</b><extra></extra>`,
      }))
    }
    if (!colorPoints.data) return []
    if (colorPoints.data.is_categorical) {
      const byVal = new Map<string, { x: number[]; y: number[] }>()
      const values = colorPoints.data.values_str ?? []
      for (let i = 0; i < values.length; i++) {
        const v = values[i]
        const bucket = byVal.get(v) ?? { x: [], y: [] }
        bucket.x.push(colorPoints.data.umap_0[i])
        bucket.y.push(colorPoints.data.umap_1[i])
        byVal.set(v, bucket)
      }
      const sortedKeys = [...byVal.keys()].sort((a, b) => {
        const ai = parseInt(a)
        const bi = parseInt(b)
        if (!Number.isNaN(ai) && !Number.isNaN(bi)) return ai - bi
        return a.localeCompare(b)
      })
      return sortedKeys.map((v) => {
        const bucket = byVal.get(v)!
        return {
          type: 'scattergl' as const,
          mode: 'markers' as const,
          name: v,
          x: bucket.x,
          y: bucket.y,
          marker: { size: 3, opacity: 0.75 },
          hovertemplate: `<b>${v}</b><extra></extra>`,
        }
      })
    }
    return [
      {
        type: 'scattergl' as const,
        mode: 'markers' as const,
        x: colorPoints.data.umap_0,
        y: colorPoints.data.umap_1,
        marker: {
          size: 3,
          opacity: 0.8,
          color: colorPoints.data.values_num ?? [],
          colorscale: 'Viridis',
          showscale: true,
          colorbar: { title: { text: colorColumn ?? '' } },
        },
        hovertemplate: `<b>${colorColumn}: %{marker.color:.3f}</b><extra></extra>`,
      },
    ]
  }, [
    colorType,
    colorPoints.data,
    summary.umap_points,
    activeCellTypeMap,
    teddyDiseaseMap,
    colorColumn,
  ])

  const titleSuffix =
    colorType === 'cluster'
      ? summary.cluster_col
      : colorType === 'predicted_cell_type'
        ? `Predicted cell type (${overlaySource === 'scimilarity' ? 'SCimilarity' : 'TEDDY'})`
        : colorType === 'predicted_disease'
          ? 'Predicted disease (TEDDY)'
          : colorType === 'gene'
            ? selectedGene
            : selectedMetric

  return (
    <div className="space-y-4">
      <DualAnnotationPanel
        useScim={useScim}
        setUseScim={setUseScim}
        useTeddy={useTeddy}
        setUseTeddy={setUseTeddy}
        biasCorrect={biasCorrect}
        setBiasCorrect={setBiasCorrect}
        cells={annotationCells}
        setCells={setAnnotationCells}
        k={annotationK}
        setK={setAnnotationK}
        onRun={runAnnotate}
        scim={scim}
        teddy={teddy}
      />

      <div className="flex flex-wrap items-end gap-3">
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Color by
          </span>
          <select
            value={colorType}
            onChange={(e) => setColorType(e.target.value as ColorType)}
            className="rounded-md border border-border bg-background px-3 py-2 text-sm"
          >
            <option value="cluster">Cluster</option>
            {(hasScim || hasTeddy) && (
              <option value="predicted_cell_type">Predicted cell type</option>
            )}
            {hasTeddy && <option value="predicted_disease">Predicted disease (TEDDY)</option>}
            <option value="gene">Marker gene</option>
            <option value="metric">QC metric</option>
          </select>
        </label>

        {colorType === 'predicted_cell_type' && hasScim && hasTeddy && (
          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Overlay source
            </span>
            <div className="flex gap-1">
              {(['scimilarity', 'teddy'] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setOverlaySource(m)}
                  className={cn(
                    'rounded-md border px-3 py-2 text-xs transition-colors',
                    overlaySource === m
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-border text-muted-foreground hover:bg-accent',
                  )}
                >
                  {m === 'scimilarity' ? 'SCimilarity' : 'TEDDY'}
                </button>
              ))}
            </div>
          </label>
        )}

        {colorType === 'gene' && (
          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Gene</span>
            <select
              value={selectedGene}
              onChange={(e) => setSelectedGene(e.target.value)}
              className="w-56 rounded-md border border-border bg-background px-3 py-2 text-sm"
            >
              {summary.expr_genes.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
          </label>
        )}

        {colorType === 'metric' && (
          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Metric
            </span>
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="rounded-md border border-border bg-background px-3 py-2 text-sm"
            >
              {summary.obs_numerical.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {colorPoints.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(colorPoints.error)}
        </div>
      )}

      <div className="rounded-md border border-border bg-card p-2">
        <Plot
          data={traces as never}
          layout={{
            title: { text: `UMAP — coloured by ${titleSuffix}` },
            height: 560,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { size: 11 },
            xaxis: { title: { text: 'UMAP_0' }, gridcolor: '#333' },
            yaxis: { title: { text: 'UMAP_1' }, gridcolor: '#333' },
            legend: { font: { size: 10 } },
            margin: { l: 50, r: 20, t: 50, b: 50 },
          }}
          config={{ displaylogo: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  )
}

function DualAnnotationPanel({
  useScim,
  setUseScim,
  useTeddy,
  setUseTeddy,
  biasCorrect,
  setBiasCorrect,
  cells,
  setCells,
  k,
  setK,
  onRun,
  scim,
  teddy,
}: {
  useScim: boolean
  setUseScim: (v: boolean) => void
  useTeddy: boolean
  setUseTeddy: (v: boolean) => void
  biasCorrect: boolean
  setBiasCorrect: (v: boolean) => void
  cells: number
  setCells: (n: number) => void
  k: number
  setK: (n: number) => void
  onRun: () => void
  scim: any
  teddy: any
}) {
  const scimColumns = useMemo<ColumnDef<ClusterAnnotation, unknown>[]>(
    () => [
      { id: 'cluster', header: 'Cluster', accessorKey: 'cluster' },
      { id: 'predicted_cell_type', header: 'Cell type', accessorKey: 'predicted_cell_type' },
      {
        id: 'confidence',
        header: 'Conf.',
        cell: (ctx) => `${ctx.row.original.confidence_pct.toFixed(0)}%`,
      },
      { id: 'top_predictions', header: 'Top 3', accessorKey: 'top_predictions' },
    ],
    [],
  )

  const teddyColumns = useMemo<ColumnDef<TeddyClusterAnnotation, unknown>[]>(
    () => [
      { id: 'cluster', header: 'Cluster', accessorKey: 'cluster' },
      { id: 'cell_type', header: 'Cell type', accessorKey: 'predicted_cell_type' },
      {
        id: 'ct_conf',
        header: 'CT conf.',
        cell: (ctx) => `${ctx.row.original.cell_type_confidence_pct.toFixed(0)}%`,
      },
      { id: 'disease', header: 'Disease', accessorKey: 'predicted_disease' },
      {
        id: 'ds_conf',
        header: 'Ds. conf.',
        cell: (ctx) => `${ctx.row.original.disease_confidence_pct.toFixed(0)}%`,
      },
    ],
    [],
  )

  const anyRunning = scim.isPending || teddy.isPending
  const noneSelected = !useScim && !useTeddy
  const hasScim = Boolean(scim.data)
  const hasTeddy = Boolean(teddy.data)

  return (
    <section className="space-y-3 rounded-md border border-border bg-card p-4">
      <header>
        <h4 className="text-sm font-medium">Cell Type Annotation</h4>
        <p className="text-xs text-muted-foreground">
          Run <strong>SCimilarity</strong> (23M-cell pan-tissue reference) and/or{' '}
          <strong>TEDDY</strong> (Merck foundation model + CELLxGENE reference, also predicts
          disease). Results overlay on the UMAP below.
        </p>
      </header>

      <div className="flex flex-wrap items-end gap-3">
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={useScim}
            onChange={(e) => setUseScim(e.target.checked)}
          />
          SCimilarity
        </label>
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={useTeddy}
            onChange={(e) => setUseTeddy(e.target.checked)}
          />
          TEDDY
        </label>
        <label
          className={cn(
            'flex items-center gap-2 text-xs',
            !useTeddy && 'opacity-50',
          )}
          title="(TEDDY only) IDF-weight each neighbour vote to counteract the disease-biased reference."
        >
          <input
            type="checkbox"
            checked={biasCorrect}
            onChange={(e) => setBiasCorrect(e.target.checked)}
            disabled={!useTeddy}
          />
          Bias-correct (IDF)
        </label>

        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Cells per cluster
          </span>
          <input
            type="number"
            min={3}
            max={50}
            step={1}
            value={cells}
            onChange={(e) => setCells(parseInt(e.target.value || '10'))}
            className="w-24 rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Neighbours (k)
          </span>
          <input
            type="number"
            min={5}
            max={200}
            step={5}
            value={k}
            onChange={(e) => setK(parseInt(e.target.value || '20'))}
            className="w-24 rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
        </label>

        <button
          onClick={onRun}
          disabled={noneSelected || anyRunning}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
        >
          {anyRunning ? 'Annotating…' : 'Annotate clusters'}
        </button>
      </div>

      {scim.isPending && (
        <RealtimeProgress
          title="SCimilarity pipeline"
          pct={scim.progress?.pct ?? 0}
          msg={scim.progress?.msg ?? 'Starting…'}
          stages={[
            { label: 'Fetching gene order', pctEnd: 10 },
            { label: 'Generating cell embeddings (batched)', pctEnd: 55 },
            { label: 'Vector Search nearest-neighbour lookup', pctEnd: 96 },
            { label: 'Majority-vote per cluster', pctEnd: 100 },
          ]}
        />
      )}
      {teddy.isPending && (
        <RealtimeProgress
          title="TEDDY pipeline"
          pct={teddy.progress?.pct ?? 0}
          msg={teddy.progress?.msg ?? 'Starting…'}
          stages={[
            { label: 'Loading reference assets', pctEnd: 10 },
            { label: 'Generating cell embeddings (batched)', pctEnd: 55 },
            { label: 'Vector Search nearest-neighbour lookup', pctEnd: 96 },
            { label: 'Voting cell type + disease', pctEnd: 100 },
          ]}
        />
      )}

      {(scim.error || teddy.error) && (
        <div className="space-y-1">
          {scim.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
              SCimilarity: {String(scim.error)}
            </div>
          )}
          {teddy.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
              TEDDY: {String(teddy.error)}
            </div>
          )}
        </div>
      )}

      {(hasScim || hasTeddy) && (
        <div
          className={cn(
            'grid gap-4',
            hasScim && hasTeddy ? 'md:grid-cols-2' : 'grid-cols-1',
          )}
        >
          {hasScim && (
            <div>
              <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                SCimilarity — cell type
              </div>
              {scim.data!.warnings && scim.data!.warnings.length > 0 && (
                <details className="mb-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-2 text-xs">
                  <summary className="cursor-pointer font-medium text-amber-700 dark:text-amber-400">
                    ⚠ {scim.data!.warnings.length} embedding batch
                    {scim.data!.warnings.length === 1 ? '' : 'es'} skipped — clusters annotated
                    with partial data
                  </summary>
                  <ul className="mt-1 list-disc space-y-0.5 pl-5 text-muted-foreground">
                    {scim.data!.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </details>
              )}
              <DataTable columns={scimColumns} data={scim.data.annotations} />
            </div>
          )}
          {hasTeddy && (
            <div>
              <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                TEDDY — cell type + disease
              </div>
              <DataTable columns={teddyColumns} data={teddy.data.annotations} />
            </div>
          )}
        </div>
      )}
    </section>
  )
}

// ─── Marker Genes dotplot ──────────────────────────────────────────────────

export function MarkerDotplotSubTab({
  runId,
  summary,
}: {
  runId: string
  summary: RunSummaryResponse
}) {
  const [nTopGenes, setNTopGenes] = useState(3)
  const [scale, setScale] = useState(true)

  const q = useQuery({
    queryKey: ['sc', 'dotplot', runId, nTopGenes, scale],
    queryFn: () =>
      api.singleCellDotplot({
        run_id: runId,
        n_top_genes_per_cluster: nTopGenes,
        scale_data: scale,
      }),
    staleTime: 60_000,
  })

  return (
    <div className="space-y-4">
      <header>
        <h4 className="text-sm font-medium">Marker gene expression by cluster</h4>
        <p className="text-xs text-muted-foreground">
          Top N marker genes per cluster (by z-scored mean expression). Dot size encodes
          |expression|; colour encodes signed value.
        </p>
      </header>

      <div className="flex flex-wrap items-end gap-3">
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Top genes per cluster
          </span>
          <input
            type="number"
            min={1}
            max={20}
            step={1}
            value={nTopGenes}
            onChange={(e) => setNTopGenes(parseInt(e.target.value || '3'))}
            className="w-24 rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
        </label>
        <label className="flex items-center gap-2 text-xs">
          <input type="checkbox" checked={scale} onChange={(e) => setScale(e.target.checked)} />
          Scale (z-score across clusters)
        </label>
      </div>

      <WorkflowProgress
        active={q.isLoading}
        title="Computing dotplot"
        stages={[
          { label: 'Downloading markers_flat from MLflow', estSeconds: 5 },
          { label: 'Computing per-cluster mean expression', estSeconds: 2 },
        ]}
      />

      {q.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(q.error)}
        </div>
      )}

      {q.data && <Dotplot data={q.data} />}

      {/* summary.cluster_col is referenced indirectly via the response; this no-op keeps the
          `summary` arg meaningfully used by the lint pass and lets us add custom controls later. */}
      <span className="hidden">{summary.cluster_col}</span>
    </div>
  )
}

function Dotplot({ data }: { data: DotplotResponse }) {
  if (data.cells.length === 0) {
    return (
      <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
        No dotplot data to display.
      </div>
    )
  }
  const maxSize = Math.max(...data.cells.map((c) => c.size), 1)
  const trace = {
    type: 'scatter' as const,
    mode: 'markers' as const,
    x: data.cells.map((c) => c.gene),
    y: data.cells.map((c) => `Cluster ${c.cluster}`),
    marker: {
      color: data.cells.map((c) => c.expression),
      colorscale: data.color_scale,
      size: data.cells.map((c) => (c.size / maxSize) * 30 + 4),
      sizemode: 'diameter',
      line: { color: 'white', width: 0.5 },
      showscale: true,
      colorbar: { title: { text: data.color_label } },
    },
    hovertemplate:
      '<b>%{y}</b><br>%{x}<br>' + data.color_label + ': %{marker.color:.3f}<extra></extra>',
  }
  return (
    <div className="rounded-md border border-border bg-card p-2">
      <Plot
        data={[trace as never]}
        layout={{
          title: { text: 'Marker expression by cluster' },
          height: Math.max(400, data.clusters.length * 50),
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { size: 11 },
          xaxis: { tickangle: 45, gridcolor: '#333' },
          yaxis: { gridcolor: '#333', automargin: true },
          margin: { l: 100, r: 20, t: 60, b: 100 },
        }}
        config={{ displaylogo: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  )
}

// ─── Differential Expression ───────────────────────────────────────────────

type ScoredGene = DEGene & { score: number; isHighlighted: boolean }

export function DESubTab({ runId, summary }: { runId: string; summary: RunSummaryResponse }) {
  const [a, setA] = useState(summary.clusters[0] ?? '')
  const [b, setB] = useState(summary.clusters[1] ?? summary.clusters[0] ?? '')

  const de = useMutation({
    mutationFn: () => api.singleCellDE({ run_id: runId, cluster_a: a, cluster_b: b }),
  })

  // Saved cluster annotations (if the user already ran Cell Type Annotation),
  // surfaced here as a reference so you can tell which numbered cluster is
  // which cell type when picking A vs B.
  const annotations = useQuery({
    queryKey: ['de_saved_annotations', runId],
    queryFn: () => api.singleCellSavedAnnotations(runId),
  })
  const annoRows = useMemo(() => {
    const data = annotations.data
    const byCluster = new Map<
      string,
      { cluster: string; scim?: string; scimConf?: number; teddy?: string; teddyConf?: number; disease?: string }
    >()
    for (const r of data?.scimilarity?.annotations ?? []) {
      const row = byCluster.get(r.cluster) ?? { cluster: r.cluster }
      row.scim = r.predicted_cell_type
      row.scimConf = r.confidence_pct
      byCluster.set(r.cluster, row)
    }
    for (const r of data?.teddy?.annotations ?? []) {
      const row = byCluster.get(r.cluster) ?? { cluster: r.cluster }
      row.teddy = r.predicted_cell_type
      row.teddyConf = r.cell_type_confidence_pct
      row.disease = r.predicted_disease
      byCluster.set(r.cluster, row)
    }
    return [...byCluster.values()].sort(
      (x, y) => Number(x.cluster) - Number(y.cluster) || x.cluster.localeCompare(y.cluster),
    )
  }, [annotations.data])
  const hasAnno = annoRows.length > 0

  // Rank significant hits by a directional score = log2FC × -log10(p_adj):
  // effect size weighted by significance, signed so genes most strongly (and
  // significantly) enriched in cluster A sort to the top. Domain-agnostic; the
  // optional highlight set just flags + can float genes the user cares about.
  const [highlight, setHighlight] = useState<Highlight | null>(null)
  const [highlightFirst, setHighlightFirst] = useState(false)
  const scoredGenes = useMemo<ScoredGene[]>(() => {
    const hl = highlight?.genes ?? null
    const rows = (de.data?.genes ?? [])
      .filter((g) => g.significant)
      .map((g) => ({
        ...g,
        score: g.log2fc * g.neg_log10_p_adj,
        isHighlighted: hl ? hl.has(g.gene.toUpperCase()) : false,
      }))
    rows.sort((x, y) => {
      if (highlightFirst && x.isHighlighted !== y.isHighlighted) return x.isHighlighted ? -1 : 1
      return y.score - x.score
    })
    return rows
  }, [de.data, highlight, highlightFirst])
  const nHighlighted = useMemo(
    () => scoredGenes.filter((g) => g.isHighlighted).length,
    [scoredGenes],
  )

  const dirClass = (up: boolean) =>
    up ? 'text-rose-600 dark:text-rose-400' : 'text-sky-600 dark:text-sky-400'

  const tableColumns = useMemo<ColumnDef<ScoredGene, unknown>[]>(
    () => [
      {
        id: 'gene',
        header: 'Gene',
        cell: (ctx) => {
          const g = ctx.row.original
          return (
            <span className="flex items-center gap-1.5">
              <span className="font-medium">{g.gene}</span>
              {g.isHighlighted && (
                <span
                  title={highlight ? `In gene set: ${highlight.label}` : 'In highlighted gene set'}
                  className="rounded bg-yellow-400/20 px-1 text-[10px] font-semibold text-yellow-700 dark:text-yellow-400"
                >
                  ◆
                </span>
              )}
            </span>
          )
        },
      },
      {
        id: 'direction',
        header: 'Enriched in',
        cell: (ctx) => {
          const up = ctx.row.original.log2fc > 0
          return <span className={dirClass(up)}>{up ? `↑ ${a}` : `↑ ${b}`}</span>
        },
      },
      {
        id: 'score',
        header: 'Score',
        cell: (ctx) => {
          const s = ctx.row.original.score
          return (
            <span className={`font-medium tabular-nums ${dirClass(s > 0)}`}>{s.toFixed(2)}</span>
          )
        },
      },
      {
        id: 'log2fc',
        header: 'log2 FC',
        cell: (ctx) => ctx.row.original.log2fc.toFixed(3),
      },
      {
        id: 'p_adj',
        header: 'p-adj',
        cell: (ctx) =>
          ctx.row.original.p_adj < 1e-3
            ? ctx.row.original.p_adj.toExponential(2)
            : ctx.row.original.p_adj.toFixed(4),
      },
      {
        id: 'mean_a',
        header: `Mean ${a}`,
        cell: (ctx) => ctx.row.original.mean_a.toFixed(3),
      },
      {
        id: 'mean_b',
        header: `Mean ${b}`,
        cell: (ctx) => ctx.row.original.mean_b.toFixed(3),
      },
    ],
    [a, b, highlight],
  )

  return (
    <div className="space-y-4">
      <header>
        <h4 className="text-sm font-medium">Differential expression (cluster A vs B)</h4>
        <p className="text-xs text-muted-foreground">
          Mann-Whitney U per gene plus log2 FC of mean expression. p_adj is Benjamini-Hochberg.
          Significant = |log2FC| &gt; 1 and p_adj &lt; 0.05.
        </p>
      </header>

      <details open={hasAnno} className="rounded-md border border-border">
        <summary className="cursor-pointer px-4 py-2 text-sm font-medium">
          Cell-type annotation by cluster{hasAnno ? '' : ' — none yet'}
        </summary>
        <div className="p-3">
          {annotations.isLoading ? (
            <p className="text-xs text-muted-foreground">Loading saved annotations…</p>
          ) : !hasAnno ? (
            <p className="text-xs text-muted-foreground">
              No saved annotation for this run yet. Run <strong>Cell Type Annotation</strong> first —
              the predicted cell type per cluster will appear here so you can tell which numbered
              cluster is which when choosing A and B.
            </p>
          ) : (
            <div className="max-h-72 overflow-auto">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-card text-muted-foreground">
                  <tr className="text-left">
                    <th className="px-2 py-1">Cluster</th>
                    <th className="px-2 py-1">SCimilarity</th>
                    <th className="px-2 py-1">TEDDY</th>
                    <th className="px-2 py-1">TEDDY disease</th>
                  </tr>
                </thead>
                <tbody>
                  {annoRows.map((r) => (
                    <tr key={r.cluster} className="border-t border-border">
                      <td className="px-2 py-1 font-medium">{r.cluster}</td>
                      <td className="px-2 py-1">
                        {r.scim
                          ? `${r.scim}${r.scimConf != null ? ` (${r.scimConf.toFixed(0)}%)` : ''}`
                          : '—'}
                      </td>
                      <td className="px-2 py-1">
                        {r.teddy
                          ? `${r.teddy}${r.teddyConf != null ? ` (${r.teddyConf.toFixed(0)}%)` : ''}`
                          : '—'}
                      </td>
                      <td className="px-2 py-1">{r.disease ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </details>

      <div className="flex flex-wrap items-end gap-3">
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Cluster A
          </span>
          <select
            value={a}
            onChange={(e) => setA(e.target.value)}
            className="w-32 rounded-md border border-border bg-background px-3 py-2 text-sm"
          >
            {summary.clusters.map((c) => (
              <option key={c} value={c}>
                Cluster {c}
              </option>
            ))}
          </select>
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Cluster B
          </span>
          <select
            value={b}
            onChange={(e) => setB(e.target.value)}
            className="w-32 rounded-md border border-border bg-background px-3 py-2 text-sm"
          >
            {summary.clusters
              .filter((c) => c !== a)
              .map((c) => (
                <option key={c} value={c}>
                  Cluster {c}
                </option>
              ))}
          </select>
        </label>
        <button
          onClick={() => de.mutate()}
          disabled={!a || !b || a === b || de.isPending}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
        >
          {de.isPending ? 'Computing…' : 'Compute DE'}
        </button>
      </div>

      <WorkflowProgress
        active={de.isPending}
        title={`DE: Cluster ${a} vs ${b}`}
        stages={[
          { label: 'Downloading markers_flat from MLflow', estSeconds: 5 },
          { label: 'Mann-Whitney U per gene', estSeconds: 6 },
          { label: 'Adjusting p-values', estSeconds: 1 },
        ]}
      />

      {de.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(de.error)}
        </div>
      )}

      {de.data && de.data.warnings && de.data.warnings.length > 0 && (
        <details className="rounded-md border border-amber-500/40 bg-amber-500/10 p-2 text-xs">
          <summary className="cursor-pointer font-medium text-amber-700 dark:text-amber-400">
            ⚠ Differential expression result has {de.data.warnings.length} data-quality
            note{de.data.warnings.length === 1 ? '' : 's'} — click to expand
          </summary>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-muted-foreground">
            {de.data.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </details>
      )}

      <GeneHighlightPicker highlight={highlight} onChange={setHighlight} />

      {de.data && (
        <VolcanoPlot
          genes={de.data.genes}
          a={a}
          b={b}
          highlight={highlight?.genes ?? null}
          highlightLabel={highlight?.label ?? ''}
        />
      )}

      {de.data && (
        <div>
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div className="text-xs text-muted-foreground">
              {de.data.n_significant} significant genes (|log2 FC| &gt; 1, p_adj &lt; 0.05) · sorted
              by <strong>score</strong> = log2FC × −log10(p_adj), so genes most enriched in{' '}
              <span className={dirClass(true)}>↑ {a}</span> lead.
              {highlight && (
                <>
                  {' '}
                  {nHighlighted} are in{' '}
                  <span className="text-yellow-700 dark:text-yellow-400">◆ {highlight.label}</span>.
                </>
              )}
            </div>
            {highlight && (
              <label className="flex cursor-pointer items-center gap-1.5 text-xs">
                <input
                  type="checkbox"
                  checked={highlightFirst}
                  onChange={(e) => setHighlightFirst(e.target.checked)}
                />
                Highlighted genes first
              </label>
            )}
          </div>
          <p className="mb-2 text-[11px] text-muted-foreground">
            Tip: set <strong>Cluster A</strong> to the cell population you’re studying — its
            enriched genes then carry the highest positive scores and appear at the top.
          </p>
          <DataTable
            columns={tableColumns}
            data={scoredGenes.slice(0, 100)}
            emptyText="No genes meet the significance threshold."
          />
        </div>
      )}
    </div>
  )
}

function VolcanoPlot({
  genes,
  a,
  b,
  highlight,
  highlightLabel,
}: {
  genes: DEGene[]
  a: string
  b: string
  highlight: Set<string> | null
  highlightLabel: string
}) {
  if (genes.length === 0) return null
  const isHl = (g: DEGene) => (highlight ? highlight.has(g.gene.toUpperCase()) : false)
  const sig = genes.filter((g) => g.significant && !isHl(g))
  const nonsig = genes.filter((g) => !g.significant && !isHl(g))
  // Highlighted genes get their own labelled trace so they pop out of the
  // cloud regardless of significance.
  const cancer = genes.filter(isHl)
  const traces = [
    {
      type: 'scattergl' as const,
      mode: 'markers' as const,
      name: 'Not significant',
      x: nonsig.map((g) => g.log2fc),
      y: nonsig.map((g) => g.neg_log10_p_adj),
      marker: { size: 4, color: '#95A5A6', opacity: 0.5 },
      text: nonsig.map((g) => g.gene),
      hovertemplate: '<b>%{text}</b><br>log2FC %{x:.2f}<br>-log10 p_adj %{y:.2f}<extra></extra>',
    },
    {
      type: 'scattergl' as const,
      mode: 'markers' as const,
      name: 'Significant',
      x: sig.map((g) => g.log2fc),
      y: sig.map((g) => g.neg_log10_p_adj),
      marker: { size: 5, color: '#E74C3C', opacity: 0.85 },
      text: sig.map((g) => g.gene),
      hovertemplate: '<b>%{text}</b><br>log2FC %{x:.2f}<br>-log10 p_adj %{y:.2f}<extra></extra>',
    },
    {
      type: 'scattergl' as const,
      mode: 'markers+text' as const,
      name: `◆ ${highlightLabel || 'Highlighted'}`,
      x: cancer.map((g) => g.log2fc),
      y: cancer.map((g) => g.neg_log10_p_adj),
      marker: {
        size: 11,
        color: '#F1C40F',
        symbol: 'diamond',
        line: { width: 1, color: '#7A5C00' },
      },
      text: cancer.map((g) => g.gene),
      textposition: 'top center' as const,
      textfont: { size: 9 },
      hovertemplate:
        '<b>%{text}</b> (highlighted)<br>log2FC %{x:.2f}<br>-log10 p_adj %{y:.2f}<extra></extra>',
    },
  ]
  return (
    <div className="rounded-md border border-border bg-card p-2">
      <Plot
        data={traces.filter((t) => t.x.length > 0) as never}
        layout={{
          title: { text: `Volcano: Cluster ${a} vs ${b}` },
          height: 460,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { size: 11 },
          xaxis: { title: { text: 'log2 fold change' }, gridcolor: '#333' },
          yaxis: { title: { text: '-log10 adjusted p' }, gridcolor: '#333' },
          shapes: [
            { type: 'line', x0: 1, x1: 1, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: '#888' } },
            { type: 'line', x0: -1, x1: -1, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: '#888' } },
            { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: -Math.log10(0.05), y1: -Math.log10(0.05), line: { dash: 'dash', color: '#888' } },
          ],
          margin: { l: 60, r: 20, t: 50, b: 50 },
        }}
        config={{ displaylogo: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  )
}

// ─── Pathway Enrichment ────────────────────────────────────────────────────

const ENRICHMENT_DBS = [
  'GO_Biological_Process_2023',
  'KEGG_2021_Human',
  'Reactome_2022',
  'GO_Molecular_Function_2023',
  'GO_Cellular_Component_2023',
]

export function EnrichmentSubTab({
  runId,
  summary,
}: {
  runId: string
  summary: RunSummaryResponse
}) {
  const [cluster, setCluster] = useState(summary.clusters[0] ?? '')
  const [dbs, setDbs] = useState<string[]>([ENRICHMENT_DBS[0]])
  const env = useUserStore((s) => s.bootstrap?.env)
  const gmtDir = env
    ? `/Volumes/${env.catalog}/${env.schema_name}/scanpy_reference/genesets/`
    : '/Volumes/{catalog}/{schema}/scanpy_reference/genesets/'

  const enrich = useMutation({
    mutationFn: () => api.singleCellEnrichment({ run_id: runId, cluster, dbs }),
  })

  const tableColumns = useMemo<ColumnDef<EnrichmentTerm, unknown>[]>(
    () => [
      { id: 'term', header: 'Term', accessorKey: 'term' },
      { id: 'gene_set', header: 'Source', accessorKey: 'gene_set' },
      { id: 'overlap', header: 'Overlap', accessorKey: 'overlap' },
      {
        id: 'p_adj',
        header: 'p_adj',
        cell: (ctx) =>
          ctx.row.original.p_adj < 1e-3
            ? ctx.row.original.p_adj.toExponential(2)
            : ctx.row.original.p_adj.toFixed(4),
      },
    ],
    [],
  )

  const topTerms = (enrich.data?.terms ?? []).slice(0, 15)

  return (
    <div className="space-y-4">
      <header>
        <h4 className="text-sm font-medium">Pathway enrichment</h4>
        <p className="text-xs text-muted-foreground">
          Fisher's exact test on the cluster's top Wilcoxon marker genes (from
          {' '}<code>top_markers_per_cluster.csv</code>) against curated GMT databases
          (loaded from <code>{gmtDir}</code>).
        </p>
      </header>

      <div className="flex flex-wrap items-end gap-3">
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Cluster
          </span>
          <select
            size={5}
            value={cluster}
            onChange={(e) => setCluster(e.target.value)}
            className="w-32 rounded-md border border-border bg-background px-3 py-2 text-xs"
          >
            {summary.clusters.map((c) => (
              <option key={c} value={c}>
                Cluster {c}
              </option>
            ))}
          </select>
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Databases
          </span>
          <select
            multiple
            size={5}
            value={dbs}
            onChange={(e) => {
              const opts = Array.from(e.target.selectedOptions).map((o) => o.value)
              setDbs(opts)
            }}
            className="w-72 rounded-md border border-border bg-background px-3 py-2 text-xs"
          >
            {ENRICHMENT_DBS.map((db) => (
              <option key={db} value={db}>
                {db}
              </option>
            ))}
          </select>
        </label>
        <button
          onClick={() => enrich.mutate()}
          disabled={!cluster || dbs.length === 0 || enrich.isPending}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
        >
          {enrich.isPending ? 'Running…' : 'Run enrichment'}
        </button>
      </div>

      <WorkflowProgress
        active={enrich.isPending}
        title="Pathway enrichment"
        stages={[
          { label: 'Downloading markers_flat from MLflow', estSeconds: 5 },
          { label: 'Selecting top genes', estSeconds: 1 },
          { label: 'Loading GMT files from UC volume', estSeconds: 4 },
          { label: "Running Fisher's exact per term", estSeconds: 8 },
        ]}
      />

      {enrich.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(enrich.error)}
        </div>
      )}

      {enrich.data && enrich.data.terms.length === 0 && (
        <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
          No enriched terms.
        </div>
      )}
      {enrich.data && topTerms.length > 0 && <EnrichmentBar terms={topTerms} cluster={cluster} />}
      {enrich.data && enrich.data.terms.length > 0 && (
        <DataTable columns={tableColumns} data={enrich.data.terms.slice(0, 30)} />
      )}
    </div>
  )
}

function EnrichmentBar({ terms, cluster }: { terms: EnrichmentTerm[]; cluster: string }) {
  const sorted = [...terms].sort((a, b) => a.neg_log10_p_adj - b.neg_log10_p_adj)
  const trace = {
    type: 'bar' as const,
    orientation: 'h' as const,
    x: sorted.map((t) => t.neg_log10_p_adj),
    y: sorted.map((t) => t.term),
    marker: {
      color: sorted.map((t) => t.neg_log10_p_adj),
      colorscale: 'Viridis',
    },
    hovertemplate:
      '<b>%{y}</b><br>-log10 p_adj %{x:.2f}<br>' + sorted.map(() => '').join('') + '<extra></extra>',
  }
  return (
    <div className="rounded-md border border-border bg-card p-2">
      <Plot
        data={[trace as never]}
        layout={{
          title: { text: `Top enriched terms — Cluster ${cluster}` },
          height: Math.max(360, sorted.length * 28),
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { size: 10 },
          xaxis: { title: { text: '-log10 p_adj' }, gridcolor: '#333' },
          yaxis: { automargin: true },
          margin: { l: 200, r: 20, t: 50, b: 50 },
        }}
        config={{ displaylogo: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  )
}

// ─── Trajectory ────────────────────────────────────────────────────────────

export function TrajectorySubTab({
  runId,
  summary,
}: {
  runId: string
  summary: RunSummaryResponse
}) {
  const [gene, setGene] = useState<string | null>(null)

  const traj = useQuery({
    queryKey: ['sc', 'trajectory', runId, gene],
    queryFn: () => api.singleCellTrajectory({ run_id: runId, gene }),
    enabled: summary.has_pseudotime,
    staleTime: 60_000,
  })

  if (!summary.has_pseudotime) {
    return (
      <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
        Pseudotime data is not available for this run. Re-run processing with{' '}
        <strong className="text-foreground">Compute Pseudotime</strong> enabled.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <header>
        <h4 className="text-sm font-medium">Trajectory (diffusion pseudotime)</h4>
      </header>

      {traj.isLoading && <div className="text-sm text-muted-foreground">Loading…</div>}
      {traj.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(traj.error)}
        </div>
      )}

      {traj.data && traj.data.umap_points.length > 0 && (
        <div className="rounded-md border border-border bg-card p-2">
          <Plot
            data={[
              {
                type: 'scattergl' as const,
                mode: 'markers' as const,
                x: traj.data.umap_points.map((p) => p.umap_0),
                y: traj.data.umap_points.map((p) => p.umap_1),
                marker: {
                  color: traj.data.umap_points.map((p) => p.pseudotime),
                  colorscale: 'Viridis',
                  size: 3,
                  opacity: 0.75,
                  showscale: true,
                  colorbar: { title: { text: 'Pseudotime' } },
                },
                hovertemplate: 'Pseudotime: %{marker.color:.3f}<extra></extra>',
              } as never,
            ]}
            layout={{
              title: { text: 'UMAP — coloured by pseudotime' },
              height: 500,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { size: 11 },
              xaxis: { title: { text: 'UMAP_0' }, gridcolor: '#333' },
              yaxis: { title: { text: 'UMAP_1' }, gridcolor: '#333' },
              margin: { l: 50, r: 20, t: 50, b: 50 },
            }}
            config={{ displaylogo: false, responsive: true }}
            style={{ width: '100%' }}
            useResizeHandler
          />
        </div>
      )}

      <label className="block text-xs">
        <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
          Plot gene expression along pseudotime
        </span>
        <select
          value={gene ?? ''}
          onChange={(e) => setGene(e.target.value || null)}
          className="w-56 rounded-md border border-border bg-background px-3 py-2 text-sm"
        >
          <option value="">— pick a gene —</option>
          {summary.expr_genes.map((g) => (
            <option key={g} value={g}>
              {g}
            </option>
          ))}
        </select>
      </label>

      {traj.data && traj.data.gene_points.length > 0 && gene && (
        <div className="rounded-md border border-border bg-card p-2">
          {(() => {
            const xs = traj.data.gene_points.map((p) => p.pseudotime)
            const ys = traj.data.gene_points.map((p) => p.expression)
            const smooth = lowess(xs, ys, 0.3)
            return (
          <Plot
            data={[
              {
                type: 'scattergl' as const,
                mode: 'markers' as const,
                name: 'cells',
                x: xs,
                y: ys,
                marker: { size: 3, opacity: 0.5 },
                hovertemplate: 't %{x:.2f}<br>expr %{y:.2f}<extra></extra>',
                showlegend: false,
              } as never,
              {
                type: 'scatter' as const,
                mode: 'lines' as const,
                name: 'LOWESS',
                x: smooth.x,
                y: smooth.y,
                line: { color: '#ef4444', width: 2 },
                hoverinfo: 'skip',
                showlegend: false,
              } as never,
            ]}
            layout={{
              title: { text: `${gene} expression along pseudotime` },
              height: 360,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { size: 11 },
              xaxis: { title: { text: 'Pseudotime' }, gridcolor: '#333' },
              yaxis: { title: { text: `${gene} expression` }, gridcolor: '#333' },
              margin: { l: 60, r: 20, t: 40, b: 50 },
            }}
            config={{ displaylogo: false, responsive: true }}
            style={{ width: '100%' }}
            useResizeHandler
          />
            )
          })()}
        </div>
      )}
    </div>
  )
}

// LOWESS smoother — local linear regression with tricube weights.
// Span = fraction of points in each local neighborhood (statsmodels default 0.3).
// Used as the trendline overlay for "gene expression along pseudotime",
// 
function lowess(xs: number[], ys: number[], span = 0.3): { x: number[]; y: number[] } {
  const n = xs.length
  if (n === 0) return { x: [], y: [] }
  const order = Array.from(xs.keys()).sort((a, b) => xs[a] - xs[b])
  const sx = order.map((i) => xs[i])
  const sy = order.map((i) => ys[i])
  const k = Math.max(2, Math.min(n, Math.floor(span * n)))
  const out: number[] = new Array(n)
  for (let i = 0; i < n; i++) {
    const x0 = sx[i]
    const dists: { d: number; j: number }[] = []
    for (let j = 0; j < n; j++) dists.push({ d: Math.abs(sx[j] - x0), j })
    dists.sort((a, b) => a.d - b.d)
    const near = dists.slice(0, k)
    const maxD = near[near.length - 1].d || 1
    let Sw = 0, Swx = 0, Swy = 0, Swxx = 0, Swxy = 0
    for (const { d, j } of near) {
      const u = Math.min(d / maxD, 1)
      const w = Math.pow(1 - u * u * u, 3)
      const x = sx[j], y = sy[j]
      Sw += w; Swx += w * x; Swy += w * y; Swxx += w * x * x; Swxy += w * x * y
    }
    const denom = Sw * Swxx - Swx * Swx
    if (Math.abs(denom) < 1e-12) {
      out[i] = Sw > 0 ? Swy / Sw : 0
      continue
    }
    const m = (Sw * Swxy - Swx * Swy) / denom
    const b = (Swy - m * Swx) / Sw
    out[i] = m * x0 + b
  }
  return { x: sx, y: out }
}

// ─── QC & Outputs ──────────────────────────────────────────────────────────

export function QcSubTab({ summary }: { summary: RunSummaryResponse }) {
  return (
    <div className="space-y-3">
      <h4 className="text-sm font-medium">QC & other analysis outputs</h4>
      <p className="text-sm text-muted-foreground">
        Full QC plots, PCA, highly-variable genes, full-resolution UMAP, marker-gene heatmap, and
        the complete AnnData object are logged to the MLflow run by the processing notebook.
      </p>
      {summary.mlflow_run_url && (
        <a
          href={summary.mlflow_run_url}
          target="_blank"
          rel="noreferrer"
          className="inline-block rounded-md border border-primary bg-primary/10 px-4 py-2 text-sm text-primary hover:bg-primary/20"
        >
          Open MLflow Run (view all plots & artifacts) ↗
        </a>
      )}
      <ul className="ml-6 list-disc text-xs text-muted-foreground">
        <li>Quality control plots</li>
        <li>PCA & variance-explained plots</li>
        <li>Highly variable genes</li>
        <li>Full-resolution UMAP</li>
        <li>Marker-genes heatmap</li>
        <li>Complete AnnData object</li>
      </ul>
    </div>
  )
}

// ─── Raw Data ──────────────────────────────────────────────────────────────

export function RawDataSubTab({ runId, summary }: { runId: string; summary: RunSummaryResponse }) {
  const defaultCols = useMemo(() => {
    const out: string[] = [summary.cluster_col]
    if (summary.has_umap) out.push('UMAP_0', 'UMAP_1')
    out.push(...summary.expr_genes.slice(0, 3).map((g) => `expr_${g}`))
    return out
  }, [summary])

  const [columns, setColumns] = useState<string[]>(defaultCols)
  const [limit, setLimit] = useState(100)

  const q = useQuery({
    queryKey: ['sc', 'rawdata', runId, columns.join(','), limit],
    queryFn: () =>
      api.singleCellRawData({ run_id: runId, columns, limit }),
    staleTime: 30_000,
    enabled: columns.length > 0,
  })

  const tableColumns = useMemo<ColumnDef<Record<string, unknown>, unknown>[]>(
    () =>
      (q.data?.columns ?? columns).map((c) => ({
        id: c,
        header: c,
        cell: (ctx) => {
          const v = ctx.row.original[c]
          if (v === null || v === undefined) return ''
          if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3)
          return String(v)
        },
      })),
    [q.data, columns],
  )

  const downloadCsv = () => {
    if (!q.data) return
    const header = q.data.columns.join(',')
    const lines = q.data.rows.map((r) =>
      q.data.columns
        .map((c) => {
          const v = r[c]
          if (v === null || v === undefined) return ''
          const s = String(v)
          return s.includes(',') ? `"${s.replace(/"/g, '""')}"` : s
        })
        .join(','),
    )
    const blob = new Blob([header + '\n' + lines.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `singlecell_run_${runId.slice(0, 8)}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-4">
      <header className="flex items-baseline justify-between">
        <h4 className="text-sm font-medium">Raw data table</h4>
        {q.data && (
          <button
            onClick={downloadCsv}
            className="rounded-md border border-border px-3 py-1.5 text-xs hover:bg-accent"
          >
            Download CSV (first {limit} rows)
          </button>
        )}
      </header>

      <div className="flex flex-wrap items-end gap-3">
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Columns
          </span>
          <select
            multiple
            size={8}
            value={columns}
            onChange={(e) => {
              const opts = Array.from(e.target.selectedOptions).map((o) => o.value)
              setColumns(opts)
            }}
            className={cn(
              'w-96 rounded-md border border-border bg-background px-3 py-2 font-mono text-xs',
            )}
          >
            {summary.all_columns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Row limit
          </span>
          <select
            value={limit}
            onChange={(e) => setLimit(parseInt(e.target.value))}
            className="rounded-md border border-border bg-background px-3 py-2 text-sm"
          >
            {[50, 100, 200, 500, 1000, 2000].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      </div>

      <WorkflowProgress
        active={q.isLoading}
        title="Fetching rows"
        stages={[{ label: 'Downloading markers_flat from MLflow', estSeconds: 5 }]}
      />

      {q.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(q.error)}
        </div>
      )}
      {q.data && (
        <>
          <div className="text-xs text-muted-foreground">
            Showing {q.data.rows.length} of {q.data.total_cells.toLocaleString()} cells.
          </div>
          <DataTable columns={tableColumns} data={q.data.rows} />
        </>
      )}

    </div>
  )
}

// Avoid unused-import warnings when ColorPointsResponse is only used for inference.
export type _LintHelper = ColorPointsResponse | DotplotResponse

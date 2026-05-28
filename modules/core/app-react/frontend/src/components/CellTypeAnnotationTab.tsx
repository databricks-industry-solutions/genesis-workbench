import { useMemo, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'
import { PlotlyChart as Plot } from '@/components/PlotlyChart'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { SingleCellRunPicker } from '@/components/SingleCellRunPicker'
import { WorkflowProgress } from '@/components/WorkflowProgress'
import type { ClusterAnnotation, SingleCellRun, UmapPoint } from '@/types/api'

export function CellTypeAnnotationTab() {
  const [run, setRun] = useState<SingleCellRun | null>(null)
  const [cellsPerCluster, setCellsPerCluster] = useState(10)
  const [kNeighbors, setKNeighbors] = useState(20)

  const annotate = useMutation({
    mutationFn: () => {
      if (!run) throw new Error('Pick a run first')
      return api.singleCellAnnotate({
        run_id: run.run_id,
        cells_per_cluster: cellsPerCluster,
        k_neighbors: kNeighbors,
      })
    },
  })

  const annotationColumns = useMemo<ColumnDef<ClusterAnnotation, unknown>[]>(
    () => [
      { id: 'cluster', header: 'Cluster', accessorKey: 'cluster' },
      { id: 'predicted_cell_type', header: 'Predicted Cell Type', accessorKey: 'predicted_cell_type' },
      {
        id: 'confidence',
        header: 'Confidence',
        cell: (ctx) => `${ctx.row.original.confidence_pct.toFixed(0)}%`,
      },
      { id: 'top_predictions', header: 'Top Predictions', accessorKey: 'top_predictions' },
    ],
    [],
  )

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Cell Type Annotation (SCimilarity)</h3>
        <p className="text-xs text-muted-foreground">
          Annotate clusters from a completed processing run using SCimilarity's 23M-cell
          reference. Samples N cells per cluster, embeds them via the SCimilarity endpoints,
          searches the Vector Search index for nearest neighbors, then majority-votes per
          cluster.
        </p>
      </div>

      <div className="flex flex-col gap-3 md:flex-row md:items-end">
        <div className="flex-1">
          <SingleCellRunPicker value={run?.run_id ?? null} onChange={setRun} />
        </div>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Cells per cluster
          </span>
          <input
            type="number"
            min={3}
            max={50}
            step={1}
            value={cellsPerCluster}
            onChange={(e) => setCellsPerCluster(parseInt(e.target.value || '10'))}
            className="w-32 rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Neighbors (k)
          </span>
          <input
            type="number"
            min={5}
            max={200}
            step={5}
            value={kNeighbors}
            onChange={(e) => setKNeighbors(parseInt(e.target.value || '20'))}
            className="w-32 rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
        </label>
        <button
          onClick={() => annotate.mutate()}
          disabled={!run || annotate.isPending}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
        >
          {annotate.isPending ? 'Annotating…' : 'Annotate clusters'}
        </button>
      </div>

      <WorkflowProgress
        active={annotate.isPending}
        title="SCimilarity annotation pipeline"
        stages={[
          { label: 'Loading markers_flat from MLflow', estSeconds: 4 },
          { label: 'Fetching gene order', estSeconds: 2 },
          { label: 'Generating cell embeddings (batched)', estSeconds: 12 },
          { label: 'Vector Search nearest-neighbour lookup per cell', estSeconds: 8 },
          { label: 'Majority-vote per cluster', estSeconds: 1 },
        ]}
        note="Endpoint cold-starts can add 20–60s. Pre-warm via Settings → Endpoint Management."
      />

      {annotate.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(annotate.error)}
        </div>
      )}

      {annotate.data && (
        <section className="space-y-4 border-t border-border pt-4">
          <h4 className="text-sm font-medium">
            Annotated {annotate.data.annotations.length} cluster
            {annotate.data.annotations.length === 1 ? '' : 's'}
          </h4>

          <DataTable
            columns={annotationColumns}
            data={annotate.data.annotations}
            emptyText="No clusters annotated."
          />

          {annotate.data.umap_points.length > 0 && <UmapScatter points={annotate.data.umap_points} />}
        </section>
      )}
    </div>
  )
}

function UmapScatter({ points }: { points: UmapPoint[] }) {
  // Group points by predicted_cell_type so each type gets its own legend entry.
  const traces = useMemo(() => {
    const byType = new Map<string, { x: number[]; y: number[]; clusters: string[] }>()
    for (const p of points) {
      const t = p.predicted_cell_type || 'Unknown'
      const bucket = byType.get(t) ?? { x: [], y: [], clusters: [] }
      bucket.x.push(p.umap_0)
      bucket.y.push(p.umap_1)
      bucket.clusters.push(p.cluster)
      byType.set(t, bucket)
    }
    return [...byType.entries()].map(([cellType, bucket]) => ({
      type: 'scattergl' as const,
      mode: 'markers' as const,
      name: cellType,
      x: bucket.x,
      y: bucket.y,
      text: bucket.clusters.map((c) => `Cluster ${c}`),
      hovertemplate: '<b>%{text}</b><br>%{fullData.name}<extra></extra>',
      marker: { size: 4, opacity: 0.75 },
    }))
  }, [points])

  return (
    <div className="rounded-md border border-border bg-card p-2">
      <Plot
        data={traces}
        layout={{
          title: { text: 'UMAP — Predicted Cell Types', font: { color: '#e6e6e6' } },
          height: 560,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: '#e6e6e6', size: 11 },
          xaxis: { title: { text: 'UMAP_0' }, gridcolor: '#333' },
          yaxis: { title: { text: 'UMAP_1' }, gridcolor: '#333' },
          legend: { font: { size: 11 } },
          margin: { l: 50, r: 20, t: 50, b: 50 },
        }}
        config={{ displaylogo: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  )
}

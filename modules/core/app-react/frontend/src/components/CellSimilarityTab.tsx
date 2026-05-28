import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { PlotlyChart as Plot } from '@/components/PlotlyChart'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { RealtimeProgress } from '@/components/RealtimeProgress'
import { useSseMutation } from '@/hooks/useSseMutation'
import type { CategoryCount, SimilarityResponse } from '@/types/api'

type Reference = 'SCimilarity' | 'TEDDY'

/** Runs against the run selected by the outer Analysis tab. Pass null to
 * render an empty-state nudge instead of the form. */
export function CellSimilarityTab({ runId }: { runId: string | null }) {
  const [cluster, setCluster] = useState<string>('')
  const [reference, setReference] = useState<Reference>('SCimilarity')
  const [kNeighbors, setKNeighbors] = useState(100)

  const runInfo = useQuery({
    queryKey: ['single_cell', 'run-info', runId],
    queryFn: () => api.singleCellRunInfo(runId!),
    enabled: Boolean(runId),
    staleTime: 5 * 60_000,
  })

  // Reset the selected cluster whenever a new run loads.
  useEffect(() => {
    if (runInfo.data?.clusters.length) setCluster(runInfo.data.clusters[0])
    else setCluster('')
  }, [runInfo.data])

  const search = useSseMutation<
    {
      run_id: string
      cluster: string
      k_neighbors: number
      cells_per_cluster: number
      reference: 'scimilarity' | 'teddy'
    },
    SimilarityResponse
  >('/api/single_cell/similarity/stream')

  const runSearch = () => {
    if (!runId || !cluster) return
    search.start({
      run_id: runId,
      cluster,
      k_neighbors: kNeighbors,
      cells_per_cluster: 20,
      reference: reference === 'TEDDY' ? 'teddy' : 'scimilarity',
    })
  }

  if (!runId) {
    return (
      <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
        Select a completed run above to run cell similarity search.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Cell Similarity Search</h3>
        <p className="text-xs text-muted-foreground">
          For each cell in the selected cluster, embed via the chosen foundation model and search
          its reference index for nearest neighbours. SCimilarity uses a 23M-cell pan-tissue
          reference (column: <code>study</code>); TEDDY uses a curated CELLxGENE subset (column:{' '}
          <code>dataset_id</code>) and also annotates disease.
        </p>
      </div>

      <div className="flex flex-col gap-3 md:flex-row md:items-end">
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Cluster</span>
          <select
            value={cluster}
            onChange={(e) => setCluster(e.target.value)}
            disabled={!runInfo.data}
            className="w-40 rounded-md border border-border bg-background px-3 py-2 text-sm disabled:opacity-50"
          >
            {runInfo.data?.clusters.map((c) => (
              <option key={c} value={c}>
                Cluster {c}
              </option>
            )) ?? <option>—</option>}
          </select>
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Reference</span>
          <div className="flex gap-1">
            {(['SCimilarity', 'TEDDY'] as const).map((r) => (
              <button
                key={r}
                onClick={() => setReference(r)}
                className={
                  'rounded-md border px-3 py-2 text-xs transition-colors ' +
                  (r === reference
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border text-muted-foreground hover:bg-accent')
                }
              >
                {r}
              </button>
            ))}
          </div>
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Neighbours (k)
          </span>
          <input
            type="number"
            min={10}
            max={1000}
            step={50}
            value={kNeighbors}
            onChange={(e) => setKNeighbors(parseInt(e.target.value || '100'))}
            className="w-28 rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
        </label>
        <button
          onClick={runSearch}
          disabled={!runId || !cluster || search.isPending}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
        >
          {search.isPending ? 'Searching…' : 'Search'}
        </button>
      </div>

      {search.isPending && (
        <RealtimeProgress
          title={`Cell similarity search (${reference})`}
          pct={search.progress?.pct ?? 0}
          msg={search.progress?.msg ?? 'Starting…'}
          stages={
            reference === 'TEDDY'
              ? [
                  { label: 'Loading TEDDY reference assets', pctEnd: 10 },
                  { label: 'Embedding sampled cells', pctEnd: 40 },
                  { label: 'Vector Search nearest-neighbours per cell', pctEnd: 98 },
                  { label: 'Aggregating metadata', pctEnd: 100 },
                ]
              : [
                  { label: 'Fetching SCimilarity gene order', pctEnd: 15 },
                  { label: 'Embedding sampled cells', pctEnd: 40 },
                  { label: 'Vector Search nearest-neighbours per cell', pctEnd: 98 },
                  { label: 'Aggregating metadata', pctEnd: 100 },
                ]
          }
        />
      )}

      {search.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(search.error)}
        </div>
      )}

      {search.data && (
        <section className="space-y-4 border-t border-border pt-4">
          <h4 className="text-sm font-medium">
            {search.data.total_neighbors.toLocaleString()} neighbours from Cluster {cluster}
          </h4>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <CountsBar
              title="Neighbour cell types"
              rows={search.data.cell_types}
              colorScale="Viridis"
            />
            <CountsBar
              title="Neighbour disease distribution"
              rows={search.data.diseases}
              colorScale="Reds"
            />
          </div>

          {search.data.tissues.length > 0 && (
            <CountsBar
              title="Neighbour tissue distribution"
              rows={search.data.tissues}
              colorScale="Blues"
            />
          )}

          {search.data.sources.length > 0 && (
            <details className="rounded-md border border-border">
              <summary className="cursor-pointer px-4 py-2 text-sm">Neighbour source studies</summary>
              <div className="p-3">
                <DataTable
                  columns={[
                    { id: 'name', header: 'Study', accessorKey: 'name' },
                    { id: 'count', header: 'Count', accessorKey: 'count' },
                  ]}
                  data={search.data.sources}
                />
              </div>
            </details>
          )}
        </section>
      )}
    </div>
  )
}

function CountsBar({
  title,
  rows,
  colorScale,
}: {
  title: string
  rows: CategoryCount[]
  colorScale: string
}) {
  if (rows.length === 0) {
    return (
      <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
        {title}: no data.
      </div>
    )
  }
  const top = rows.slice(0, 15).reverse() // bar charts read bottom-to-top
  return (
    <div className="rounded-md border border-border bg-card p-2">
      <Plot
        data={[
          {
            type: 'bar',
            orientation: 'h',
            x: top.map((r) => r.count),
            y: top.map((r) => r.name),
            marker: {
              color: top.map((r) => r.count),
              colorscale: colorScale,
            },
            hovertemplate: '<b>%{y}</b><br>%{x} neighbours<extra></extra>',
          } as any,
        ]}
        layout={{
          title: { text: title, font: { color: '#e6e6e6', size: 13 } },
          height: 380,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: '#e6e6e6', size: 10 },
          margin: { l: 160, r: 20, t: 40, b: 30 },
          xaxis: { gridcolor: '#333' },
          yaxis: { automargin: true },
        }}
        config={{ displaylogo: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  )
}

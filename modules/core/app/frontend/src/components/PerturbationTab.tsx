import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'
import { PlotlyChart as Plot } from '@/components/PlotlyChart'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { NarrativePanel } from '@/components/NarrativePanel'
import { RealtimeProgress } from '@/components/RealtimeProgress'
import { WorkflowProgress } from '@/components/WorkflowProgress'
import { useSseMutation } from '@/hooks/useSseMutation'
import { ClipboardPaste } from '@/components/ClipboardPaste'
import type { PerturbationGene, PerturbationResponse } from '@/types/api'

type PerturbType = 'knockout' | 'overexpress'

/** Runs against the run selected by the outer Analysis tab. */
export function PerturbationTab({ runId }: { runId: string | null }) {
  const [cluster, setCluster] = useState<string>('')
  const [perturbType, setPerturbType] = useState<PerturbType>('knockout')
  const [selectedGenes, setSelectedGenes] = useState<string[]>([])
  const [extraGenes, setExtraGenes] = useState<string>('')

  const runInfo = useQuery({
    queryKey: ['single_cell', 'run-info', runId, 'perturbation'],
    queryFn: () => api.singleCellRunInfo(runId!, 50),
    enabled: Boolean(runId),
    staleTime: 5 * 60_000,
  })

  useEffect(() => {
    if (runInfo.data?.clusters.length) setCluster(runInfo.data.clusters[0])
    else setCluster('')
    setSelectedGenes([])
    setExtraGenes('')
  }, [runInfo.data])

  const topGenesForCluster = useMemo(() => {
    if (!cluster || !runInfo.data) return []
    return runInfo.data.top_genes_by_cluster[cluster] ?? []
  }, [runInfo.data, cluster])

  const allGenesToPerturb = useMemo(() => {
    const extras = extraGenes
      .split(',')
      .map((g) => g.trim().toUpperCase())
      .filter(Boolean)
    return Array.from(new Set([...selectedGenes, ...extras]))
  }, [selectedGenes, extraGenes])

  const predict = useSseMutation<
    {
      run_id: string
      cluster: string
      perturbation_type: PerturbType
      genes_to_perturb: string[]
    },
    PerturbationResponse
  >('/api/single_cell/perturbation/stream')

  const runPredict = () => {
    if (!runId || !cluster || allGenesToPerturb.length === 0) return
    predict.start({
      run_id: runId,
      cluster,
      perturbation_type: perturbType,
      genes_to_perturb: allGenesToPerturb,
    })
  }

  const resultRows = predict.data?.results ?? []

  // Paste genes collected on the Clipboard straight into the perturbation targets.
  const mergeIntoExtra = (incoming: string[]) =>
    setExtraGenes((cur) => {
      const set = new Set(
        cur
          .split(',')
          .map((s) => s.trim().toUpperCase())
          .filter(Boolean),
      )
      incoming.forEach((g) => set.add(g.toUpperCase()))
      return Array.from(set).join(', ')
    })

  // Predicted cell type for the perturbed cluster (if annotation was run) —
  // gives the narrative biological context.
  const annoQ = useQuery({
    queryKey: ['sc', 'pert-anno', runId],
    queryFn: () => api.singleCellSavedAnnotations(runId!),
    enabled: Boolean(runId),
  })
  const cellTypeForCluster = useMemo(() => {
    const scim = annoQ.data?.scimilarity?.annotations.find((a) => a.cluster === cluster)
    const teddy = annoQ.data?.teddy?.annotations.find((a) => a.cluster === cluster)
    return scim?.predicted_cell_type || teddy?.predicted_cell_type || null
  }, [annoQ.data, cluster])

  // AI narrative (Claude Opus 4.8) interpreting the current result.
  const narrative = useMutation({
    mutationFn: () => {
      const d = predict.data!
      const top = [...d.results]
        .filter((r) => r.delta != null)
        .sort((x, y) => (y.abs_delta ?? 0) - (x.abs_delta ?? 0))
        .slice(0, 30)
        .map((r) => ({ gene: r.gene_name, delta: r.delta as number }))
      return api.singleCellPerturbationNarrative({
        run_id: runId ?? undefined,
        cluster,
        perturbation_type: perturbType,
        genes_to_perturb: allGenesToPerturb,
        cell_type: cellTypeForCluster,
        summary_total_genes: d.summary_total_genes,
        summary_significant_count: d.summary_significant_count,
        summary_max_abs_delta: d.summary_max_abs_delta,
        top_genes: top,
      })
    },
  })

  // Auto-interpret once a (non-empty) result lands — no button click needed.
  useEffect(() => {
    if (predict.data && predict.data.summary_total_genes > 0) narrative.mutate()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predict.data])

  // Pattern: the AI interpretation is the final stage — keep a progress
  // indicator up through "Interpreting result" and reveal the result only once
  // the narrative has settled (so the page appears complete, not mid-spinner).
  const willNarrate = (predict.data?.summary_total_genes ?? 0) > 0
  const narrativeSettled = narrative.isSuccess || narrative.isError
  const interpreting = Boolean(predict.data) && willNarrate && !narrativeSettled
  const showResult = Boolean(predict.data) && (!willNarrate || narrativeSettled)

  const tableColumns = useMemo<ColumnDef<PerturbationGene, unknown>[]>(
    () => [
      { id: 'gene_name', header: 'Gene', accessorKey: 'gene_name' },
      {
        id: 'original_expression',
        header: 'Original',
        cell: (ctx) => fmt(ctx.row.original.original_expression),
      },
      {
        id: 'predicted_expression',
        header: 'Predicted',
        cell: (ctx) => fmt(ctx.row.original.predicted_expression),
      },
      {
        id: 'delta',
        header: 'Δ',
        cell: (ctx) => fmt(ctx.row.original.delta),
      },
      {
        id: 'abs_delta',
        header: '|Δ|',
        cell: (ctx) => fmt(ctx.row.original.abs_delta),
      },
    ],
    [],
  )

  if (!runId) {
    return (
      <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
        Select a completed run above to predict gene perturbations.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Predict Gene Perturbation Effects</h3>
        <p className="text-xs text-muted-foreground">
          Predict the effect of knocking out or overexpressing one or more genes on the cluster's
          mean expression profile using scGPT.
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
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Perturbation type
          </span>
          <div className="flex gap-1">
            {(['knockout', 'overexpress'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setPerturbType(t)}
                className={
                  'rounded-md border px-3 py-2 text-xs transition-colors capitalize ' +
                  (t === perturbType
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border text-muted-foreground hover:bg-accent')
                }
              >
                {t}
              </button>
            ))}
          </div>
        </label>
      </div>

      <div className="flex items-center gap-2 text-xs">
        <span className="text-muted-foreground">Add targets from your Clipboard:</span>
        <ClipboardPaste
          kind="gene"
          label="Paste gene"
          onPick={(it) => mergeIntoExtra([it.value])}
        />
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Pick from top-expressed marker genes in this cluster
          </span>
          <select
            multiple
            value={selectedGenes}
            onChange={(e) => {
              const opts = Array.from(e.target.selectedOptions).map((o) => o.value)
              setSelectedGenes(opts)
            }}
            disabled={!cluster || topGenesForCluster.length === 0}
            size={8}
            className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs disabled:opacity-50"
          >
            {topGenesForCluster.map((g) => (
              <option key={g.gene} value={g.gene}>
                {g.gene} ({g.mean_expr.toFixed(2)})
              </option>
            ))}
          </select>
          <span className="mt-1 block text-[10px] text-muted-foreground">
            Cmd/Ctrl-click to multi-select. Genes ranked by mean expression in the cluster.
          </span>
        </label>

        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Or add custom genes (comma-separated)
          </span>
          <input
            type="text"
            value={extraGenes}
            onChange={(e) => setExtraGenes(e.target.value)}
            placeholder="e.g. TP53, BRCA1"
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
          <span className="mt-1 block text-[10px] text-muted-foreground">
            Must be in scGPT's vocabulary; otherwise the model ignores them.
          </span>
        </label>
      </div>

      <div className="flex items-center justify-between">
        <div className="text-xs text-muted-foreground">
          Will perturb: <strong className="font-mono text-foreground">{allGenesToPerturb.join(', ') || '—'}</strong>
        </div>
        <button
          onClick={runPredict}
          disabled={!runId || !cluster || allGenesToPerturb.length === 0 || predict.isPending}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
        >
          {predict.isPending ? 'Predicting…' : 'Predict perturbation effect'}
        </button>
      </div>

      {predict.isPending && (
        <RealtimeProgress
          title="scGPT perturbation prediction"
          pct={predict.progress?.pct ?? 0}
          msg={predict.progress?.msg ?? 'Starting…'}
          stages={[
            { label: 'Computing cluster mean expression', pctEnd: 30 },
            { label: 'Calling scGPT Perturbation endpoint', pctEnd: 85 },
            { label: 'Sorting + summarising results', pctEnd: 100 },
          ]}
        />
      )}

      {interpreting && (
        <WorkflowProgress
          active
          title="scGPT perturbation prediction"
          stages={[{ label: 'Interpreting Results', estSeconds: 8 }]}
        />
      )}

      {predict.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(predict.error)}
        </div>
      )}

      {showResult && predict.data && (
        <section className="space-y-4 border-t border-border pt-4">
          {predict.data.summary_total_genes === 0 ? (
            <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
              The endpoint returned no rows. The selected gene(s) may not be in the scGPT vocabulary.
            </div>
          ) : (
            <>
              <NarrativePanel
                isPending={narrative.isPending}
                data={narrative.data}
                error={narrative.error}
                onRegenerate={() => narrative.mutate()}
              />

              <div className="grid grid-cols-3 gap-3">
                <Metric label="Total genes analysed" value={predict.data.summary_total_genes.toLocaleString()} />
                <Metric
                  label="Significantly affected (top 5%)"
                  value={predict.data.summary_significant_count.toLocaleString()}
                />
                <Metric
                  label="Max |Δ|"
                  value={predict.data.summary_max_abs_delta.toFixed(4)}
                />
              </div>

              {resultRows.length > 0 && (
                <>
                  <DeltaBar rows={resultRows.slice(0, 20)} />
                  <OrigVsPredictedScatter rows={resultRows} />
                </>
              )}

              <details className="rounded-md border border-border">
                <summary className="cursor-pointer px-4 py-2 text-sm">Full results table</summary>
                <div className="p-3">
                  <DataTable columns={tableColumns} data={resultRows.slice(0, 100)} />
                </div>
              </details>
            </>
          )}
        </section>
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-muted/30 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-lg font-medium">{value}</div>
    </div>
  )
}

function fmt(value: number | null): string {
  if (value === null || Number.isNaN(value)) return '—'
  return Math.abs(value) < 1e-3 ? value.toExponential(2) : value.toFixed(4)
}

function DeltaBar({ rows }: { rows: PerturbationGene[] }) {
  const ordered = [...rows].reverse()
  return (
    <div className="rounded-md border border-border bg-card p-2">
      <Plot
        data={[
          {
            type: 'bar',
            orientation: 'h',
            x: ordered.map((r) => r.delta ?? 0),
            y: ordered.map((r) => r.gene_name),
            marker: {
              color: ordered.map((r) => r.delta ?? 0),
              colorscale: 'RdBu',
              cmid: 0,
            },
            hovertemplate: '<b>%{y}</b><br>Δ = %{x:.4f}<extra></extra>',
          } as any,
        ]}
        layout={{
          title: { text: 'Top affected genes (signed Δ)', font: { size: 13 } },
          height: Math.max(360, ordered.length * 22),
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { size: 10 },
          margin: { l: 120, r: 20, t: 40, b: 30 },
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

function OrigVsPredictedScatter({ rows }: { rows: PerturbationGene[] }) {
  const filtered = rows.filter(
    (r) => r.original_expression !== null && r.predicted_expression !== null,
  )
  if (filtered.length === 0) return null
  const maxVal = Math.max(
    ...filtered.map((r) => Math.max(r.original_expression ?? 0, r.predicted_expression ?? 0)),
    1,
  )
  return (
    <div className="rounded-md border border-border bg-card p-2">
      <Plot
        data={[
          {
            type: 'scattergl',
            mode: 'markers',
            x: filtered.map((r) => r.original_expression ?? 0),
            y: filtered.map((r) => r.predicted_expression ?? 0),
            text: filtered.map((r) => `${r.gene_name} (Δ=${(r.delta ?? 0).toFixed(4)})`),
            hovertemplate: '%{text}<extra></extra>',
            marker: { size: 5, opacity: 0.7 },
          } as any,
          {
            type: 'scatter',
            mode: 'lines',
            x: [0, maxVal],
            y: [0, maxVal],
            line: { color: '#888', dash: 'dash' },
            hoverinfo: 'skip',
            showlegend: false,
          } as any,
        ]}
        layout={{
          title: { text: 'Original vs Predicted expression', font: { size: 13 } },
          height: 400,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { size: 10 },
          margin: { l: 50, r: 20, t: 40, b: 50 },
          xaxis: { title: { text: 'Original' }, gridcolor: '#333' },
          yaxis: { title: { text: 'Predicted' }, gridcolor: '#333' },
          showlegend: false,
        }}
        config={{ displaylogo: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  )
}

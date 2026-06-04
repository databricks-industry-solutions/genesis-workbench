import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { CellSimilarityTab } from '@/components/CellSimilarityTab'
import { PerturbationTab } from '@/components/PerturbationTab'
import { SingleCellRunPicker } from '@/components/SingleCellRunPicker'
import { Tabs } from '@/components/Tabs'
import { WorkflowProgress } from '@/components/WorkflowProgress'
import {
  DESubTab,
  EnrichmentSubTab,
  MarkerDotplotSubTab,
  TrajectorySubTab,
  UmapSubTab,
} from '@/components/RunViewSubTabs'
import type { SingleCellRun } from '@/types/api'

/** Single Cell → Analysis tab.
 *
 * One shared run picker drives all the analysis sub-tabs (UMAP, Markers, DE,
 * Pathway Enrichment, Trajectory, Cell Similarity, Perturbation Prediction)
 * — replaces the per-tab pickers from the older layout and the embedded
 * sub-tab cluster that lived inside Raw Processing → View Loaded Run.
 *
 * QC & Outputs and Raw Data are *not* here on purpose — they're metadata
 * about a single run's processing, so they live in the popup dialog on the
 * Raw Processing tab (where you pick a specific run to inspect). */
export function AnalysisTab() {
  const [run, setRun] = useState<SingleCellRun | null>(null)

  return (
    <div className="space-y-4">
      <SingleCellRunPicker value={run?.run_id ?? null} onChange={setRun} />

      {run && (
        <div className="space-y-0.5 text-sm font-semibold">
          <div>Experiment: <code className="font-mono">{run.experiment_name}</code></div>
          <div>Mode: <code className="font-mono">{run.processing_mode}</code></div>
        </div>
      )}

      {!run ? (
        <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
          Pick a completed run above to run UMAP, markers, DE, pathway enrichment,
          trajectory, cell-similarity, and perturbation analyses against it.
        </div>
      ) : (
        <AnalysisBody run={run} />
      )}
    </div>
  )
}

function AnalysisBody({ run }: { run: SingleCellRun }) {
  // Five of the analysis sub-tabs need the run summary (clusters, expr_genes,
  // umap_points, …). Cell Similarity and Perturbation only need the run id +
  // run-info; they fetch it themselves.
  const summary = useQuery({
    queryKey: ['single_cell', 'run-summary', run.run_id],
    queryFn: () => api.singleCellRunSummary(run.run_id),
    staleTime: 5 * 60_000,
  })

  if (summary.isLoading) {
    return (
      <WorkflowProgress
        active
        title="Loading results"
        stages={[{ label: 'Downloading markers_flat.parquet from MLflow', estSeconds: 5 }]}
      />
    )
  }
  if (summary.error) {
    return (
      <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
        {String(summary.error)}
      </div>
    )
  }
  if (!summary.data) return null

  const s = summary.data
  return (
    <>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Metric
          label="Total cells"
          value={
            s.cells_total != null
              ? s.cells_total.toLocaleString()
              : `${s.cells_subsample.toLocaleString()}*`
          }
        />
        <Metric
          label={s.cells_total != null ? 'Subsample loaded' : 'Subsample loaded*'}
          value={s.cells_subsample.toLocaleString()}
        />
        <Metric label="Clusters" value={String(s.clusters_count)} />
        <Metric label="Marker genes" value={String(s.markers_count)} />
      </div>

      {s.key_metrics.length > 0 && (
        <div className="rounded-md border border-border bg-card p-3">
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Key MLflow metrics
          </div>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs md:grid-cols-3">
            {s.key_metrics.map((m) => (
              <div key={m.label} className="flex justify-between gap-3">
                <dt className="text-muted-foreground">{m.label}</dt>
                <dd className="font-medium">{m.value}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      <Tabs
        tabs={[
        // Ordered to follow the discovery flow:
        // UMAP (how many populations?) → Markers/Similarity (what are they?) →
        // DE (which genes differ?) → Enrichment (what programs?) →
        // Trajectory (how did it get there?) → Perturbation (what if I hit a target?).
        {
          id: 'umap',
          label: 'UMAP',
          content: <UmapSubTab runId={run.run_id} summary={s} />,
        },
        {
          id: 'markers',
          label: 'Marker Genes',
          content: <MarkerDotplotSubTab runId={run.run_id} summary={s} />,
        },
        {
          id: 'similarity',
          label: 'Cell Similarity',
          content: <CellSimilarityTab runId={run.run_id} />,
        },
        {
          id: 'de',
          label: 'Differential Expression',
          content: <DESubTab runId={run.run_id} summary={s} />,
        },
        {
          id: 'enrich',
          label: 'Pathway Enrichment',
          content: <EnrichmentSubTab runId={run.run_id} summary={s} />,
        },
        {
          id: 'traj',
          label: 'Trajectory',
          content: <TrajectorySubTab runId={run.run_id} summary={s} />,
        },
        {
          id: 'perturbation',
          label: 'Perturbation Prediction',
          content: <PerturbationTab runId={run.run_id} />,
        },
      ]}
    />
    </>
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

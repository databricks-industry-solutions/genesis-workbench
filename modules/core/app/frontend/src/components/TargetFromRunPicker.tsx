// "Pick a target from a prior run" — surfaces candidate target genes from
// completed Genomics (variant annotation) and Single Cell (marker) runs, so a
// scientist carries a target across modules by selecting a run instead of
// retyping a gene. Returns the chosen gene symbol; the caller resolves it to a
// sequence (GeneResolveInput). Reuses existing module endpoints — no new backend.
import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { Dialog } from '@/components/Dialog'
import { cn } from '@/lib/utils'

type Source = 'genomics' | 'single_cell'

export function TargetFromRunPicker({ onPickGene }: { onPickGene: (gene: string) => void }) {
  const [open, setOpen] = useState(false)
  const [source, setSource] = useState<Source>('genomics')
  const [selectedRun, setSelectedRun] = useState<string | null>(null)

  // Run lists (only fetched while the dialog is open).
  const genomicsRuns = useQuery({
    queryKey: ['target_picker', 'genomics_runs'],
    queryFn: () => api.variantAnnotationSearch('experiment_name', 'annotation'),
    enabled: open && source === 'genomics',
  })
  const scRuns = useQuery({
    queryKey: ['target_picker', 'sc_runs'],
    queryFn: api.singleCellRuns,
    enabled: open && source === 'single_cell',
  })

  // Genes for the selected run (deduped; carries an optional significance note).
  const genes = useMutation({
    mutationFn: async (runId: string): Promise<{ gene: string; note?: string }[]> => {
      if (source === 'genomics') {
        const res = await api.variantAnnotationResults(runId)
        const seen = new Map<string, string | undefined>()
        for (const v of res.variants) {
          if (!v.gene) continue
          const prev = seen.get(v.gene)
          // prefer a pathogenic annotation note if present
          const note = v.clinical_significance ?? undefined
          if (!seen.has(v.gene) || (note && /patho/i.test(note) && !(prev && /patho/i.test(prev)))) {
            seen.set(v.gene, note)
          }
        }
        return [...seen.entries()].map(([gene, note]) => ({ gene, note }))
      }
      // Prefer the genes the scientist explicitly marked on this run (the
      // study list, persisted from Single Cell). Fall back to top markers.
      const scRun = (scRuns.data?.runs ?? []).find((r) => r.run_id === runId)
      if (scRun && scRun.marked_genes && scRun.marked_genes.length > 0) {
        return scRun.marked_genes.map((gene) => ({ gene, note: 'marked of interest' }))
      }
      const info = await api.singleCellRunInfo(runId)
      const set = new Set<string>()
      for (const entries of Object.values(info.top_genes_by_cluster)) {
        for (const e of entries.slice(0, 15)) set.add(e.gene)
      }
      return [...set].map((gene) => ({ gene }))
    },
  })

  const pickRun = (runId: string) => {
    setSelectedRun(runId)
    genes.mutate(runId)
  }
  const switchSource = (s: Source) => {
    setSource(s)
    setSelectedRun(null)
    genes.reset()
  }

  const runRows =
    source === 'genomics'
      ? (genomicsRuns.data?.runs ?? []).map((r) => ({ id: r.run_id, label: r.run_name, sub: r.experiment_name, ok: r.status === 'annotation_complete' || r.status === 'complete' }))
      : (scRuns.data?.runs ?? []).map((r) => ({
          id: r.run_id,
          label: r.run_name,
          sub:
            r.marked_genes && r.marked_genes.length > 0
              ? `${r.experiment_name} · ◆ ${r.marked_genes.length} marked`
              : r.experiment_name,
          ok: r.status === 'complete' || r.status === 'finished',
        }))

  const loading = source === 'genomics' ? genomicsRuns.isLoading : scRuns.isLoading

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-accent"
      >
        Pick from a prior run
      </button>

      <Dialog open={open} onClose={() => setOpen(false)} title="Pick a target from a prior run" width="max-w-2xl">
        <div className="mb-3 flex gap-1">
          {([['genomics', 'Genomics (variants)'], ['single_cell', 'Single Cell (markers)']] as [Source, string][]).map(
            ([s, label]) => (
              <button
                key={s}
                type="button"
                onClick={() => switchSource(s)}
                className={cn(
                  'rounded-full border px-3 py-1 text-xs',
                  s === source ? 'border-primary bg-primary/10 text-primary' : 'border-border text-muted-foreground hover:bg-accent',
                )}
              >
                {label}
              </button>
            ),
          )}
        </div>

        <div className="grid grid-cols-2 gap-3">
          {/* Runs */}
          <div>
            <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Completed runs
            </div>
            {loading ? (
              <p className="text-xs text-muted-foreground">Loading…</p>
            ) : runRows.length === 0 ? (
              <p className="text-xs text-muted-foreground">No runs found.</p>
            ) : (
              <div className="max-h-72 space-y-1 overflow-auto">
                {runRows.map((r) => (
                  <button
                    key={r.id}
                    type="button"
                    disabled={!r.ok}
                    onClick={() => pickRun(r.id)}
                    title={r.ok ? '' : 'Run not complete'}
                    className={cn(
                      'block w-full rounded-md border px-2 py-1 text-left text-xs',
                      selectedRun === r.id ? 'border-primary bg-primary/10' : 'border-border hover:bg-accent',
                      !r.ok && 'opacity-40',
                    )}
                  >
                    <div className="truncate font-medium">{r.label || r.id}</div>
                    <div className="truncate text-[10px] text-muted-foreground">{r.sub}</div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Genes from the selected run */}
          <div>
            <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Genes in this run
            </div>
            {!selectedRun ? (
              <p className="text-xs text-muted-foreground">Select a run to see its genes.</p>
            ) : genes.isPending ? (
              <p className="text-xs text-muted-foreground">Loading genes…</p>
            ) : (genes.data?.length ?? 0) === 0 ? (
              <p className="text-xs text-muted-foreground">No genes surfaced by this run.</p>
            ) : (
              <div className="flex max-h-72 flex-wrap content-start gap-1 overflow-auto">
                {genes.data!.map((g) => (
                  <button
                    key={g.gene}
                    type="button"
                    onClick={() => {
                      onPickGene(g.gene)
                      setOpen(false)
                    }}
                    title={g.note ? `${g.gene} — ${g.note}` : g.gene}
                    className={cn(
                      'rounded-md border px-2 py-0.5 text-xs hover:bg-primary/20',
                      g.note && /patho/i.test(g.note)
                        ? 'border-destructive/50 bg-destructive/10 text-destructive'
                        : 'border-primary/40 bg-primary/10 text-primary',
                    )}
                  >
                    {g.gene}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </Dialog>
    </>
  )
}

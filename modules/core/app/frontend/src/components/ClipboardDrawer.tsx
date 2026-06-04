// App-wide "clipboard" companion: a slide-out drawer pinned to the right edge,
// available on every module page. Holds the session-persisted genes of interest
// marked across Single Cell (DE / Enrichment / Trajectory) and offers the
// hand-off action (save onto the open Single Cell run → Large Molecule targets).
// Designed to grow into other "stuff of interest" (structures, sequences, runs).
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'

import { api } from '@/api/client'
import { useGeneClipboard } from '@/stores/geneClipboard'
import { useScRunContext } from '@/stores/scRunContext'
import { cn } from '@/lib/utils'

export function ClipboardDrawer() {
  const [open, setOpen] = useState(false)
  const genes = useGeneClipboard((s) => s.genes)
  const remove = useGeneClipboard((s) => s.remove)
  const clear = useGeneClipboard((s) => s.clear)
  const runId = useScRunContext((s) => s.runId)
  const runLabel = useScRunContext((s) => s.runLabel)
  const mark = useMutation({
    mutationFn: () => api.singleCellMarkGenes({ run_id: runId!, genes }),
  })
  const count = genes.length

  return (
    <>
      {/* Right-edge handle — always visible, shows the count. */}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        title="Clipboard — your genes of interest"
        className="fixed right-0 top-1/3 z-40 flex items-center gap-1.5 rounded-l-md border border-r-0 border-border bg-card px-1.5 py-3 text-xs font-medium shadow-md hover:bg-accent"
        style={{ writingMode: 'vertical-rl' }}
      >
        <span>📋 Clipboard</span>
        {count > 0 && (
          <span className="rounded-full bg-primary px-1.5 py-0.5 text-[10px] font-semibold text-primary-foreground">
            {count}
          </span>
        )}
      </button>

      {open && (
        <div className="fixed inset-0 z-40 bg-black/30" onClick={() => setOpen(false)} />
      )}

      <aside
        className={cn(
          'fixed right-0 top-0 z-50 flex h-full w-[360px] max-w-[85vw] flex-col border-l border-border bg-card shadow-xl transition-transform duration-200',
          open ? 'translate-x-0' : 'translate-x-full',
        )}
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div>
            <h2 className="text-sm font-semibold">📋 Clipboard</h2>
            <p className="text-[10px] text-muted-foreground">
              Genes of interest · kept for this session
            </p>
          </div>
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="rounded-md px-2 py-1 text-muted-foreground hover:bg-accent hover:text-foreground"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs font-medium text-muted-foreground">
              Genes ({count})
            </span>
            {count > 0 && (
              <button
                type="button"
                onClick={clear}
                className="text-[11px] text-muted-foreground hover:text-destructive"
              >
                Clear all
              </button>
            )}
          </div>

          {count === 0 ? (
            <p className="text-xs text-muted-foreground">
              No genes yet. Mark genes of interest in Single Cell — the <strong>+</strong> in the DE
              table, or <strong>+ Study list</strong> in Pathway Enrichment / Trajectory — and they
              collect here, available across modules.
            </p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {genes.map((g) => (
                <button
                  key={g}
                  type="button"
                  onClick={() => remove(g)}
                  title="Remove"
                  className="rounded border border-primary/40 bg-primary/10 px-1.5 py-0.5 text-xs text-primary hover:bg-primary/20"
                >
                  {g} ✕
                </button>
              ))}
            </div>
          )}
        </div>

        {count > 0 && (
          <div className="border-t border-border p-4">
            {runId ? (
              <>
                <button
                  type="button"
                  onClick={() => mark.mutate()}
                  disabled={mark.isPending}
                  className="w-full rounded-md border border-primary/50 bg-primary/10 px-3 py-2 text-xs font-medium text-primary hover:bg-primary/20 disabled:opacity-50"
                >
                  {mark.isPending
                    ? 'Saving…'
                    : mark.isSuccess
                      ? '✓ Saved to run → available in Large Molecule'
                      : 'Save to this run → Large Molecule'}
                </button>
                <p className="mt-1 text-[10px] text-muted-foreground">
                  Saves onto {runLabel ? <strong>{runLabel}</strong> : 'the open run'} (MLflow tag),
                  so the target shows up in Large Molecule’s “Pick a target from a prior run”.
                </p>
              </>
            ) : (
              <p className="text-[10px] text-muted-foreground">
                Open a Single Cell → Analysis run to save these onto it for the Large Molecule
                hand-off.
              </p>
            )}
          </div>
        )}
      </aside>
    </>
  )
}

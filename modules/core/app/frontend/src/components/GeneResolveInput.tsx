// Compact "target from gene" control — an icon button that opens a popover to
// turn a gene symbol into its canonical protein sequence (self-contained
// gene_sequences lookup), sourced by typing, from the Clipboard, or a prior run.
// Hands the resolved sequence to the parent (fills a sequence field).
import { useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'

import { api } from '@/api/client'
import { ClipboardPaste } from '@/components/ClipboardPaste'
import { TargetFromRunPicker } from '@/components/TargetFromRunPicker'
import { useOutsideDismiss } from '@/hooks/useOutsideDismiss'

export function GeneResolveInput({ onResolved }: { onResolved: (sequence: string) => void }) {
  const [open, setOpen] = useState(false)
  const [gene, setGene] = useState('')
  const btnRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)

  const resolve = useMutation({
    mutationFn: (g: string) => api.resolveGene(g),
    onSuccess: (res) => {
      if (res.found && res.sequence) {
        onResolved(res.sequence)
        setOpen(false)
      }
    },
  })
  const data = resolve.data

  const pickGene = (g: string) => {
    setGene(g)
    resolve.mutate(g)
  }

  // Click-away/Esc to close (the run-picker dialog is nested inside this ref, so
  // clicking it doesn't close the popover). Anchor to button + panel (not the
  // wrapper, which can stretch) so an outside click reliably closes it.
  useOutsideDismiss([btnRef, panelRef], () => setOpen(false), open)

  return (
    <div className="relative inline-block text-xs">
      <button
        ref={btnRef}
        type="button"
        onClick={() => setOpen((o) => !o)}
        title="Fill the sequence from a target gene — type it, or pull from Clipboard / a prior run"
        className="rounded-md border border-primary/50 bg-primary/10 px-2.5 py-1 font-medium text-primary hover:bg-primary/20"
      >
        🧬 Target from gene
      </button>

      {open && (
        <div ref={panelRef} className="absolute left-0 z-40 mt-1 w-80 space-y-2 rounded-md border border-border bg-card p-3 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
              Resolve a gene → protein sequence
            </span>
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close"
              className="rounded px-1 text-muted-foreground hover:text-foreground"
            >
              ✕
            </button>
          </div>
          <div className="flex gap-2">
            <input
              value={gene}
              onChange={(e) => setGene(e.target.value)}
              placeholder="e.g. PARP1"
              autoFocus
              onKeyDown={(e) => e.key === 'Enter' && gene.trim() && resolve.mutate(gene.trim())}
              className="min-w-0 flex-1 rounded-md border border-border bg-background px-2 py-1"
            />
            <button
              type="button"
              onClick={() => resolve.mutate(gene.trim())}
              disabled={!gene.trim() || resolve.isPending}
              className="shrink-0 rounded-md border border-primary/50 bg-primary/10 px-2.5 py-1 text-primary hover:bg-primary/20 disabled:opacity-40"
            >
              {resolve.isPending ? 'Resolving…' : 'Resolve'}
            </button>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[11px] text-muted-foreground">or from:</span>
            <ClipboardPaste kind="gene" label="Clipboard" onPick={(it) => pickGene(it.value)} />
            <TargetFromRunPicker onPickGene={pickGene} />
          </div>
          {data && !data.found && (
            <div className="text-[11px] text-destructive">
              No reviewed human protein found for that gene symbol.
            </div>
          )}
          {resolve.isError && (
            <div className="text-[11px] text-destructive">{String(resolve.error)}</div>
          )}
        </div>
      )}
    </div>
  )
}

// "Resolve from gene symbol" — turns a target gene (e.g. surfaced by Single
// Cell / Genomics) into its canonical protein sequence via the self-contained
// gene_sequences lookup, and hands it to the parent (fills a sequence field).
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'

import { api } from '@/api/client'
import { ClipboardPaste } from '@/components/ClipboardPaste'
import { TargetFromRunPicker } from '@/components/TargetFromRunPicker'

export function GeneResolveInput({ onResolved }: { onResolved: (sequence: string) => void }) {
  const [gene, setGene] = useState('')
  const resolve = useMutation({
    mutationFn: (g: string) => api.resolveGene(g),
    onSuccess: (res) => {
      if (res.found && res.sequence) onResolved(res.sequence)
    },
  })
  const data = resolve.data

  const pickGene = (g: string) => {
    setGene(g)
    resolve.mutate(g)
  }

  return (
    <div className="rounded-md border border-border bg-muted/20 p-2">
      <div className="flex items-center gap-2">
        <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
          Resolve from gene
        </span>
        <input
          value={gene}
          onChange={(e) => setGene(e.target.value)}
          placeholder="e.g. PARP1"
          className="min-w-0 flex-1 rounded-md border border-border bg-background px-2 py-1 text-xs"
          onKeyDown={(e) => e.key === 'Enter' && gene.trim() && resolve.mutate(gene.trim())}
        />
        <button
          type="button"
          onClick={() => resolve.mutate(gene.trim())}
          disabled={!gene.trim() || resolve.isPending}
          className="shrink-0 rounded-md border border-primary/50 bg-primary/10 px-2.5 py-1 text-xs text-primary hover:bg-primary/20 disabled:opacity-40"
        >
          {resolve.isPending ? 'Resolving…' : 'Resolve'}
        </button>
        <ClipboardPaste kind="gene" label="Paste gene" onPick={(it) => pickGene(it.value)} />
        <TargetFromRunPicker onPickGene={pickGene} />
      </div>
      {data?.found && (
        <div className="mt-1 text-[11px] text-success">
          ✓ {data.gene} — {data.protein_name} ({data.accession}, {data.length} aa). Sequence filled in.
        </div>
      )}
      {data && !data.found && (
        <div className="mt-1 text-[11px] text-destructive">
          No reviewed human protein found for that gene symbol.
        </div>
      )}
      {resolve.isError && (
        <div className="mt-1 text-[11px] text-destructive">{String(resolve.error)}</div>
      )}
    </div>
  )
}

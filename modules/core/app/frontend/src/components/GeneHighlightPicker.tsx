// Lets the user pick a gene set of interest to highlight in DE results —
// either by pasting symbols or by choosing a pathway/term from the GO/KEGG/
// Reactome libraries already on the volume. Domain-agnostic: oncology is just
// one optional example preset, not a built-in assumption.
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { CANCER_GENES } from '@/lib/cancerGenes'
import type { GenesetTerm } from '@/types/api'

export type Highlight = { genes: Set<string>; label: string }

export function GeneHighlightPicker({
  highlight,
  onChange,
}: {
  highlight: Highlight | null
  onChange: (h: Highlight | null) => void
}) {
  const [mode, setMode] = useState<'paste' | 'library'>('paste')
  const [text, setText] = useState('')
  const [db, setDb] = useState('')
  const [q, setQ] = useState('')

  const dbs = useQuery({
    queryKey: ['geneset_dbs'],
    queryFn: api.singleCellGenesetDbs,
    enabled: mode === 'library',
  })
  const terms = useQuery({
    queryKey: ['geneset_terms', db, q],
    queryFn: () => api.singleCellGenesetTerms(db, q),
    enabled: mode === 'library' && !!db,
  })

  const applyPaste = () => {
    const genes = new Set(
      text
        .split(/[\s,;]+/)
        .map((s) => s.trim().toUpperCase())
        .filter(Boolean),
    )
    if (genes.size) onChange({ genes, label: `Custom (${genes.size} genes)` })
  }
  const applyTerm = (t: GenesetTerm) =>
    onChange({ genes: new Set(t.genes.map((g) => g.toUpperCase())), label: t.term })
  const applyCancer = () =>
    onChange({ genes: new Set(CANCER_GENES), label: 'Cancer (COSMIC + HGSOC) example' })

  return (
    <div className="rounded-md border border-border bg-muted/20 p-3 text-xs">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <span className="font-medium">Highlight a gene set of interest (optional)</span>
        {highlight && (
          <span className="flex items-center gap-1 rounded bg-yellow-400/20 px-2 py-0.5 text-yellow-700 dark:text-yellow-400">
            ◆ {highlight.label} ({highlight.genes.size})
            <button
              type="button"
              onClick={() => onChange(null)}
              title="Clear highlight"
              className="ml-1 font-bold hover:opacity-70"
            >
              ✕
            </button>
          </span>
        )}
      </div>

      <div className="mb-2 flex flex-wrap gap-1">
        {(['paste', 'library'] as const).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={
              'rounded-full border px-2.5 py-0.5 ' +
              (m === mode
                ? 'border-primary bg-primary/10 text-primary'
                : 'border-border text-muted-foreground hover:bg-accent')
            }
          >
            {m === 'paste' ? 'Paste genes' : 'From pathway library'}
          </button>
        ))}
        <button
          type="button"
          onClick={applyCancer}
          title="Load the curated cancer example set (COSMIC drivers + HGSOC markers)"
          className="ml-auto rounded-full border border-border px-2.5 py-0.5 text-muted-foreground hover:bg-accent"
        >
          Load cancer example
        </button>
      </div>

      {mode === 'paste' ? (
        <div className="flex gap-2">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={2}
            placeholder="Paste gene symbols (comma, space, or newline separated) — e.g. PARP1, BRCA1, RAD51"
            className="min-w-0 flex-1 rounded-md border border-border bg-background px-2 py-1"
          />
          <button
            type="button"
            onClick={applyPaste}
            disabled={!text.trim()}
            className="shrink-0 self-start rounded-md border border-primary/50 bg-primary/10 px-2.5 py-1 text-primary disabled:opacity-40"
          >
            Apply
          </button>
        </div>
      ) : (
        <div>
          <div className="mb-2 flex flex-wrap gap-2">
            <select
              value={db}
              onChange={(e) => setDb(e.target.value)}
              className="rounded-md border border-border bg-background px-2 py-1"
            >
              <option value="">{dbs.isLoading ? 'Loading libraries…' : 'Select library…'}</option>
              {(dbs.data?.dbs ?? []).map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search terms… e.g. DNA repair, cell cycle"
              disabled={!db}
              className="min-w-0 flex-1 rounded-md border border-border bg-background px-2 py-1 disabled:opacity-40"
            />
          </div>
          {db && (
            <div className="max-h-40 overflow-auto rounded-md border border-border">
              {terms.isLoading ? (
                <p className="p-2 text-muted-foreground">Loading terms…</p>
              ) : (terms.data?.terms.length ?? 0) === 0 ? (
                <p className="p-2 text-muted-foreground">No matching terms.</p>
              ) : (
                terms.data!.terms.map((t) => (
                  <button
                    key={t.term}
                    type="button"
                    onClick={() => applyTerm(t)}
                    title={t.term}
                    className="block w-full truncate px-2 py-1 text-left hover:bg-accent"
                  >
                    {t.term} <span className="text-muted-foreground">({t.size})</span>
                  </button>
                ))
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

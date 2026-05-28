import { useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { Dialog } from '@/components/Dialog'
import { MolstarViewer } from '@/components/MolstarViewer'
import { RealtimeProgress } from '@/components/RealtimeProgress'
import { WorkflowProgress } from '@/components/WorkflowProgress'
import { useSseMutation } from '@/hooks/useSseMutation'
import type { SequenceHit, SequenceSearchResponse } from '@/types/api'
import { cn } from '@/lib/utils'

const EXAMPLE_SEQUENCE =
  'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'

const TOP_K_OPTIONS = [25, 50, 100, 200, 500]
const VALID_AA = new Set('ACDEFGHIKLMNPQRSTVWY')

function parseFasta(content: string): string {
  // Take the first record's sequence (ignore header line).
  const lines = content.split(/\r?\n/)
  const out: string[] = []
  let started = false
  for (const raw of lines) {
    const line = raw.trim()
    if (!line) continue
    if (line.startsWith('>')) {
      if (started) break
      started = true
      continue
    }
    out.push(line)
  }
  return out.join('')
}

export function SequenceSearchTab() {
  const [inputMode, setInputMode] = useState<'paste' | 'fasta'>('paste')
  const [sequence, setSequence] = useState(EXAMPLE_SEQUENCE)
  const [fastaName, setFastaName] = useState<string | null>(null)
  const [topK, setTopK] = useState(50)
  const [viewing, setViewing] = useState<SequenceHit | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const search = useSseMutation<
    { sequence: string; top_k: number },
    SequenceSearchResponse
  >('/api/protein_studies/sequence_search/stream')

  const runSearch = () => search.start({ sequence, top_k: topK })

  const cleanSeq = useMemo(
    () => sequence.replace(/\s/g, '').toUpperCase(),
    [sequence],
  )
  const invalidChars = useMemo(() => {
    if (!cleanSeq) return ''
    return Array.from(new Set([...cleanSeq].filter((c) => !VALID_AA.has(c)))).join(', ')
  }, [cleanSeq])

  const handleFile = async (file: File) => {
    const text = await file.text()
    const parsed = parseFasta(text)
    if (!parsed) {
      alert('Could not parse FASTA — no sequence found after the > header line.')
      return
    }
    setFastaName(file.name)
    setSequence(parsed)
  }

  const columns = useMemo<ColumnDef<SequenceHit, unknown>[]>(
    () => [
      { id: 'seq_id', header: 'Seq ID', accessorKey: 'seq_id' },
      {
        id: 'description',
        header: 'Description',
        cell: (ctx) => {
          const d = ctx.row.original.description
          return d.length > 80 ? d.slice(0, 80) + '…' : d
        },
      },
      {
        id: 'identity_pct',
        header: 'Identity %',
        accessorFn: (r) => r.identity_pct.toFixed(1),
      },
      { id: 'sw_score', header: 'SW Score', accessorKey: 'sw_score' },
      { id: 'alignment_length', header: 'Aln Len', accessorKey: 'alignment_length' },
      { id: 'seq_length', header: 'Seq Len', accessorKey: 'seq_length' },
      {
        id: 'vector_distance',
        header: 'Vec Dist',
        accessorFn: (r) => r.vector_distance.toFixed(4),
      },
      {
        id: 'view',
        header: '',
        cell: (ctx) => (
          <button
            onClick={() => setViewing(ctx.row.original)}
            className="rounded-md border border-primary bg-primary/10 px-3 py-1 text-xs text-primary hover:bg-primary/20"
          >
            View
          </button>
        ),
      },
    ],
    [],
  )

  const canSearch = cleanSeq.length > 0 && !search.isPending

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <h3 className="text-sm font-semibold">Sequence Similarity Search</h3>
        <span className="text-xs text-muted-foreground">
          BLAST-like search: ESM-2 embeddings + Vector Search + Smith-Waterman alignment
        </span>
      </div>

      <div className="flex flex-col gap-3 md:flex-row md:items-stretch">
        <div className="flex-1 space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-muted-foreground">Input:</span>
            {(['paste', 'fasta'] as const).map((m) => (
              <button
                key={m}
                onClick={() => setInputMode(m)}
                className={cn(
                  'rounded-full border px-3 py-1 text-xs transition-colors',
                  m === inputMode
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border text-muted-foreground hover:bg-accent',
                )}
              >
                {m === 'paste' ? 'Paste Sequence' : 'Upload FASTA'}
              </button>
            ))}
          </div>

          {inputMode === 'paste' ? (
            <textarea
              rows={5}
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
              placeholder="Amino acid sequence (single-letter code)"
              className="w-full rounded-md border border-border bg-background p-3 font-mono text-xs"
            />
          ) : (
            <div className="rounded-md border border-border bg-muted/30 p-3 text-xs">
              <input
                ref={fileInputRef}
                type="file"
                accept=".fasta,.fa,.faa"
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                className="text-sm text-muted-foreground"
              />
              {fastaName && (
                <div className="mt-2">
                  <div className="text-muted-foreground">Loaded {fastaName}</div>
                  <pre className="mt-1 overflow-x-auto whitespace-pre-wrap font-mono text-[10px]">
                    {cleanSeq.slice(0, 200)}
                    {cleanSeq.length > 200 ? '…' : ''}
                  </pre>
                </div>
              )}
            </div>
          )}

          {invalidChars && (
            <div className="text-xs text-amber-500">
              Non-standard residues will be tolerated by the aligner: {invalidChars}
            </div>
          )}
        </div>

        <div className="flex flex-col items-end gap-2">
          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Max results
            </span>
            <select
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="rounded-md border border-border bg-background px-3 py-2 text-sm"
            >
              {TOP_K_OPTIONS.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <button
            onClick={runSearch}
            disabled={!canSearch}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
          >
            {search.isPending ? 'Searching…' : 'Search'}
          </button>
        </div>
      </div>

      {search.isPending && (
        <RealtimeProgress
          title="Hybrid funnel search"
          pct={search.progress?.pct ?? 0}
          msg={search.progress?.msg ?? 'Starting…'}
          stages={[
            { label: 'Embedding query (ESM-2)', pctEnd: 35 },
            { label: 'Vector Search ANN over UniRef90', pctEnd: 50 },
            { label: 'Fetching candidate sequences', pctEnd: 60 },
            { label: 'Smith-Waterman alignment', pctEnd: 96 },
            { label: 'Ranking results', pctEnd: 100 },
          ]}
        />
      )}

      {search.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(search.error)}
        </div>
      )}

      {search.data && (
        <section className="space-y-3 border-t border-border pt-4">
          <div className="flex items-baseline justify-between">
            <h4 className="text-sm font-medium">
              Results — {search.data.hits.length} hits
            </h4>
            <span className="text-xs text-muted-foreground">
              Sorted by Smith-Waterman score
            </span>
          </div>
          {search.data.hits.length === 0 ? (
            <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
              No matches.
            </div>
          ) : (
            <DataTable columns={columns} data={search.data.hits} />
          )}
        </section>
      )}

      <Dialog
        open={!!viewing}
        onClose={() => setViewing(null)}
        title={viewing?.seq_id ?? ''}
        width="max-w-5xl"
      >
        {viewing && <HitDetail hit={viewing} />}
      </Dialog>
    </div>
  )
}

function HitDetail({ hit }: { hit: SequenceHit }) {
  const targetSeq = useMemo(() => hit.aligned_target.replace(/-/g, ''), [hit])

  const organism = useQuery({
    queryKey: ['seq-search', 'organism', hit.seq_id],
    queryFn: () => api.sequenceOrganism(hit.description),
    enabled: Boolean(hit.description),
  })

  const structure = useQuery({
    queryKey: ['seq-search', 'esmfold', hit.seq_id, targetSeq],
    queryFn: () => api.esmfold(targetSeq),
    enabled: targetSeq.length > 0,
    staleTime: 60_000,
  })

  return (
    <div className="space-y-4">
      <div className="text-xs text-muted-foreground">{hit.description}</div>
      <div className="text-sm">
        <span className="text-muted-foreground">Suggested organism: </span>
        <span className="font-medium">
          {organism.isLoading ? 'identifying…' : organism.data?.organism ?? 'Unknown'}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <Metric label="Identity" value={`${hit.identity_pct}%`} />
        <Metric label="SW Score" value={String(hit.sw_score)} />
        <Metric label="Alignment Length" value={String(hit.alignment_length)} />
        <Metric label="Seq Length" value={String(hit.seq_length)} />
      </div>

      <details open className="rounded-md border border-border bg-muted/20">
        <summary className="cursor-pointer px-4 py-2 text-sm font-medium">
          Sequence alignment
        </summary>
        <pre className="overflow-x-auto whitespace-pre border-t border-border px-4 py-3 font-mono text-xs">
          {chunkAlignment(hit.aligned_query, hit.aligned_comp, hit.aligned_target)}
        </pre>
      </details>

      <div>
        <h4 className="mb-2 text-sm font-medium">Predicted structure (ESMFold)</h4>
        <WorkflowProgress
          active={structure.isLoading}
          title="ESMFold of target sequence"
          stages={[{ label: 'Predicting structure', estSeconds: 12 }]}
        />
        {structure.error ? (
          <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
            Structure prediction unavailable: {String(structure.error)}
          </div>
        ) : structure.data ? (
          <MolstarViewer viewerHtml={structure.data.viewer_html} height={460} />
        ) : null}
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-muted/30 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-base font-medium">{value}</div>
    </div>
  )
}

function chunkAlignment(query: string, comp: string, target: string, chunk = 60): string {
  const lines: string[] = []
  for (let i = 0; i < query.length; i += chunk) {
    lines.push(`Query:  ${query.slice(i, i + chunk)}`)
    lines.push(`        ${comp.slice(i, i + chunk)}`)
    lines.push(`Target: ${target.slice(i, i + chunk)}`)
    lines.push('')
  }
  return lines.join('\n')
}

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
  const [minCoverage, setMinCoverage] = useState(30)
  const [viewing, setViewing] = useState<SequenceHit | null>(null)
  const [showHelp, setShowHelp] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const search = useSseMutation<
    { sequence: string; top_k: number; min_coverage_pct: number },
    SequenceSearchResponse
  >('/api/large_molecule/sequence_search/stream')

  const runSearch = () =>
    search.start({ sequence, top_k: topK, min_coverage_pct: minCoverage })

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
      {
        id: 'similarity_score',
        header: 'Similarity',
        accessorFn: (r) => r.similarity_score.toFixed(1),
        meta: { thClass: 'min-w-[80px]' },
      },
      {
        id: 'identity_pct',
        header: 'Identity %',
        accessorFn: (r) => r.identity_pct.toFixed(1),
      },
      {
        id: 'query_coverage_pct',
        header: 'Coverage %',
        accessorFn: (r) => r.query_coverage_pct.toFixed(0),
      },
      { id: 'seq_id', header: 'Seq ID', accessorKey: 'seq_id' },
      {
        id: 'description',
        header: 'Description',
        cell: (ctx) => {
          const d = ctx.row.original.description
          return d.length > 80 ? d.slice(0, 80) + '…' : d
        },
        meta: { thClass: 'min-w-[280px]', tdClass: 'whitespace-normal' },
      },
      { id: 'seq_length', header: 'Seq Len', accessorKey: 'seq_length' },
      { id: 'alignment_length', header: 'Aln Len', accessorKey: 'alignment_length' },
      { id: 'sw_score', header: 'SW Score', accessorKey: 'sw_score' },
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
      <div>
        <h3 className="text-sm font-semibold">Find Similar Protein Sequences</h3>
        <p className="mt-1 text-xs text-muted-foreground">
          BLAST-like search: ESM-2 embeddings + Vector Search + Smith-Waterman alignment
        </p>
      </div>

      <div className="flex flex-col gap-3 md:flex-row md:items-stretch">
        <div className="space-y-2 md:w-[55%] md:max-w-2xl">
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
          <label
            className="block text-xs"
            title="Drops candidates whose Smith-Waterman alignment covers less than this fraction of the query — keeps small fragment hits out of the top-K. Set to 0 to disable."
          >
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Min coverage %
            </span>
            <input
              type="number"
              min={0}
              max={100}
              step={5}
              value={minCoverage}
              onChange={(e) => setMinCoverage(parseInt(e.target.value || '0'))}
              className="w-24 rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
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
            <h4 className="flex items-center gap-2 text-sm font-medium">
              Results — {search.data.hits.length} hits
              <button
                type="button"
                onClick={() => setShowHelp(true)}
                className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-border text-[11px] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                aria-label="About the search and the result columns"
                title="What do these columns mean?"
              >
                i
              </button>
            </h4>
            <span className="text-xs text-muted-foreground">
              Sorted by similarity (identity × coverage)
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

      <Dialog
        open={showHelp}
        onClose={() => setShowHelp(false)}
        title="About sequence search"
        width="max-w-2xl"
      >
        <SequenceSearchHelp />
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
        <Metric label="Similarity" value={hit.similarity_score.toFixed(1)} />
        <Metric label="Identity" value={`${hit.identity_pct}%`} />
        <Metric label="Query coverage" value={`${hit.query_coverage_pct.toFixed(0)}%`} />
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

function SequenceSearchHelp() {
  const columns: { label: string; desc: string }[] = [
    {
      label: 'Similarity',
      desc: 'Composite score = Identity % × (Coverage / 100). 0–100, higher is better. Primary sort key.',
    },
    {
      label: 'Identity %',
      desc: 'Fraction of aligned residues that match exactly, over the aligned region only.',
    },
    {
      label: 'Coverage %',
      desc: 'Fraction of the QUERY sequence that was aligned (gaps excluded). Penalises hits that match only a tiny window of the query.',
    },
    { label: 'Seq ID', desc: 'Database identifier for the matched sequence.' },
    { label: 'Description', desc: 'FASTA header / annotation for the matched sequence in the reference.' },
    {
      label: 'Seq Len',
      desc: 'Total length of the matched target sequence (not just the aligned region).',
    },
    {
      label: 'Aln Len',
      desc: 'Length of the Smith-Waterman local alignment (including gap characters).',
    },
    {
      label: 'SW Score',
      desc: 'Raw Smith-Waterman score (BLOSUM62, gap_open=11, gap_extend=1). Tiebreaker for ranking.',
    },
    {
      label: 'Vec Dist',
      desc: 'Cosine-ish distance between query and target ESM-2 embeddings — what the vector-search stage ranked by before alignment.',
    },
  ]

  return (
    <div className="space-y-4 text-sm">
      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Search technique — Hybrid funnel
        </h3>
        <ol className="list-decimal space-y-1 pl-5 text-xs">
          <li>
            <strong>ESM-2 embed</strong> the query sequence to a fixed-length vector.
          </li>
          <li>
            <strong>Vector Search ANN</strong> over the UniRef-derived{' '}
            <code>sequence_embedding_index</code> for the top-K nearest candidates (K = 10×
            your Max-results).
          </li>
          <li>
            <strong>Fetch</strong> the candidate sequences from the Delta reference.
          </li>
          <li>
            <strong>Smith-Waterman</strong> local alignment via parasail (BLOSUM62) of the
            query against each candidate.
          </li>
          <li>
            <strong>Filter + rank</strong>: drop hits below the Min-coverage threshold, then
            sort by Similarity descending (SW Score breaks ties).
          </li>
        </ol>
      </section>

      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Sort order
        </h3>
        <p className="text-xs">
          Results are sorted by <strong>Similarity</strong> (Identity % × Coverage / 100)
          descending — short fragment hits with perfect identity but tiny coverage rank
          below longer hits that span most of the query. SW Score is the tiebreaker.
        </p>
      </section>

      <section>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Column definitions
        </h3>
        <dl className="grid grid-cols-1 gap-x-4 gap-y-2 md:grid-cols-[120px,1fr]">
          {columns.map((c) => (
            <div key={c.label} className="contents">
              <dt className="font-medium text-foreground">{c.label}</dt>
              <dd className="text-xs text-muted-foreground">{c.desc}</dd>
            </div>
          ))}
        </dl>
      </section>
    </div>
  )
}

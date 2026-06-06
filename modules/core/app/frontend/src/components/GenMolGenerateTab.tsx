import { useMemo, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { ClipboardPaste } from '@/components/ClipboardPaste'
import { DataTable } from '@/components/DataTable'
import { MaterialIcon } from '@/components/MaterialIcon'
import { RealtimeProgress } from '@/components/RealtimeProgress'
import { useSseMutation } from '@/hooks/useSseMutation'
import { useClipboard } from '@/stores/clipboard'
import type { GenMolGenerateResponse, GenMolMolecule, SeedMotif } from '@/types/api'

type Mode = 'denovo' | 'fragment'

export function GenMolGenerateTab() {
  const [mode, setMode] = useState<Mode>('denovo')
  const [fragments, setFragments] = useState('c1ccc(cc1)C(=O)N')
  const [numMolecules, setNumMolecules] = useState(20)
  const [temperature, setTemperature] = useState(1.0)
  const [randomness, setRandomness] = useState(1.0)
  const [scoring, setScoring] = useState<'qed' | 'logp'>('qed')
  const [unique, setUnique] = useState(true)

  const clipAdd = useClipboard((s) => s.add)
  const clipItems = useClipboard((s) => s.items)
  const onClip = useMemo(
    () => new Set(clipItems.filter((i) => i.kind === 'molecule').map((i) => i.value)),
    [clipItems],
  )

  // Seed from an identified target: gene (or a protein sequence to reverse-
  // resolve) → known-binder Murcko scaffolds (motifs) → seed fragment mode.
  const [gene, setGene] = useState('')
  const motifs = useMutation({
    mutationFn: (p: { gene?: string; sequence?: string }) => api.genmolSeedMotifs(p),
  })

  const useMotif = (m: SeedMotif) => {
    setMode('fragment')
    setFragments((prev) => {
      const lines = prev.split('\n').map((s) => s.trim()).filter(Boolean)
      return lines.includes(m.scaffold) ? prev : [...lines, m.scaffold].join('\n')
    })
  }

  const gen = useSseMutation<
    {
      seeds: string[]
      num_molecules: number
      temperature: number
      randomness: number
      scoring: string
      unique: boolean
    },
    GenMolGenerateResponse
  >('/api/small_molecule/genmol/generate/stream')

  const fragmentSeeds = useMemo(
    () => fragments.split('\n').map((s) => s.trim()).filter(Boolean),
    [fragments],
  )

  const run = () =>
    gen.start({
      seeds: mode === 'denovo' ? [''] : fragmentSeeds,
      num_molecules: numMolecules,
      temperature,
      randomness,
      scoring,
      unique,
    })

  const canRun = !gen.isPending && (mode === 'denovo' || fragmentSeeds.length > 0)
  const molecules = gen.data?.molecules ?? []

  const columns: ColumnDef<GenMolMolecule, unknown>[] = [
    {
      id: 'idx',
      header: '#',
      cell: ({ row }) => <span className="text-muted-foreground">{row.index + 1}</span>,
    },
    { id: 'smiles', header: 'SMILES', accessorKey: 'smiles',
      cell: ({ row }) => <span className="font-mono text-xs">{row.original.smiles}</span> },
    {
      id: 'score',
      header: scoring === 'qed' ? 'QED' : 'LogP',
      accessorKey: 'score',
      cell: ({ row }) =>
        row.original.score == null ? '—' : row.original.score.toFixed(3),
    },
    {
      id: 'seed',
      header: 'Source',
      accessorKey: 'seed',
      cell: ({ row }) => <span className="text-xs text-muted-foreground">{row.original.seed}</span>,
    },
    {
      id: 'clip',
      header: '',
      cell: ({ row }) => {
        const on = onClip.has(row.original.smiles)
        return (
          <button
            type="button"
            onClick={() =>
              clipAdd({ kind: 'molecule', value: row.original.smiles, source: 'GenMol' })
            }
            title="Copy this molecule to the Clipboard (→ Docking / ADMET)"
            className="inline-flex items-center gap-1 rounded-md border border-primary/50 bg-primary/10 px-2 py-0.5 text-xs text-primary hover:bg-primary/20"
          >
            <MaterialIcon name="assignment" className="text-[14px] text-cyan-400" />
            {on ? '✓' : 'Clip'}
          </button>
        )
      },
    },
  ]

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Generate Novel Small Molecules</h3>
        <p className="text-xs text-muted-foreground">
          GenMol (NVIDIA) invents drug-like molecules — <strong>de novo</strong>, or by growing a{' '}
          <strong>seed fragment</strong> into analogs that keep its motif. Generated candidates flow
          to <strong>Molecular Docking</strong> (binding vs a target) and <strong>ADMET &amp; Safety</strong>{' '}
          via the Clipboard. Research use — evaluate before any downstream use.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(320px,420px)_1fr]">
        {/* Left: form */}
        <div className="space-y-3">
          {/* Seed from an identified target protein → its binding motif. */}
          <div className="rounded-md border border-border bg-card p-3 text-xs">
            <div className="mb-1.5 font-medium uppercase tracking-wide text-muted-foreground">
              Find binding motif from target
            </div>
            <div className="flex items-center gap-2">
              <input
                value={gene}
                onChange={(e) => setGene(e.target.value)}
                placeholder="Target gene, e.g. PARP1"
                className="min-w-0 flex-1 rounded-md border border-border bg-background px-2 py-1.5 text-sm"
              />
              <button
                type="button"
                onClick={() => motifs.mutate({ gene })}
                disabled={!gene.trim() || motifs.isPending}
                className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
              >
                {motifs.isPending ? 'Finding…' : 'Find'}
              </button>
            </div>
            <div className="mt-1.5">
              <ClipboardPaste
                kind="sequence"
                label="From Clipboard sequence"
                onPick={(it) => motifs.mutate({ sequence: it.value })}
              />
            </div>

            {motifs.data && (
              <div className="mt-2 space-y-1">
                {motifs.data.motifs.length === 0 ? (
                  <p className="text-[11px] text-muted-foreground">
                    No binding motifs found{motifs.data.gene ? ` for ${motifs.data.gene}` : ''}. The
                    target_binders table may still be building, or the target has no known binders.
                  </p>
                ) : (
                  <>
                    <p className="text-[11px] text-muted-foreground">
                      {motifs.data.gene} — click a motif to seed fragment generation:
                    </p>
                    {motifs.data.motifs.map((m) => (
                      <button
                        key={m.scaffold}
                        type="button"
                        onClick={() => useMotif(m)}
                        title={`Seed GenMol with this scaffold (${m.count} known binders)`}
                        className="block w-full truncate rounded border border-border bg-background px-2 py-1 text-left font-mono text-[11px] hover:bg-accent"
                      >
                        <span className="text-primary">◆</span> {m.scaffold}
                        <span className="ml-1 font-sans text-[10px] text-muted-foreground">
                          · {m.count} binders
                          {m.best_pchembl != null ? ` · pChEMBL ${m.best_pchembl.toFixed(1)}` : ''}
                        </span>
                      </button>
                    ))}
                  </>
                )}
              </div>
            )}
            {motifs.error && (
              <p className="mt-1 text-[11px] text-destructive">{String(motifs.error)}</p>
            )}
          </div>

          <div className="flex gap-2">
            {(['denovo', 'fragment'] as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={
                  'rounded-full border px-3 py-1 text-xs transition-colors ' +
                  (m === mode
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border text-muted-foreground hover:bg-accent')
                }
              >
                {m === 'denovo' ? 'De novo' : 'From fragment(s)'}
              </button>
            ))}
          </div>

          {mode === 'fragment' && (
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Seed fragment SMILES — one per line
              </span>
              <textarea
                rows={5}
                value={fragments}
                onChange={(e) => setFragments(e.target.value)}
                placeholder="c1ccc(cc1)C(=O)N"
                className="w-full rounded-md border border-border bg-background p-3 font-mono text-xs"
              />
            </label>
          )}

          <div className="grid grid-cols-2 gap-3">
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Molecules per seed
              </span>
              <input
                type="number"
                min={1}
                max={200}
                value={numMolecules}
                onChange={(e) => setNumMolecules(parseInt(e.target.value) || 1)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Rank by
              </span>
              <select
                value={scoring}
                onChange={(e) => setScoring(e.target.value as 'qed' | 'logp')}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              >
                <option value="qed">QED (drug-likeness)</option>
                <option value="logp">LogP</option>
              </select>
            </label>
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Temperature
              </span>
              <input
                type="number"
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value) || 1.0)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Randomness
              </span>
              <input
                type="number"
                step={0.1}
                value={randomness}
                onChange={(e) => setRandomness(parseFloat(e.target.value) || 1.0)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
          </div>

          <label className="flex items-center gap-2 text-xs">
            <input type="checkbox" checked={unique} onChange={(e) => setUnique(e.target.checked)} />
            Unique molecules only
          </label>

          <div className="flex gap-2">
            <button
              onClick={run}
              disabled={!canRun}
              className="flex-1 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
            >
              {gen.isPending ? 'Generating…' : 'Generate'}
            </button>
            <button
              onClick={() => gen.reset()}
              disabled={!gen.data && !gen.error}
              className="rounded-md border border-border px-4 py-2 text-sm hover:bg-accent disabled:opacity-50"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Right: progress + results */}
        <div className="space-y-3">
          {gen.isPending && (
            <RealtimeProgress
              title="GenMol generation"
              pct={gen.progress?.pct ?? 0}
              msg={gen.progress?.msg ?? 'Starting…'}
              stages={[
                { label: 'Submitting to GenMol', pctEnd: 20 },
                { label: 'Generating (endpoint cold-start ~30-60s)', pctEnd: 95 },
                { label: 'Ranking', pctEnd: 100 },
              ]}
            />
          )}

          {gen.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
              {String(gen.error)}
            </div>
          )}

          {molecules.length > 0 ? (
            <>
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium">
                  {molecules.length} molecule{molecules.length === 1 ? '' : 's'} (ranked by{' '}
                  {scoring === 'qed' ? 'QED ↑' : 'LogP'})
                </h4>
                <button
                  type="button"
                  onClick={() =>
                    molecules.forEach((m) =>
                      clipAdd({ kind: 'molecule', value: m.smiles, source: 'GenMol' }),
                    )
                  }
                  className="inline-flex items-center gap-1 rounded-md border border-primary/50 bg-primary/10 px-2.5 py-1 text-xs text-primary hover:bg-primary/20"
                >
                  <MaterialIcon name="assignment" className="text-[15px] text-cyan-400" />
                  Clip all
                </button>
              </div>
              <DataTable columns={columns} data={molecules} emptyText="No molecules" />
            </>
          ) : (
            !gen.isPending &&
            !gen.error && (
              <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
                Set options and hit Generate. Generated molecules can be sent to Docking and ADMET
                via the Clipboard.
              </div>
            )
          )}
        </div>
      </div>
    </div>
  )
}

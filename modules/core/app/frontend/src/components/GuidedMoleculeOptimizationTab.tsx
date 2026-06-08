import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { ClipboardPaste } from '@/components/ClipboardPaste'
import { ClipToggle } from '@/components/ClipToggle'
import { DataTable } from '@/components/DataTable'
import { DispatchSuccess } from '@/components/DispatchSuccess'
import { RunSearchSection } from '@/components/RunSearchSection'
import { SequenceSourceControls } from '@/components/SequenceSourceControls'
import type { DBRunRow, MolOptStatus, MolOptTopKItem, SeedMotif } from '@/types/api'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

const fmt = (v: number | null | undefined, d = 3) =>
  v == null || Number.isNaN(v) ? '—' : v.toFixed(d)

export function GuidedMoleculeOptimizationTab() {
  const [seeds, setSeeds] = useState('')
  const [numIterations, setNumIterations] = useState(25)
  const [numSamples, setNumSamples] = useState(24)
  const [selectTop, setSelectTop] = useState(3)
  const [dockTopK, setDockTopK] = useState(5)
  const [wQed, setWQed] = useState(1.0)
  const [wAdmet, setWAdmet] = useState(1.0)
  const [wDock, setWDock] = useState(1.0)
  const [targetSequence, setTargetSequence] = useState('')
  const [dockPerIter, setDockPerIter] = useState(8)
  const [experiment, setExperiment] = useState('gwb_molecule_optimization')
  const [runName, setRunName] = useState(`mol_opt_${ts()}`)

  // Find binding motif from target → seed scaffold(s).
  const [gene, setGene] = useState('')
  const motifs = useMutation({
    mutationFn: (p: { gene?: string; sequence?: string }) => api.genmolSeedMotifs(p),
  })
  const addMotif = (m: SeedMotif) =>
    setSeeds((prev) => {
      const lines = prev.split('\n').map((s) => s.trim()).filter(Boolean)
      return lines.includes(m.scaffold) ? prev : [...lines, m.scaffold].join('\n')
    })

  const seedList = seeds.split('\n').map((s) => s.trim()).filter(Boolean)

  const start = useMutation({
    mutationFn: () =>
      api.molOptStart({
        seed_smiles: seedList,
        num_samples: numSamples,
        num_iterations: numIterations,
        select_top: selectTop,
        dock_top_k: dockTopK,
        weights: { qed: wQed, admet: wAdmet, dock: targetSequence.trim() ? wDock : 0 },
        temperature: 1.2,
        randomness: 2.0,
        target_sequence: targetSequence.trim(),
        dock_per_iter: dockPerIter,
        mlflow_experiment: experiment,
        mlflow_run_name: runName,
      }),
  })

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Guided Molecule Design</h3>
        <p className="text-xs text-muted-foreground">
          A design-make-test loop: GenMol grows the seed motif into candidates → each scored on{' '}
          <strong>QED</strong> + <strong>ADMET</strong> (and <strong>DiffDock binding</strong> when a
          target structure is given) → the best reseed the next round. Runs as a batch job — track
          and open results under <strong>Search past runs</strong>.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(340px,440px)_1fr]">
        {/* Left: form */}
        <div className="space-y-3">
          {/* Find binding motif from an identified target → seed scaffold(s). */}
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
            <div className="mt-1.5 flex flex-wrap gap-2">
              <ClipboardPaste
                kind="gene"
                label="Clipboard gene"
                onPick={(it) => {
                  setGene(it.value)
                  motifs.mutate({ gene: it.value })
                }}
              />
              <ClipboardPaste
                kind="sequence"
                label="Clipboard sequence"
                onPick={(it) => motifs.mutate({ sequence: it.value })}
              />
            </div>
            {motifs.data && (
              <div className="mt-2 space-y-1">
                {motifs.data.motifs.length === 0 ? (
                  <p className="text-[11px] text-muted-foreground">
                    No binding motifs found{motifs.data.gene ? ` for ${motifs.data.gene}` : ''}.
                  </p>
                ) : (
                  <>
                    <p className="text-[11px] text-muted-foreground">
                      {motifs.data.gene} — click to add a seed scaffold:
                    </p>
                    {motifs.data.motifs.map((m) => (
                      <button
                        key={m.scaffold}
                        type="button"
                        onClick={() => addMotif(m)}
                        title={`Add as seed (${m.count} known binders)`}
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
          </div>

          <label className="block text-xs">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="block uppercase tracking-wide text-muted-foreground">
                Seed scaffold SMILES — one per line
              </span>
              <ClipboardPaste
                kind="molecule"
                label="Paste molecule"
                onPick={(it) =>
                  setSeeds((prev) => {
                    const lines = prev.split('\n').map((s) => s.trim()).filter(Boolean)
                    return lines.includes(it.value) ? prev : [...lines, it.value].join('\n')
                  })
                }
              />
            </div>
            <textarea
              rows={4}
              value={seeds}
              onChange={(e) => setSeeds(e.target.value)}
              placeholder="O=C(c1cccc(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCNCC1"
              className="w-full rounded-md border border-border bg-background p-3 font-mono text-xs"
            />
          </label>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Iterations</span>
              <input type="number" min={1} max={50} value={numIterations}
                onChange={(e) => setNumIterations(parseInt(e.target.value) || 1)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Candidates / iter</span>
              <input type="number" min={4} max={100} value={numSamples}
                onChange={(e) => setNumSamples(parseInt(e.target.value) || 4)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Parents kept</span>
              <input type="number" min={1} max={10} value={selectTop}
                onChange={(e) => setSelectTop(parseInt(e.target.value) || 1)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Top-K kept</span>
              <input type="number" min={1} max={20} value={dockTopK}
                onChange={(e) => setDockTopK(parseInt(e.target.value) || 1)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">QED weight</span>
              <input type="number" step={0.1} value={wQed}
                onChange={(e) => setWQed(parseFloat(e.target.value) || 0)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">ADMET weight</span>
              <input type="number" step={0.1} value={wAdmet}
                onChange={(e) => setWAdmet(parseFloat(e.target.value) || 0)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
            </label>
          </div>

          {/* Optional: dock candidates into the reward (binding drives the loop).
              Target comes in as a sequence (gene-resolve / paste), folded by the loop. */}
          <div className="rounded-md border border-border bg-card p-3 text-xs">
            <div className="mb-1.5 font-medium uppercase tracking-wide text-muted-foreground">
              Dock into reward (optional)
            </div>
            <span className="mb-1.5 block text-[11px] text-muted-foreground">
              Resolve a target gene or paste a sequence — it's folded (ESMFold) and DiffDock binding
              joins the reward. Leave empty for a QED+ADMET-only loop.
            </span>
            <SequenceSourceControls onSequence={setTargetSequence} className="mb-1.5" />
            <textarea
              rows={3}
              value={targetSequence}
              onChange={(e) => setTargetSequence(e.target.value)}
              placeholder="Target protein sequence (single-letter), or resolve from a gene above…"
              className="w-full rounded-md border border-border bg-background p-2 font-mono text-[11px]"
            />
            {targetSequence.trim() && (
              <div className="mt-1 text-[10px] text-muted-foreground">
                Target: {targetSequence.replace(/\s+/g, '').length} aa
              </div>
            )}
            <div className="mt-2 grid grid-cols-2 gap-3">
              <label className="block">
                <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Dock weight</span>
                <input type="number" step={0.1} value={wDock}
                  onChange={(e) => setWDock(parseFloat(e.target.value) || 0)}
                  disabled={!targetSequence.trim()}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm disabled:opacity-50" />
              </label>
              <label className="block">
                <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Dock / iter</span>
                <input type="number" min={1} max={20} value={dockPerIter}
                  onChange={(e) => setDockPerIter(parseInt(e.target.value) || 1)}
                  disabled={!targetSequence.trim()}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm disabled:opacity-50" />
              </label>
            </div>
          </div>

          {/* MLflow tracking — same as other workflow screens. */}
          <div className="grid grid-cols-2 gap-3 text-xs">
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">MLflow Experiment</span>
              <input value={experiment} onChange={(e) => setExperiment(e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Run name</span>
              <input value={runName} onChange={(e) => setRunName(e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
            </label>
          </div>

          <button
            onClick={() => start.mutate()}
            disabled={start.isPending || seedList.length === 0 || !experiment.trim() || !runName.trim()}
            className="w-full rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
          >
            {start.isPending ? 'Starting…' : 'Start optimization'}
          </button>
          {start.data && (
            <DispatchSuccess jobRunId={start.data.job_run_id} runUrl={start.data.run_url} />
          )}
          {start.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive">
              {String(start.error)}
            </div>
          )}
        </div>

        {/* Right: Search past runs (standard workflow pattern — RunSearchSection
            renders its own "Search Past Runs" header + in-progress badge). */}
        <div>
          <RunSearchSection
            searchKey={['molopt', 'search'] as const}
            searchFn={api.molOptSearch}
            detailLabel="Iterations"
            initialText="mol_opt"
            viewableStatuses={['complete']}
            detailColClass="min-w-[80px]"
            renderDialog={(run) => <MolOptResultBody run={run} />}
          />
        </div>
      </div>
    </div>
  )
}

// View dialog: the run's reward trajectory + optimized top-K (with Clip hand-off).
function MolOptResultBody({ run }: { run: DBRunRow }) {
  const status = useQuery<MolOptStatus>({
    queryKey: ['molopt', 'status', run.run_id],
    queryFn: () => api.molOptStatus(run.run_id),
  })
  const topk = useQuery({
    queryKey: ['molopt', 'topk', run.run_id],
    queryFn: () => api.molOptTopK(run.run_id),
  })

  const traj = status.data?.best_reward_history ?? []
  const meanTraj = status.data?.mean_reward_history ?? []
  const maxReward = Math.max(0.001, ...traj.map((p) => p.value))
  const molecules = topk.data?.top_k ?? []
  // Only show the Dock column when this run actually docked (target provided);
  // otherwise it's an all-"—" column. New runs dock the full shortlist.
  const hasDock = molecules.some((m) => m.dock_confidence != null)

  const columns: ColumnDef<MolOptTopKItem, unknown>[] = [
    { id: 'idx', header: '#', cell: ({ row }) => row.index + 1 },
    { id: 'smiles', header: 'SMILES', accessorKey: 'smiles',
      cell: ({ row }) => <span className="font-mono text-xs">{row.original.smiles}</span> },
    { id: 'reward', header: 'Reward', cell: ({ row }) => fmt(row.original.reward) },
    { id: 'qed', header: 'QED', cell: ({ row }) => fmt(row.original.qed) },
    { id: 'tox', header: 'ClinTox', cell: ({ row }) => fmt(row.original.tox) },
    ...(hasDock
      ? [{ id: 'dock', header: 'Dock', cell: ({ row }) => fmt(row.original.dock_confidence) } as ColumnDef<MolOptTopKItem, unknown>]
      : []),
    {
      id: 'clip', header: '',
      cell: ({ row }) => (
        <ClipToggle
          kind="molecule"
          value={row.original.smiles}
          source="Guided Molecule Design"
          addTitle="Add molecule to clipboard"
          removeTitle="On clipboard — click to remove"
        />
      ),
    },
  ]

  return (
    <div className="space-y-4 text-sm">
      {status.data && (
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Reward trajectory</div>
          {traj.length === 0 ? (
            <p className="text-xs text-muted-foreground">No iterations logged.</p>
          ) : (
            <div className="space-y-1">
              {traj.map((p, i) => (
                <div key={p.step} className="flex items-center gap-2">
                  <span className="w-12 text-[10px] text-muted-foreground">iter {p.step}</span>
                  <div className="h-2 flex-1 rounded bg-muted">
                    <div className="h-2 rounded bg-primary"
                      style={{ width: `${Math.min(100, (p.value / maxReward) * 100)}%` }} />
                  </div>
                  <span className="w-24 text-right text-[10px]">
                    best {fmt(p.value)} {meanTraj[i] ? `· mean ${fmt(meanTraj[i].value)}` : ''}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      <div>
        <div className="mb-1 text-xs font-medium text-muted-foreground">
          Optimized top-K {molecules.length ? `(${molecules.length})` : ''}
        </div>
        {topk.isLoading ? (
          <p className="text-xs text-muted-foreground">Loading…</p>
        ) : molecules.length === 0 ? (
          <p className="text-xs text-muted-foreground">No molecules in this run.</p>
        ) : (
          <DataTable columns={columns} data={molecules} emptyText="No molecules" />
        )}
      </div>
    </div>
  )
}

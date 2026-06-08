import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { ClipboardPaste } from '@/components/ClipboardPaste'
import { ClipToggle } from '@/components/ClipToggle'
import { DataTable } from '@/components/DataTable'
import { DispatchSuccess } from '@/components/DispatchSuccess'
import { PlotlyChart as Plot } from '@/components/PlotlyChart'
import { RunSearchSection } from '@/components/RunSearchSection'
import { SequenceSourceControls } from '@/components/SequenceSourceControls'
import type { DBRunRow, MolOptStatus, MolOptTopKItem, SeedMotif } from '@/types/api'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

const fmt = (v: number | null | undefined, d = 3) =>
  v == null || Number.isNaN(v) ? '—' : v.toFixed(d)

// ClinTox probabilities are often tiny (e.g. 6e-5); toFixed(3) would show 0.000.
// Use scientific notation below 0.001 so a near-zero value reads as such, not 0.
const fmtTox = (v: number | null | undefined) => {
  if (v == null || Number.isNaN(v)) return '—'
  if (v === 0) return '0'
  if (v < 0.001) return v.toExponential(1)
  return v.toFixed(3)
}

export function GuidedMoleculeOptimizationTab() {
  const [seeds, setSeeds] = useState('')
  const [numIterations, setNumIterations] = useState(25)
  const [numSamples, setNumSamples] = useState(24)
  const [selectTop, setSelectTop] = useState(3)
  const [dockTopK, setDockTopK] = useState(5)
  // Hard-constraint targets: keep molecules with QED >= qedMin and ClinTox <= toxMax.
  const [qedMin, setQedMin] = useState(0.5)
  const [toxMax, setToxMax] = useState(0.3)
  const [targetSequence, setTargetSequence] = useState('')
  const [dockPerIter, setDockPerIter] = useState(8)
  const [experiment, setExperiment] = useState('gwb_molecule_optimization')
  const [runName, setRunName] = useState(`mol_opt_${ts()}`)
  // Bumped after a successful dispatch so Search Past Runs auto-loads the new run.
  const [searchToken, setSearchToken] = useState(0)

  // Find binding motif from target → seed scaffold(s).
  const [gene, setGene] = useState('')
  const motifs = useMutation({
    mutationFn: (p: { gene?: string; sequence?: string }) => api.genmolSeedMotifs(p),
  })
  // Finding motifs by gene also resolves that gene to its protein sequence and
  // pre-fills the docking target — most runs dock against the same protein the
  // seeds come from. Non-clobbering: only fills when the dock box is empty, so a
  // deliberate choice (a different target, or just the binding domain) is kept.
  const resolveTarget = useMutation({
    mutationFn: api.resolveGene,
    onSuccess: (data) => {
      if (data.found && data.sequence) {
        setTargetSequence((prev) => (prev.trim() ? prev : data.sequence!))
      }
    },
  })
  const findMotifsByGene = (g: string) => {
    if (!g.trim()) return
    motifs.mutate({ gene: g })
    resolveTarget.mutate(g)
  }
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
        qed_min: qedMin,
        tox_max: toxMax,
        temperature: 1.2,
        randomness: 2.0,
        target_sequence: targetSequence.trim(),
        target_label: gene.trim(),
        dock_per_iter: dockPerIter,
        mlflow_experiment: experiment,
        mlflow_run_name: runName,
      }),
    onSuccess: () => setSearchToken((t) => t + 1),
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
                onClick={() => findMotifsByGene(gene)}
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
                  findMotifsByGene(it.value)
                }}
              />
              <ClipboardPaste
                kind="sequence"
                label="Clipboard sequence"
                onPick={(it) => {
                  motifs.mutate({ sequence: it.value })
                  // We already have the sequence — pre-fill the dock target too.
                  setTargetSequence((prev) => (prev.trim() ? prev : it.value))
                }}
              />
            </div>
            {motifs.data && (
              <div className="mt-2 space-y-1">
                {motifs.data.motifs.length === 0 ? (
                  motifs.data.gene ? (
                    <p className="text-[11px] text-muted-foreground">
                      No catalogued small-molecule binders found for{' '}
                      <strong>{motifs.data.gene}</strong> — it may not be an established drug target.
                    </p>
                  ) : (
                    <p className="text-[11px] text-muted-foreground">
                      This sequence isn't a recognized drug target with catalogued binders. Enter a
                      target <strong>gene</strong> (e.g. PARP1), or a <strong>known</strong> target's
                      exact sequence — designed/de-novo proteins (e.g. from Protein Binder Design)
                      won't match.
                    </p>
                  )
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
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Min QED</span>
              <input type="number" step={0.05} min={0} max={1} value={qedMin}
                onChange={(e) => setQedMin(parseFloat(e.target.value) || 0)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Max ClinTox</span>
              <input type="number" step={0.05} min={0} max={1} value={toxMax}
                onChange={(e) => setToxMax(parseFloat(e.target.value) || 0)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
            </label>
          </div>
          <p className="text-[10px] text-muted-foreground">
            Hard constraints: only molecules with <strong>QED ≥ {qedMin.toFixed(2)}</strong> and{' '}
            <strong>ClinTox ≤ {toxMax.toFixed(2)}</strong> are kept; survivors are then optimized
            (docking, when a target is set, ranks the survivors).
          </p>

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
            {targetSequence.trim() && (() => {
              const aa = targetSequence.replace(/\s+/g, '').length
              return (
                <div className="mt-1 space-y-1">
                  <div className="text-[10px] text-muted-foreground">Target: {aa} aa</div>
                  {aa > 1000 && (
                    <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-[10px] text-amber-300">
                      ⚠ Long target ({aa} aa). Folding (ESMFold) may run out of GPU memory; if it
                      does, docking is skipped and the run falls back to a QED+ADMET-only loop.
                      Consider docking against just the binding domain (e.g. the catalytic domain)
                      rather than the full-length protein.
                    </div>
                  )}
                </div>
              )
            })()}
            <div className="mt-2 grid grid-cols-2 gap-3">
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
            searchToken={searchToken}
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
  const valid = topk.data?.top_k ?? []
  const explored = topk.data?.explored ?? []
  const qedMin = status.data?.qed_min
  const toxMax = status.data?.tox_max
  const targetTxt = qedMin != null && toxMax != null ? ` (Drug-likeness ≥ ${qedMin}, Toxicity ≤ ${toxMax})` : ''

  // Shared columns; the explored table adds a "Meets targets" ✓/✗ so you can see
  // why each attempt didn't make the valid list. Dock column only when docked.
  const makeColumns = (
    rows: MolOptTopKItem[],
    withFeasible: boolean,
  ): ColumnDef<MolOptTopKItem, unknown>[] => {
    const hasDock = rows.some((m) => m.dock_confidence != null)
    return [
      { id: 'idx', header: '#', cell: ({ row }) => row.index + 1 },
      { id: 'smiles', header: 'SMILES', accessorKey: 'smiles',
        cell: ({ row }) => <span className="font-mono text-xs">{row.original.smiles}</span> },
      { id: 'reward', header: 'Reward', cell: ({ row }) => fmt(row.original.reward) },
      { id: 'qed', header: 'Drug-likeness (QED)', cell: ({ row }) => fmt(row.original.qed) },
      { id: 'tox', header: 'Toxicity', cell: ({ row }) => fmtTox(row.original.tox) },
      ...(withFeasible
        ? [{ id: 'meets', header: 'Meets targets',
            cell: ({ row }: { row: { original: MolOptTopKItem } }) =>
              row.original.feasible
                ? <span className="text-primary">✓</span>
                : <span className="text-muted-foreground">✗</span> } as ColumnDef<MolOptTopKItem, unknown>]
        : []),
      ...(hasDock
        ? [{ id: 'dock', header: 'Dock', cell: ({ row }: { row: { original: MolOptTopKItem } }) => fmt(row.original.dock_confidence) } as ColumnDef<MolOptTopKItem, unknown>]
        : []),
      { id: 'clip', header: '',
        cell: ({ row }) => (
          <ClipToggle kind="molecule" value={row.original.smiles} source="Guided Molecule Design"
            addTitle="Add molecule to clipboard" removeTitle="On clipboard — click to remove" />
        ) },
    ]
  }

  return (
    <div className="space-y-4 text-sm">
      {status.data && (
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Reward trajectory</div>
          {traj.length === 0 ? (
            <p className="text-xs text-muted-foreground">No iterations logged.</p>
          ) : (
            <div className="rounded-md border border-border bg-card p-2">
              <Plot
                data={[
                  {
                    x: traj.map((p) => p.step),
                    y: traj.map((p) => p.value),
                    mode: 'lines+markers',
                    name: 'best',
                    line: { color: '#86efac' },
                  } as never,
                  {
                    x: meanTraj.map((p) => p.step),
                    y: meanTraj.map((p) => p.value),
                    mode: 'lines+markers',
                    name: 'mean',
                    line: { color: '#60a5fa' },
                  } as never,
                ]}
                layout={{
                  title: { text: 'Reward by iteration' },
                  height: 240,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  font: { size: 11 },
                  xaxis: { title: { text: 'iteration' }, gridcolor: '#333' },
                  yaxis: { title: { text: 'reward' }, gridcolor: '#333' },
                  legend: { font: { size: 10 } },
                  margin: { l: 50, r: 20, t: 40, b: 40 },
                }}
                config={{ displaylogo: false, responsive: true }}
                style={{ width: '100%' }}
                useResizeHandler
              />
            </div>
          )}
        </div>
      )}
      {topk.isLoading ? (
        <p className="text-xs text-muted-foreground">Loading…</p>
      ) : valid.length > 0 ? (
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">
            Valid candidates ({valid.length}){targetTxt}
          </div>
          <DataTable columns={makeColumns(valid, false)} data={valid} emptyText="No molecules" />
        </div>
      ) : (
        <div className="rounded-md border border-amber-500/50 bg-amber-500/10 p-4 text-center">
          <p className="text-lg font-bold text-amber-300">No candidates could be found</p>
          <p className="mt-1 text-xs text-muted-foreground">
            No molecule met the targets{targetTxt} in this run. Loosen the thresholds or run more
            iterations — the closest attempts are shown below.
          </p>
        </div>
      )}
      {explored.length > 0 && (
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">
            Other molecules explored ({explored.length})
          </div>
          <DataTable columns={makeColumns(explored, true)} data={explored} emptyText="No molecules" />
        </div>
      )}
    </div>
  )
}

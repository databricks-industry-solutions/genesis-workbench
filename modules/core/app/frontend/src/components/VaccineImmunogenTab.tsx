import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { DispatchSuccess } from '@/components/DispatchSuccess'
import { MolstarViewer } from '@/components/MolstarViewer'
import { RunSearchSection } from '@/components/RunSearchSection'
import type { DBRunRow } from '@/types/api'
import { cn } from '@/lib/utils'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

function parseResidues(csv: string): number[] {
  return csv
    .split(',')
    .map((t) => parseInt(t.trim(), 10))
    .filter((n) => Number.isFinite(n))
}

// Per-axis reward weights shown as sliders (same control as Antibody Design). The
// top four are the immunogen core; the last two (scaffold self-reactivity) are
// opt-in and default to 0 — a vaccine is MEANT to be immunogenic.
const WEIGHT_AXES: { key: string; label: string; help: string }[] = [
  { key: 'motif_rmsd', label: 'Epitope presentation — motif RMSD', help: 'Backbone RMSD between the input epitope and the folded design’s motif region (self-consistency). Lower = the scaffold presents the epitope in its native conformation. The headline axis.' },
  { key: 'plddt', label: 'Scaffold fold confidence — ESMFold pLDDT', help: 'Mean pLDDT of the designed scaffold structure.' },
  { key: 'solubility', label: 'Solubility — NetSolP', help: 'Predicted solubility of the designed sequence (expressibility / manufacturability).' },
  { key: 'thermostab', label: 'Thermostability — DeepSTABp Tm', help: 'Predicted melting temperature (stability / manufacturability).' },
  { key: 'liability', label: 'Low sequence liabilities', help: 'Rule-based scan for chemical/developability liability motifs (deamidation, isomerization, N-glyc sequons, free Cys, Met/Trp oxidation). Fewer = more manufacturable (lower raw count is better).' },
  { key: 'immuno', label: 'Scaffold self-reactivity — MHC-I (optional)', help: 'MHCflurry MHC-I presentation burden of the whole sequence (lower is better). OFF by default — opt in only to trim T-cell epitopes in the carrier; a vaccine is meant to be immunogenic.' },
  { key: 'immuno_mhc2', label: 'Scaffold self-reactivity — MHC-II (optional)', help: 'HLAIIPred MHC class II presentation burden across a DRB1 panel (lower is better). OFF by default — opt in only to trim T-helper epitopes in the carrier.' },
]

export function VaccineImmunogenTab() {
  const defaults = useQuery({
    queryKey: ['vaccine_immunogen', 'defaults'],
    queryFn: api.vaccineImmunogenDefaults,
    staleTime: Infinity,
  })

  // ─── Form state ────────────────────────────────────────────────────────
  const [motifPdb, setMotifPdb] = useState('')
  const [motifResidues, setMotifResidues] = useState('')
  const [motifChain, setMotifChain] = useState('A')
  const [lenMin, setLenMin] = useState(80)
  const [lenMax, setLenMax] = useState(120)
  const [numSamples, setNumSamples] = useState(8)
  const [numIterations, setNumIterations] = useState(6)
  const [runMpnn, setRunMpnn] = useState(true)
  const [experiment, setExperiment] = useState('gwb_vaccine_immunogen')
  const [runName, setRunName] = useState(`immunogen_${ts()}`)
  const [weights, setWeights] = useState<Record<string, number>>({})
  const [strategy, setStrategy] = useState<'resample' | 'noop'>('resample')
  const [resampleTemp, setResampleTemp] = useState(0.1)
  const [convEnabled, setConvEnabled] = useState(true)
  const [convThreshold, setConvThreshold] = useState(0.01)
  const [convWindow, setConvWindow] = useState(2)
  const [targetEnabled, setTargetEnabled] = useState(false)
  const [targetReward, setTargetReward] = useState(0.9)
  const [bestkEnabled, setBestkEnabled] = useState(false)
  const [bestkTarget, setBestkTarget] = useState(10)
  const [bestkThreshold, setBestkThreshold] = useState(0.8)
  const [searchToken, setSearchToken] = useState(0)

  // Prefill the demo epitope + residues + default axis weights from the server.
  useEffect(() => {
    if (!defaults.data) return
    setMotifPdb((v) => v || defaults.data!.motif_pdb)
    setMotifChain((v) => v || defaults.data!.motif_chain)
    setMotifResidues((v) => v || defaults.data!.motif_residues.join(','))
    setWeights((cur) => (Object.keys(cur).length ? cur : { ...defaults.data!.default_weights }))
  }, [defaults.data])

  const start = useMutation({
    mutationFn: api.vaccineImmunogenStart,
    onSuccess: () => setSearchToken((t) => t + 1),
  })

  const runStart = () => {
    start.mutate({
      motif_pdb: motifPdb,
      motif_residues: parseResidues(motifResidues),
      motif_chain: motifChain,
      scaffold_length_min: lenMin,
      scaffold_length_max: lenMax,
      num_samples: numSamples,
      num_iterations: numIterations,
      weights,
      resampling_temperature: resampleTemp,
      strategy,
      run_proteinmpnn: runMpnn,
      convergence_threshold: convEnabled ? convThreshold : -1,
      convergence_window: convWindow,
      target_reward: targetEnabled ? targetReward : null,
      best_k_target: bestkEnabled ? bestkTarget : null,
      best_k_threshold: bestkEnabled ? bestkThreshold : null,
      mlflow_experiment: experiment,
      mlflow_run_name: runName,
    })
  }

  const canStart =
    !start.isPending &&
    motifPdb.trim().length > 0 &&
    motifChain.trim() &&
    experiment.trim() &&
    runName.trim() &&
    lenMin <= lenMax

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Design Vaccine Immunogens with Guided Search</h3>
        <p className="text-xs text-muted-foreground">
          A vaccine teaches the immune system to recognize a small patch on a pathogen — the{' '}
          <strong>epitope</strong>. That patch alone is usually floppy and hard to make. This workflow
          uses RFD4-Proteina (in-process on an H100) <em>motif scaffolding</em> to design a brand-new,
          stable protein that holds the epitope locked in its native 3D shape ("keep the gem fixed,
          design the ring around it"), then iterates: ProteinMPNN scaffold redesign (fix the epitope) →
          ESMFold → score every candidate on epitope-presentation fidelity (motif RMSD), scaffold fold
          confidence, and manufacturability. Each iteration's composite reward biases the next round.
          Long-running — the orchestrator runs as a Databricks Job; track progress in{' '}
          <strong>Search Past Runs</strong> below.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* LEFT — epitope motif + loop params */}
        <div className="space-y-3">
          <div className="rounded-md border border-border bg-card p-3 text-xs">
            <div className="mb-2 font-medium uppercase tracking-wide text-muted-foreground">
              Epitope motif
            </div>
            <label className="block">
              <span className="mb-1 block text-muted-foreground">Epitope structure (PDB)</span>
              <textarea
                rows={8}
                value={motifPdb}
                onChange={(e) => setMotifPdb(e.target.value)}
                placeholder="Paste the epitope-motif PDB (ATOM records for the motif chain)…"
                className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-[11px]"
              />
            </label>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <label className="block">
                <span className="mb-1 block text-muted-foreground">Motif chain</span>
                <input
                  value={motifChain}
                  onChange={(e) => setMotifChain(e.target.value)}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-muted-foreground">Motif residues (CSV)</span>
                <input
                  value={motifResidues}
                  onChange={(e) => setMotifResidues(e.target.value)}
                  placeholder="254,255,…  (empty = whole chain)"
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </label>
            </div>
          </div>

          <div className="rounded-md border border-border bg-card p-3 text-xs">
            <div className="mb-2 font-medium uppercase tracking-wide text-muted-foreground">
              Design loop
            </div>
            <div className="grid grid-cols-2 gap-2">
              <label className="block">
                <span className="mb-1 block text-muted-foreground">Scaffold length min</span>
                <input
                  type="number" min={20} max={400} value={lenMin}
                  onChange={(e) => setLenMin(parseInt(e.target.value || '80'))}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-muted-foreground">Scaffold length max</span>
                <input
                  type="number" min={20} max={400} value={lenMax}
                  onChange={(e) => setLenMax(parseInt(e.target.value || '120'))}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-muted-foreground">K (candidates / iter)</span>
                <input
                  type="number" min={2} max={32} value={numSamples}
                  onChange={(e) => setNumSamples(parseInt(e.target.value || '2'))}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-muted-foreground">N (iterations)</span>
                <input
                  type="number" min={1} max={30} value={numIterations}
                  onChange={(e) => setNumIterations(parseInt(e.target.value || '1'))}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </label>
            </div>
            <label className="mt-2 flex items-center gap-2">
              <input type="checkbox" checked={runMpnn} onChange={(e) => setRunMpnn(e.target.checked)} />
              <span>ProteinMPNN scaffold redesign (fix the epitope)</span>
            </label>
            <p className="mt-1 text-[11px] text-muted-foreground">
              The total scaffold length includes the grafted epitope; it must exceed the motif length.
            </p>
          </div>

          <div className="rounded-md border border-border bg-card p-3 text-xs">
            <div className="mb-2 font-medium uppercase tracking-wide text-muted-foreground">
              MLflow tracking
            </div>
            <label className="block">
              <span className="mb-1 block text-muted-foreground">Experiment</span>
              <input
                value={experiment}
                onChange={(e) => setExperiment(e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
            <label className="mt-2 block">
              <span className="mb-1 block text-muted-foreground">Run name</span>
              <input
                value={runName}
                onChange={(e) => setRunName(e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
          </div>

          <button
            onClick={runStart}
            disabled={!canStart}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
          >
            {start.isPending ? 'Dispatching…' : 'Design immunogens'}
          </button>

          {start.data && <DispatchSuccess jobRunId={start.data.job_run_id} runUrl={start.data.run_url} />}
          {start.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
              {String(start.error)}
            </div>
          )}
        </div>

        {/* RIGHT — reward weights + advanced */}
        <div className="space-y-3">
          <div className="rounded-md border border-border bg-card p-3 text-xs">
            <div className="mb-2 font-medium uppercase tracking-wide text-muted-foreground">
              Per-axis reward weights
            </div>
            <p className="mb-2 text-[11px] text-muted-foreground">
              Weight 0 disables an axis. Each axis is z-score-then-min-max normalised within the
              iteration's batch before the weighted sum.
            </p>
            {WEIGHT_AXES.map((a) => (
              <label key={a.key} className="mb-2 block" title={a.help}>
                <div className="flex justify-between">
                  <span className="text-foreground">{a.label}</span>
                  <span className="text-muted-foreground">{(weights[a.key] ?? 0).toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={5}
                  step={0.1}
                  value={weights[a.key] ?? 0}
                  onChange={(e) => setWeights({ ...weights, [a.key]: parseFloat(e.target.value) })}
                  className="w-full"
                />
              </label>
            ))}
          </div>

          <details className="rounded-md border border-border bg-card p-3 text-xs">
            <summary className="cursor-pointer font-medium uppercase tracking-wide text-muted-foreground">
              Advanced
            </summary>
            <label className="mt-2 block">
              <span className="mb-1 block text-muted-foreground">Strategy</span>
              <div className="flex gap-1">
                {(['resample', 'noop'] as const).map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setStrategy(s)}
                    className={cn(
                      'rounded-md border px-3 py-1.5 text-xs transition-colors',
                      strategy === s
                        ? 'border-primary bg-primary/10 text-primary'
                        : 'border-border text-muted-foreground hover:bg-accent',
                    )}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </label>
            <label className="mt-2 block">
              <div className="flex justify-between text-muted-foreground">
                <span>Resampling temperature</span>
                <span>{resampleTemp.toFixed(2)}</span>
              </div>
              <input
                type="range" min={0.01} max={1} step={0.01} value={resampleTemp}
                onChange={(e) => setResampleTemp(parseFloat(e.target.value))}
                className="w-full"
              />
            </label>
          </details>

          <details className="rounded-md border border-border bg-card p-3 text-xs">
            <summary className="cursor-pointer font-medium uppercase tracking-wide text-muted-foreground">
              Stopping criteria
            </summary>
            <p className="mb-2 mt-2 text-[11px] text-muted-foreground">
              N (iterations) is the hard ceiling. The loop exits early when any enabled criterion fires.
            </p>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={convEnabled} onChange={(e) => setConvEnabled(e.target.checked)} />
              <span>Convergence stop</span>
            </label>
            <div className="mt-1 grid grid-cols-2 gap-2 pl-6">
              <label className="block">
                <span className="block text-muted-foreground">Min improvement</span>
                <input
                  type="number" min={0} max={1} step={0.01} value={convThreshold} disabled={!convEnabled}
                  onChange={(e) => setConvThreshold(parseFloat(e.target.value || '0'))}
                  className="w-full rounded-md border border-border bg-background px-2 py-1 text-sm disabled:opacity-50"
                />
              </label>
              <label className="block">
                <span className="block text-muted-foreground">Window (iters)</span>
                <input
                  type="number" min={1} max={10} value={convWindow} disabled={!convEnabled}
                  onChange={(e) => setConvWindow(parseInt(e.target.value || '2'))}
                  className="w-full rounded-md border border-border bg-background px-2 py-1 text-sm disabled:opacity-50"
                />
              </label>
            </div>
            <label className="mt-2 flex items-center gap-2">
              <input type="checkbox" checked={targetEnabled} onChange={(e) => setTargetEnabled(e.target.checked)} />
              <span>Reward-threshold stop</span>
            </label>
            <div className="mt-1 pl-6">
              <input
                type="number" min={0} max={1} step={0.01} value={targetReward} disabled={!targetEnabled}
                onChange={(e) => setTargetReward(parseFloat(e.target.value || '0.9'))}
                className="w-32 rounded-md border border-border bg-background px-2 py-1 text-sm disabled:opacity-50"
              />
            </div>
            <label className="mt-2 flex items-center gap-2">
              <input type="checkbox" checked={bestkEnabled} onChange={(e) => setBestkEnabled(e.target.checked)} />
              <span>Best-K stop</span>
            </label>
            <div className="mt-1 grid grid-cols-2 gap-2 pl-6">
              <label className="block">
                <span className="block text-muted-foreground">Count target</span>
                <input
                  type="number" min={1} max={100} value={bestkTarget} disabled={!bestkEnabled}
                  onChange={(e) => setBestkTarget(parseInt(e.target.value || '10'))}
                  className="w-full rounded-md border border-border bg-background px-2 py-1 text-sm disabled:opacity-50"
                />
              </label>
              <label className="block">
                <span className="block text-muted-foreground">Reward ≥</span>
                <input
                  type="number" min={0} max={1} step={0.01} value={bestkThreshold} disabled={!bestkEnabled}
                  onChange={(e) => setBestkThreshold(parseFloat(e.target.value || '0.8'))}
                  className="w-full rounded-md border border-border bg-background px-2 py-1 text-sm disabled:opacity-50"
                />
              </label>
            </div>
          </details>
        </div>
      </div>

      <div className="border-t border-border pt-3">
        <h3 className="mb-2 text-sm font-semibold">Search past runs</h3>
        <RunSearchSection
          searchKey={['vaccine_immunogen', 'search'] as const}
          searchFn={api.vaccineImmunogenSearch}
          detailLabel="Top reward"
          initialText="immunogen"
          viewableStatuses={['complete']}
          detailColClass="w-40"
          searchToken={searchToken}
          renderDialog={(run: DBRunRow) => <VaccineResultBody runId={run.run_id} />}
        />
      </div>
    </div>
  )
}

// ─── Result dialog body ──────────────────────────────────────────────────────

function fmt(v: unknown, digits = 3): string {
  if (v == null || v === '') return '—'
  const n = typeof v === 'number' ? v : Number(v)
  return Number.isFinite(n) ? n.toFixed(digits) : String(v)
}

const _AXIS_LABELS: [string, string][] = [
  ['composite_reward', 'Composite reward'],
  ['motif_rmsd', 'Epitope RMSD (Å)'],
  ['plddt', 'Scaffold pLDDT'],
  ['solubility', 'Solubility'],
  ['thermostab', 'Tm'],
  ['liability', 'Seq liabilities (count)'],
  ['immuno', 'Scaffold MHC-I burden'],
  ['immuno_mhc2', 'Scaffold MHC-II burden'],
  ['liability_detail', 'Liability sites'],
]

function VaccineResultBody({ runId }: { runId: string }) {
  const status = useQuery({
    queryKey: ['vaccine_immunogen', 'status', runId],
    queryFn: () => api.vaccineImmunogenStatus(runId),
    refetchInterval: (q) => (q.state.data?.job_status === 'complete' ? false : 10000),
  })
  const topK = useQuery({
    queryKey: ['vaccine_immunogen', 'topk', runId],
    queryFn: () => api.vaccineImmunogenTopK(runId),
    enabled: status.data?.job_status === 'complete',
  })

  const candidates = useMemo(() => topK.data?.candidates ?? [], [topK.data])
  const [selectedId, setSelectedId] = useState<string>('')
  const selected = useMemo(
    () => candidates.find((c) => c.candidate_id === selectedId) ?? candidates[0],
    [candidates, selectedId],
  )
  const trajRow = useMemo(() => {
    const rows = status.data?.trajectory ?? []
    return rows.find((r) => String(r.candidate_id) === String(selected?.candidate_id))
  }, [status.data, selected])

  const download = () => {
    if (!selected) return
    const blob = new Blob([selected.pdb], { type: 'chemical/x-pdb' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${selected.candidate_id}.pdb`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-3 text-sm">
      <div>
        <div className="text-xs font-medium text-muted-foreground">Run</div>
        <div>{status.data?.run_name || runId}</div>
        <div className="text-xs text-muted-foreground">Stage: {status.data?.job_status || '…'}</div>
      </div>

      {status.data?.job_status !== 'complete' && (
        <p className="text-xs text-muted-foreground">
          This run is still in progress — top candidates appear here once it reaches{' '}
          <code>complete</code>. (Auto-refreshing.)
        </p>
      )}

      {candidates.length > 0 && (
        <>
          <div className="flex items-center gap-2">
            <select
              value={selected?.candidate_id ?? ''}
              onChange={(e) => setSelectedId(e.target.value)}
              className="min-w-0 flex-1 rounded-md border border-border bg-background px-3 py-2 text-sm"
            >
              {candidates.map((c) => (
                <option key={c.candidate_id} value={c.candidate_id}>
                  {c.candidate_id}
                </option>
              ))}
            </select>
            <button
              onClick={download}
              className="rounded-md border border-border px-3 py-2 text-xs hover:bg-muted"
            >
              Download PDB
            </button>
          </div>

          {trajRow && (
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 rounded-md border border-border bg-card p-3 text-xs sm:grid-cols-3">
              {_AXIS_LABELS.filter(([k]) => k in trajRow).map(([k, label]) => (
                <div key={k}>
                  <div className="text-muted-foreground">{label}</div>
                  <div className="font-medium">{fmt(trajRow[k])}</div>
                </div>
              ))}
            </div>
          )}

          {selected && <MolstarViewer viewerHtml={selected.viewer_html} height={420} />}
        </>
      )}
    </div>
  )
}

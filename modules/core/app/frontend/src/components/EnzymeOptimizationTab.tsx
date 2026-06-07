import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { DispatchSuccess } from '@/components/DispatchSuccess'
import { Dialog } from '@/components/Dialog'
import { InProgressBadge } from '@/components/InProgressBadge'
import { MolstarViewer } from '@/components/MolstarViewer'
import { PlotlyChart as Plot } from '@/components/PlotlyChart'
import type {
  EnzymeCandidate,
  EnzymeRefRow,
  EnzymeRunRow,
  EnzymeStatusResponse,
} from '@/types/api'
import { cn } from '@/lib/utils'

type SearchMode = 'run_name' | 'experiment_name'

const AXIS_LABELS: { key: string; label: string; help: string }[] = [
  { key: 'motif_rmsd', label: 'Motif backbone RMSD', help: 'Lower is better — catalytic-site drift after redesign.' },
  { key: 'plddt', label: 'ESMFold pLDDT', help: 'Higher is better — global fold confidence.' },
  { key: 'boltz', label: 'Boltz substrate confidence', help: 'Only contributes if substrate SMILES is supplied.' },
  { key: 'solubility', label: 'NetSolP solubility', help: 'Higher is better — E. coli solubility prob in [0,1].' },
  { key: 'half_life', label: 'PLTNUM half-life (anchor-relative)', help: 'Higher is better — sigmoid vs your references. Set to 0 to drop.' },
  { key: 'thermostab', label: 'DeepSTABp Tm (°C)', help: 'Higher is better — predicted melting temperature.' },
  { key: 'immuno', label: 'MHCflurry immunogenic burden', help: 'Lower is better — strong-presenter density across the default HLA panel.' },
]

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

function parseResidues(csv: string): number[] {
  return csv
    .split(',')
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => Number.isFinite(n))
}

export function EnzymeOptimizationTab() {
  // Form defaults seeded from the server so the example PDB + weight map
  // + reference enzymes stay in one place (backend).
  const defaults = useQuery({
    queryKey: ['small_molecule', 'enzyme_opt', 'defaults'],
    queryFn: api.enzymeOptDefaults,
    staleTime: Infinity,
  })

  // ─── Form state ────────────────────────────────────────────────────────
  const [motifPdb, setMotifPdb] = useState('')
  const [targetChain, setTargetChain] = useState('B')
  const [motifResiduesCsv, setMotifResiduesCsv] = useState('1,2,3')
  const [genMode, setGenMode] = useState<'Fast' | 'Accurate'>('Fast')
  const [lenMin, setLenMin] = useState(80)
  const [lenMax, setLenMax] = useState(120)
  const [k, setK] = useState(8)
  const [n, setN] = useState(10)
  const [runProteinMpnn, setRunProteinMpnn] = useState(true)
  const [substrate, setSubstrate] = useState('')
  const [experiment, setExperiment] = useState('gwb_enzyme_optimization')
  const [runName, setRunName] = useState(`enzyme_opt_${ts()}`)
  const [weights, setWeights] = useState<Record<string, number>>({})
  const [references, setReferences] = useState<EnzymeRefRow[]>([])
  const [halfLifeMargin, setHalfLifeMargin] = useState(0.05)
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

  useEffect(() => {
    if (!defaults.data) return
    setMotifPdb((cur) => cur || defaults.data!.motif_pdb)
    setWeights((cur) =>
      Object.keys(cur).length ? cur : { ...defaults.data!.default_weights },
    )
    setReferences((cur) => (cur.length ? cur : defaults.data!.default_references))
  }, [defaults.data])

  const start = useMutation({
    mutationFn: api.enzymeOptStart,
  })

  const smokeTest = useMutation({
    mutationFn: api.enzymeOptSmokeTest,
  })

  const runStart = () => {
    let residues: number[]
    try {
      residues = parseResidues(motifResiduesCsv)
    } catch {
      return
    }
    start.mutate({
      motif_pdb: motifPdb,
      motif_residues: residues,
      target_chain: targetChain,
      scaffold_length_min: lenMin,
      scaffold_length_max: lenMax,
      num_samples: k,
      num_iterations: n,
      weights,
      substrate_smiles: substrate,
      references,
      half_life_margin: halfLifeMargin,
      resampling_temperature: resampleTemp,
      strategy,
      run_proteinmpnn: runProteinMpnn,
      convergence_threshold: convEnabled ? convThreshold : -1,
      convergence_window: convWindow,
      target_reward: targetEnabled ? targetReward : null,
      best_k_target: bestkEnabled ? bestkTarget : null,
      best_k_threshold: bestkEnabled ? bestkThreshold : null,
      use_inprocess_ame: genMode === 'Accurate',
      mlflow_experiment: experiment,
      mlflow_run_name: runName,
    })
  }

  const canStart =
    !start.isPending &&
    motifPdb.trim().length > 0 &&
    experiment.trim() &&
    runName.trim() &&
    lenMin <= lenMax

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Iterate Enzyme Designs with Guided Search</h3>
        <p className="text-xs text-muted-foreground">
          Generate scaffolds with Proteina-Complexa-AME, then iterate: ProteinMPNN redesign →
          ESMFold → score every candidate on physical fidelity <em>and</em> developability axes
          (solubility, anchor-relative half-life, thermostability, immunogenic burden). Each
          iteration's composite reward biases the next round's sampling. Long-running — the
          orchestrator runs as a Databricks Job; track progress in <strong>Search Past Runs</strong> below.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* LEFT — motif + loop params */}
        <div className="space-y-3">
          <div className="rounded-md border border-border bg-card p-3 text-xs">
            <div className="mb-2 font-medium uppercase tracking-wide text-muted-foreground">
              Motif input
            </div>
            <label className="block">
              <span className="mb-1 block text-muted-foreground">Motif + Ligand (PDB)</span>
              <textarea
                rows={8}
                value={motifPdb}
                onChange={(e) => setMotifPdb(e.target.value)}
                className="w-full rounded-md border border-border bg-background p-2 font-mono text-[10px] leading-tight"
              />
            </label>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <label className="block">
                <span className="mb-1 block text-muted-foreground">Motif chain</span>
                <input
                  value={targetChain}
                  onChange={(e) => setTargetChain(e.target.value)}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-muted-foreground">Motif residues (CSV)</span>
                <input
                  value={motifResiduesCsv}
                  onChange={(e) => setMotifResiduesCsv(e.target.value)}
                  placeholder="1,2,3"
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </label>
            </div>
          </div>

          <div className="rounded-md border border-border bg-card p-3 text-xs">
            <div className="mb-2 font-medium uppercase tracking-wide text-muted-foreground">
              Loop parameters
            </div>
            <label className="block">
              <span className="mb-1 block text-muted-foreground">Generation mode</span>
              <div className="flex gap-1">
                {(['Fast', 'Accurate'] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setGenMode(m)}
                    className={cn(
                      'rounded-md border px-3 py-2 text-xs transition-colors',
                      genMode === m
                        ? 'border-primary bg-primary/10 text-primary'
                        : 'border-border text-muted-foreground hover:bg-accent',
                    )}
                  >
                    {m}
                  </button>
                ))}
              </div>
              <span className="mt-1 block text-[10px] text-muted-foreground">
                {genMode === 'Fast'
                  ? '~30 min, no GPU. Reward applies between iterations only.'
                  : '~6 h, A10 GPU (~$22). Feynman-Kac steering biases AME diffusion during sampling.'}
              </span>
            </label>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <label className="block">
                <span className="mb-1 block text-muted-foreground">Scaffold length min</span>
                <input
                  type="number"
                  min={20}
                  max={400}
                  value={lenMin}
                  onChange={(e) => setLenMin(parseInt(e.target.value || '20'))}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-muted-foreground">Scaffold length max</span>
                <input
                  type="number"
                  min={20}
                  max={400}
                  value={lenMax}
                  onChange={(e) => setLenMax(parseInt(e.target.value || '400'))}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-muted-foreground">K (candidates / iter)</span>
                <input
                  type="number"
                  min={2}
                  max={32}
                  value={k}
                  onChange={(e) => setK(parseInt(e.target.value || '2'))}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-muted-foreground">N (iterations)</span>
                <input
                  type="number"
                  min={1}
                  max={30}
                  value={n}
                  onChange={(e) => setN(parseInt(e.target.value || '1'))}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </label>
            </div>
            <label className="mt-2 flex items-center gap-2">
              <input
                type="checkbox"
                checked={runProteinMpnn}
                onChange={(e) => setRunProteinMpnn(e.target.checked)}
              />
              <span>Redesign each scaffold with ProteinMPNN</span>
            </label>
            <label className="mt-2 block">
              <span className="mb-1 block text-muted-foreground">
                Substrate SMILES (gates Boltz axis)
              </span>
              <input
                value={substrate}
                onChange={(e) => setSubstrate(e.target.value)}
                placeholder="(optional)"
                className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
              />
            </label>
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

          <div className="flex flex-wrap gap-2">
            <button
              onClick={runStart}
              disabled={!canStart}
              className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
            >
              {start.isPending ? 'Dispatching…' : 'Launch optimization job'}
            </button>
            <button
              onClick={() => smokeTest.mutate()}
              disabled={smokeTest.isPending}
              className="rounded-md border border-border px-4 py-2 text-sm hover:bg-accent disabled:opacity-50"
              title="Round-trip each developability predictor on T4 lysozyme to confirm they're healthy."
            >
              {smokeTest.isPending ? 'Testing…' : 'Test predictors (T4 lysozyme)'}
            </button>
          </div>

          {start.data && (
            <DispatchSuccess jobRunId={start.data.job_run_id} runUrl={start.data.run_url} />
          )}
          {start.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
              {String(start.error)}
            </div>
          )}

          {smokeTest.data && <SmokeTestPanel data={smokeTest.data} />}
          {smokeTest.error && (
            <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-200">
              Smoke test failed: {String(smokeTest.error)}
            </div>
          )}
        </div>

        {/* RIGHT — guidance */}
        <div className="space-y-3">
          <div className="rounded-md border border-border bg-card p-3 text-xs">
            <div className="mb-2 font-medium uppercase tracking-wide text-muted-foreground">
              Per-axis reward weights
            </div>
            <p className="mb-2 text-[11px] text-muted-foreground">
              Weight 0 disables an axis. Each axis is z-score-then-min-max normalised within
              the iteration's batch before weighted sum (except half-life — pre-normalised via
              the anchor sigmoid).
            </p>
            {AXIS_LABELS.map((a) => (
              <label key={a.key} className="mb-2 block" title={a.help}>
                <div className="flex justify-between">
                  <span className="text-foreground">{a.label}</span>
                  <span className="text-muted-foreground">
                    {(weights[a.key] ?? 0).toFixed(1)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={5}
                  step={0.1}
                  value={weights[a.key] ?? 0}
                  onChange={(e) =>
                    setWeights({ ...weights, [a.key]: parseFloat(e.target.value) })
                  }
                  className="w-full"
                />
              </label>
            ))}
          </div>

          <details open className="rounded-md border border-border bg-card p-3 text-xs">
            <summary className="cursor-pointer font-medium uppercase tracking-wide text-muted-foreground">
              Half-life anchor (reference enzymes)
            </summary>
            <ReferencesEditor refs={references} onChange={setReferences} />
            <label className="mt-2 block">
              <div className="flex justify-between text-muted-foreground">
                <span>Anchor margin β</span>
                <span>{halfLifeMargin.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.01}
                max={0.5}
                step={0.01}
                value={halfLifeMargin}
                onChange={(e) => setHalfLifeMargin(parseFloat(e.target.value))}
                className="w-full"
              />
            </label>
          </details>

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
              <span className="mt-1 block text-[10px] text-muted-foreground">
                <code>resample</code> = softmax-weighted parent resampling.{' '}
                <code>noop</code> = verification mode (no re-generation past iter 1).
              </span>
            </label>
            <label className="mt-2 block">
              <div className="flex justify-between text-muted-foreground">
                <span>Resampling temperature</span>
                <span>{resampleTemp.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.01}
                max={1}
                step={0.01}
                value={resampleTemp}
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
              N (iterations) is the hard ceiling. The loop exits early when any of these fire.
            </p>

            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={convEnabled}
                onChange={(e) => setConvEnabled(e.target.checked)}
              />
              <span>Convergence stop</span>
            </label>
            <div className="mt-1 grid grid-cols-2 gap-2 pl-6">
              <label className="block">
                <span className="block text-muted-foreground">Min improvement</span>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.01}
                  value={convThreshold}
                  onChange={(e) => setConvThreshold(parseFloat(e.target.value || '0'))}
                  disabled={!convEnabled}
                  className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm disabled:opacity-50"
                />
              </label>
              <label className="block">
                <span className="block text-muted-foreground">Window</span>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={convWindow}
                  onChange={(e) => setConvWindow(parseInt(e.target.value || '1'))}
                  disabled={!convEnabled}
                  className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm disabled:opacity-50"
                />
              </label>
            </div>

            <label className="mt-2 flex items-center gap-2">
              <input
                type="checkbox"
                checked={targetEnabled}
                onChange={(e) => setTargetEnabled(e.target.checked)}
              />
              <span>Reward-threshold stop</span>
            </label>
            <label className="mt-1 block pl-6">
              <span className="block text-muted-foreground">Target composite reward</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={targetReward}
                onChange={(e) => setTargetReward(parseFloat(e.target.value || '0'))}
                disabled={!targetEnabled}
                className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm disabled:opacity-50"
              />
            </label>

            <label className="mt-2 flex items-center gap-2">
              <input
                type="checkbox"
                checked={bestkEnabled}
                onChange={(e) => setBestkEnabled(e.target.checked)}
              />
              <span>Best-K cap stop</span>
            </label>
            <div className="mt-1 grid grid-cols-2 gap-2 pl-6">
              <label className="block">
                <span className="block text-muted-foreground">K above threshold</span>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={bestkTarget}
                  onChange={(e) => setBestkTarget(parseInt(e.target.value || '1'))}
                  disabled={!bestkEnabled}
                  className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm disabled:opacity-50"
                />
              </label>
              <label className="block">
                <span className="block text-muted-foreground">Threshold</span>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={bestkThreshold}
                  onChange={(e) => setBestkThreshold(parseFloat(e.target.value || '0'))}
                  disabled={!bestkEnabled}
                  className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm disabled:opacity-50"
                />
              </label>
            </div>
          </details>
        </div>
      </div>

      <SearchPastRunsSection />
    </div>
  )
}

// ─── Helper components ─────────────────────────────────────────────────────

function SmokeTestPanel({ data }: { data: { sequence: string; solubility: number | null; half_life: number | null; thermostab: number | null; immuno: number | null } }) {
  const rows: { label: string; val: number | null; fmt?: (v: number) => string }[] = [
    { label: 'NetSolP solubility', val: data.solubility },
    { label: 'PLTNUM half-life', val: data.half_life },
    { label: 'DeepSTABp Tm (°C)', val: data.thermostab, fmt: (v) => v.toFixed(1) },
    { label: 'MHCflurry immunogenic burden', val: data.immuno },
  ]
  return (
    <div className="rounded-md border border-border bg-card p-3 text-xs">
      <div className="mb-2 font-medium uppercase tracking-wide text-muted-foreground">
        Smoke test on T4 lysozyme
      </div>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1">
        {rows.map((r) => (
          <div key={r.label} className="flex justify-between">
            <dt className="text-muted-foreground">{r.label}</dt>
            <dd className="font-mono">
              {r.val == null
                ? <span className="text-destructive">failed</span>
                : (r.fmt ? r.fmt(r.val) : r.val.toFixed(4))}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

function ReferencesEditor({
  refs,
  onChange,
}: {
  refs: EnzymeRefRow[]
  onChange: (rows: EnzymeRefRow[]) => void
}) {
  const update = (i: number, patch: Partial<EnzymeRefRow>) =>
    onChange(refs.map((r, idx) => (idx === i ? { ...r, ...patch } : r)))
  const remove = (i: number) => onChange(refs.filter((_, idx) => idx !== i))
  const addRow = () =>
    onChange([...refs, { sequence: '', half_life_hours: 1, cell_system: 'HEK293' }])

  return (
    <div className="mt-2 space-y-1.5">
      {/* Column header row, so we can drop the per-field labels in each row. */}
      <div className="grid grid-cols-[1fr_70px_90px_24px] gap-2 px-1 text-[10px] uppercase tracking-wide text-muted-foreground">
        <span>Sequence</span>
        <span>Half-life (h)</span>
        <span>Cell system</span>
        <span />
      </div>
      {refs.map((r, i) => (
        <div key={i} className="grid grid-cols-[1fr_70px_90px_24px] items-center gap-2">
          <input
            value={r.sequence}
            onChange={(e) => update(i, { sequence: e.target.value })}
            placeholder="Reference sequence"
            className="w-full rounded-md border border-border bg-background px-2 py-1 font-mono text-[10px]"
          />
          <input
            type="number"
            min={0}
            max={10000}
            step={0.1}
            value={r.half_life_hours}
            onChange={(e) =>
              update(i, { half_life_hours: parseFloat(e.target.value || '0') })
            }
            className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs"
          />
          <input
            value={r.cell_system}
            onChange={(e) => update(i, { cell_system: e.target.value })}
            className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs"
          />
          <button
            type="button"
            onClick={() => remove(i)}
            className="rounded-md border border-border px-1 text-[10px] text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
            aria-label="Remove reference"
            title="Remove reference"
          >
            ✕
          </button>
        </div>
      ))}
      <button
        type="button"
        onClick={addRow}
        className="rounded-md border border-border px-3 py-1 text-xs text-muted-foreground hover:bg-accent"
      >
        + Add reference
      </button>
    </div>
  )
}

// ─── Search Past Runs + result dialog ──────────────────────────────────────

function _isViewable(status: string): boolean {
  if (!status) return false
  if (status === 'complete' || status === 'failed') return true
  return status.startsWith('iter_') && status.endsWith('_complete')
}

function SearchPastRunsSection() {
  const qc = useQueryClient()
  const [mode, setMode] = useState<SearchMode>('run_name')
  const [text, setText] = useState('enzyme_opt')
  const [viewingId, setViewingId] = useState<string | null>(null)

  const search = useQuery({
    queryKey: ['enzyme_opt', 'search', mode, text],
    queryFn: () => api.enzymeOptSearch(mode, text),
    enabled: false,
  })

  const runs = search.data?.runs ?? []

  const tableColumns = useMemo<ColumnDef<EnzymeRunRow, unknown>[]>(
    () => [
      {
        id: 'run_name',
        header: 'Run',
        cell: (ctx) => {
          const r = ctx.row.original
          return r.run_url ? (
            <a
              href={r.run_url}
              target="_blank"
              rel="noreferrer"
              className="text-primary hover:underline"
              title="Open Databricks run page"
            >
              {r.run_name}
            </a>
          ) : (
            r.run_name
          )
        },
      },
      { id: 'experiment_name', header: 'Experiment', accessorKey: 'experiment_name' },
      { id: 'generation_mode', header: 'Mode', accessorKey: 'generation_mode' },
      {
        id: 'iter_max_reward',
        header: 'Max Reward',
        accessorFn: (r) => (r.iter_max_reward == null ? '—' : r.iter_max_reward.toFixed(3)),
      },
      {
        id: 'iterations_completed',
        header: 'Iters',
        accessorFn: (r) => r.iterations_completed ?? '—',
      },
      {
        id: 'start_time',
        header: 'Started',
        cell: (ctx) =>
          ctx.row.original.start_time_ms
            ? new Date(ctx.row.original.start_time_ms).toLocaleString()
            : '',
      },
      { id: 'job_status', header: 'Stage', accessorKey: 'job_status' },
      { id: 'progress', header: 'Progress', accessorKey: 'progress' },
      {
        id: 'view',
        header: '',
        cell: (ctx) => {
          const r = ctx.row.original
          const viewable = _isViewable(r.job_status)
          return (
            <button
              type="button"
              onClick={() => setViewingId(r.run_id)}
              disabled={!viewable}
              title={
                viewable
                  ? 'View trajectory + top-K candidates'
                  : 'View enabled once an iter_<N>_complete (or final complete/failed) tag is set.'
              }
              className={cn(
                'rounded-md border px-3 py-1 text-xs',
                viewable
                  ? 'border-primary bg-primary/10 text-primary hover:bg-primary/20'
                  : 'cursor-not-allowed border-border text-muted-foreground opacity-50',
              )}
            >
              View
            </button>
          )
        },
      },
    ],
    [],
  )

  const inProgressCount = runs.filter(
    (r) => r.job_status && r.job_status !== 'complete' && r.job_status !== 'failed',
  ).length

  return (
    <section className="space-y-3 border-t border-border pt-4">
      <div className="flex items-baseline justify-between">
        <h4 className="text-sm font-medium">Search Past Runs</h4>
        <InProgressBadge count={inProgressCount} />
      </div>
      <div className="flex flex-wrap items-end gap-3">
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Search by
          </span>
          <div className="flex gap-1">
            {(['run_name', 'experiment_name'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={cn(
                  'rounded-md border px-3 py-2 text-sm transition-colors',
                  mode === m
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border text-muted-foreground hover:bg-accent',
                )}
              >
                {m === 'run_name' ? 'Run name' : 'Experiment name'}
              </button>
            ))}
          </div>
        </label>
        <label className="block text-xs">
          <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
            Contains
          </span>
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-64 rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
        </label>
        <button
          type="button"
          onClick={() => qc.fetchQuery({ queryKey: ['enzyme_opt', 'search', mode, text], queryFn: () => api.enzymeOptSearch(mode, text) })}
          className="rounded-md border border-border px-3 py-2 text-sm hover:bg-accent"
        >
          Search
        </button>
      </div>

      {search.isFetching && <div className="text-xs text-muted-foreground">Searching…</div>}
      {search.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {String(search.error)}
        </div>
      )}
      {!search.isFetching && search.data && runs.length === 0 && (
        <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
          No runs match.
        </div>
      )}
      {runs.length > 0 && <DataTable columns={tableColumns} data={runs} />}

      <Dialog
        open={!!viewingId}
        onClose={() => setViewingId(null)}
        title={viewingId ? `Run ${viewingId.slice(0, 12)}…` : ''}
        width="max-w-6xl"
      >
        {viewingId && <RunResultsBody runId={viewingId} />}
      </Dialog>
    </section>
  )
}

function RunResultsBody({ runId }: { runId: string }) {
  const status = useQuery({
    queryKey: ['enzyme_opt', 'status', runId],
    queryFn: () => api.enzymeOptStatus(runId),
    refetchInterval: 15_000,
  })
  const topK = useQuery({
    queryKey: ['enzyme_opt', 'topk', runId],
    queryFn: () => api.enzymeOptTopK(runId),
    staleTime: 60_000,
  })

  const [selectedCand, setSelectedCand] = useState<string | null>(null)
  useEffect(() => {
    if (topK.data?.candidates.length && !selectedCand) {
      setSelectedCand(topK.data.candidates[0].candidate_id)
    }
  }, [topK.data, selectedCand])

  const selected: EnzymeCandidate | null = useMemo(() => {
    if (!topK.data || !selectedCand) return null
    return topK.data.candidates.find((c) => c.candidate_id === selectedCand) ?? null
  }, [topK.data, selectedCand])

  // Pull the trajectory row for the selected candidate so we can render
  // its scoring metrics above the viewer. Joins on
  // `candidate_id` (see views/small_molecule_workflows/enzyme_optimization.py).
  const selectedTrajRow = useMemo<Record<string, number | string | null> | null>(() => {
    if (!status.data || !selectedCand) return null
    return (
      status.data.trajectory.find(
        (r) => String(r.candidate_id ?? '') === selectedCand,
      ) ?? null
    )
  }, [status.data, selectedCand])

  return (
    <div className="space-y-4">
      {status.data && <StatusHeader status={status.data} />}
      {status.data && <RewardChart status={status.data} />}
      {status.data && status.data.trajectory.length > 0 && (
        <TrajectoryTable trajectory={status.data.trajectory} />
      )}

      {topK.data && topK.data.candidates.length > 0 && selected && (
        <>
          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Inspect candidate
            </span>
            <select
              value={selectedCand ?? ''}
              onChange={(e) => setSelectedCand(e.target.value)}
              className="rounded-md border border-border bg-background px-3 py-2 text-sm"
            >
              {topK.data.candidates.map((c) => (
                <option key={c.candidate_id} value={c.candidate_id}>
                  {c.candidate_id}
                </option>
              ))}
            </select>
          </label>
          {selectedTrajRow && <CandidateMetrics row={selectedTrajRow} />}
          <MolstarViewer viewerHtml={selected.viewer_html} height={480} />
          <button
            type="button"
            onClick={() => {
              const blob = new Blob([selected.pdb], { type: 'chemical/x-pdb' })
              const url = URL.createObjectURL(blob)
              const a = document.createElement('a')
              a.href = url
              a.download = `${selected.candidate_id}.pdb`
              a.click()
              URL.revokeObjectURL(url)
            }}
            className="rounded-md border border-border px-3 py-2 text-xs hover:bg-accent"
          >
            Download candidate PDB
          </button>
        </>
      )}
      {topK.data && topK.data.candidates.length === 0 && (
        <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
          No completed candidates were logged for this run yet.
        </div>
      )}
    </div>
  )
}

function fmtMetric(val: unknown, digits = 3): string {
  if (val === null || val === undefined || val === '') return '—'
  const n = typeof val === 'number' ? val : Number(val)
  return Number.isFinite(n) ? n.toFixed(digits) : String(val)
}

function CandidateMetrics({ row }: { row: Record<string, number | string | null> }) {
  // Eight metrics + formatting matching the result-dialog layout
  // (views/small_molecule_workflows/enzyme_optimization.py).
  const tiles: { label: string; value: string }[] = [
    { label: 'Composite Reward', value: fmtMetric(row.composite_reward) },
    { label: 'Motif RMSD (Å)',   value: fmtMetric(row.motif_rmsd) },
    { label: 'pLDDT',            value: fmtMetric(row.plddt, 1) },
    { label: 'Boltz',            value: fmtMetric(row.boltz) },
    { label: 'Solubility',       value: fmtMetric(row.solubility) },
    { label: 'Half-Life',        value: fmtMetric(row.half_life) },
    { label: 'Thermostab (°C)',  value: fmtMetric(row.thermostab, 1) },
    { label: 'Immunogenicity',   value: fmtMetric(row.immuno) },
  ]
  return (
    <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
      {tiles.map((t) => (
        <div
          key={t.label}
          className="rounded-md border border-border bg-muted/30 px-3 py-2"
        >
          <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
            {t.label}
          </div>
          <div className="text-base font-medium">{t.value}</div>
        </div>
      ))}
    </div>
  )
}

function StatusHeader({ status }: { status: EnzymeStatusResponse }) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span>
        Status: <code>{status.status}</code> · Stage: <code>{status.job_status}</code>
      </span>
    </div>
  )
}

function RewardChart({ status }: { status: EnzymeStatusResponse }) {
  if (status.iter_max_reward_history.length === 0) {
    return null
  }
  return (
    <div className="rounded-md border border-border bg-card p-2">
      <Plot
        data={[
          {
            x: status.iter_max_reward_history.map((p) => p.step),
            y: status.iter_max_reward_history.map((p) => p.value),
            mode: 'lines+markers',
            name: 'max',
            line: { color: '#86efac' },
          } as never,
          {
            x: status.iter_mean_reward_history.map((p) => p.step),
            y: status.iter_mean_reward_history.map((p) => p.value),
            mode: 'lines+markers',
            name: 'mean',
            line: { color: '#60a5fa' },
          } as never,
        ]}
        layout={{
          title: { text: 'Composite reward by iteration' },
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
  )
}

function TrajectoryTable({ trajectory }: { trajectory: Record<string, number | string | null>[] }) {
  const cols = useMemo<ColumnDef<Record<string, number | string | null>, unknown>[]>(() => {
    const preferred = [
      'candidate_id',
      'iteration',
      'composite_reward',
      'motif_rmsd',
      'plddt',
      'boltz',
      'solubility',
      'half_life',
      'thermostab',
      'immuno',
    ]
    const present = preferred.filter((p) => p in (trajectory[0] ?? {}))
    return present.map((p) => ({
      id: p,
      header: p,
      cell: (ctx) => {
        const v = ctx.row.original[p]
        if (v == null) return '—'
        if (typeof v === 'number') return v.toFixed(3)
        return String(v)
      },
    }))
  }, [trajectory])

  return (
    <details className="rounded-md border border-border">
      <summary className="cursor-pointer px-4 py-2 text-sm">
        Top candidates ({trajectory.length})
      </summary>
      <div className="p-3">
        <DataTable columns={cols} data={trajectory} />
      </div>
    </details>
  )
}

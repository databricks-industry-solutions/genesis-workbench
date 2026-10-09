import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { DispatchSuccess } from '@/components/DispatchSuccess'
import { MolstarViewer } from '@/components/MolstarViewer'
import { RunSearchSection } from '@/components/RunSearchSection'
import type { AntibodyRefRow, DBRunRow } from '@/types/api'
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

// Per-axis reward weights shown as sliders (same control as Guided Enzyme Optimization).
const WEIGHT_AXES: { key: string; label: string; help: string }[] = [
  { key: 'boltz', label: 'Binding — Boltz ipTM', help: 'Antigen–VHH interface confidence from co-folding the complex. The primary binding signal.' },
  { key: 'plddt', label: 'Fold confidence — ESMFold pLDDT', help: 'Mean pLDDT of the designed VHH structure.' },
  { key: 'solubility', label: 'Solubility — NetSolP', help: 'Predicted solubility of the VHH sequence.' },
  { key: 'half_life', label: 'Half-life — PLTNUM', help: 'Anchored against the reference antibodies below (pre-normalised).' },
  { key: 'thermostab', label: 'Thermostability — DeepSTABp Tm', help: 'Predicted melting temperature.' },
  { key: 'immuno', label: 'Low immunogenicity — MHCflurry', help: 'Immunogenic burden (lower is better); the weight favours low-burden designs.' },
  { key: 'liability', label: 'Low sequence liabilities', help: 'Rule-based scan for chemical/developability liability motifs (deamidation, isomerization, N-glyc sequons, free Cys, Met/Trp oxidation). Fewer = more inert/developable (lower raw count is better).' },
]

export function AntibodyDesignTab() {
  const defaults = useQuery({
    queryKey: ['antibody_design', 'defaults'],
    queryFn: api.antibodyDesignDefaults,
    staleTime: Infinity,
  })

  // ─── Form state ────────────────────────────────────────────────────────
  const [antigenPdb, setAntigenPdb] = useState('')
  const [epitope, setEpitope] = useState('')
  const [antigenChain, setAntigenChain] = useState('A')
  const [lenMin, setLenMin] = useState(110)
  const [lenMax, setLenMax] = useState(130)
  const [numSamples, setNumSamples] = useState(8)
  const [numIterations, setNumIterations] = useState(6)
  const [runMpnn, setRunMpnn] = useState(true)
  const [cofold, setCofold] = useState(true)
  const [experiment, setExperiment] = useState('gwb_antibody_design')
  const [runName, setRunName] = useState(`vhh_${ts()}`)
  const [weights, setWeights] = useState<Record<string, number>>({})
  const [refSeqs, setRefSeqs] = useState('')
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
  const [searchToken, setSearchToken] = useState(0)

  // Prefill the demo antigen + epitope + default axis weights from the server.
  useEffect(() => {
    if (!defaults.data) return
    setAntigenPdb((v) => v || defaults.data!.antigen_pdb)
    setAntigenChain((v) => v || defaults.data!.antigen_chain)
    setEpitope((v) => v || defaults.data!.epitope_residues.join(','))
    setWeights((cur) => (Object.keys(cur).length ? cur : { ...defaults.data!.default_weights }))
  }, [defaults.data])

  const start = useMutation({
    mutationFn: api.antibodyDesignStart,
    onSuccess: () => setSearchToken((t) => t + 1),
  })

  const runStart = () => {
    const references: AntibodyRefRow[] = refSeqs
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => ({ sequence: s }))
    start.mutate({
      antigen_pdb: antigenPdb,
      epitope_residues: parseResidues(epitope),
      antigen_chain: antigenChain,
      vhh_length_min: lenMin,
      vhh_length_max: lenMax,
      num_samples: numSamples,
      num_iterations: numIterations,
      weights,
      references,
      half_life_margin: halfLifeMargin,
      resampling_temperature: resampleTemp,
      strategy,
      run_proteinmpnn: runMpnn,
      cofold_antigen: cofold,
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
    antigenPdb.trim().length > 0 &&
    antigenChain.trim() &&
    experiment.trim() &&
    runName.trim() &&
    lenMin <= lenMax

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Design VHH Antibodies with Guided Search</h3>
        <p className="text-xs text-muted-foreground">
          Generate single-domain (VHH / nanobody) antibodies against an antigen epitope with
          RFD4-Proteina (in-process on an H100), then iterate: anarcii CDR numbering → ProteinMPNN
          framework redesign → ESMFold → score every candidate on binding (Boltz ipTM), fold
          confidence, <em>and</em> developability (solubility, half-life, Tm, low immunogenicity). Each
          iteration's composite reward biases the next round. Long-running — the orchestrator runs as a
          Databricks Job; track progress in <strong>Search Past Runs</strong> below.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* LEFT — antigen + loop params */}
        <div className="space-y-3">
          <div className="rounded-md border border-border bg-card p-3 text-xs">
            <div className="mb-2 font-medium uppercase tracking-wide text-muted-foreground">
              Antigen target
            </div>
            <label className="block">
              <span className="mb-1 block text-muted-foreground">Antigen structure (PDB)</span>
              <textarea
                rows={8}
                value={antigenPdb}
                onChange={(e) => setAntigenPdb(e.target.value)}
                placeholder="Paste the antigen PDB (ATOM records for the target chain)…"
                className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-[11px]"
              />
            </label>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <label className="block">
                <span className="mb-1 block text-muted-foreground">Antigen chain</span>
                <input
                  value={antigenChain}
                  onChange={(e) => setAntigenChain(e.target.value)}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-muted-foreground">Epitope residues (CSV)</span>
                <input
                  value={epitope}
                  onChange={(e) => setEpitope(e.target.value)}
                  placeholder="31,52,99"
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
                <span className="mb-1 block text-muted-foreground">VHH length min</span>
                <input
                  type="number" min={80} max={160} value={lenMin}
                  onChange={(e) => setLenMin(parseInt(e.target.value || '110'))}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-muted-foreground">VHH length max</span>
                <input
                  type="number" min={80} max={160} value={lenMax}
                  onChange={(e) => setLenMax(parseInt(e.target.value || '130'))}
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
              <span>ProteinMPNN framework redesign (fix CDRs)</span>
            </label>
            <label className="mt-2 flex items-center gap-2">
              <input type="checkbox" checked={cofold} onChange={(e) => setCofold(e.target.checked)} />
              <span>Boltz co-fold antigen + VHH (binding)</span>
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

          <button
            onClick={runStart}
            disabled={!canStart}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
          >
            {start.isPending ? 'Dispatching…' : 'Design VHH antibodies'}
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
              iteration's batch before the weighted sum (except half-life — pre-normalised via the
              anchor sigmoid).
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
              Half-life anchor (reference antibodies)
            </summary>
            <p className="mb-2 mt-2 text-[11px] text-muted-foreground">
              Optional. One reference antibody sequence per line; the half-life axis is anchored
              against their PLTNUM scores. Leave empty for a neutral half-life contribution.
            </p>
            <textarea
              rows={3}
              value={refSeqs}
              onChange={(e) => setRefSeqs(e.target.value)}
              placeholder="QVQLVESGGGLVQ...&#10;EVQLVESGGGLVQ..."
              className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-[11px]"
            />
            <label className="mt-2 block">
              <div className="flex justify-between text-muted-foreground">
                <span>Anchor margin β</span>
                <span>{halfLifeMargin.toFixed(2)}</span>
              </div>
              <input
                type="range" min={0.01} max={0.5} step={0.01} value={halfLifeMargin}
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
          searchKey={['antibody_design', 'search'] as const}
          searchFn={api.antibodyDesignSearch}
          detailLabel="Top reward"
          initialText="vhh"
          viewableStatuses={['complete']}
          detailColClass="w-40"
          searchToken={searchToken}
          renderDialog={(run: DBRunRow) => <AntibodyResultBody runId={run.run_id} />}
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
  ['boltz', 'Binding (ipTM)'],
  ['plddt', 'Fold pLDDT'],
  ['solubility', 'Solubility'],
  ['half_life', 'Half-life'],
  ['thermostab', 'Tm'],
  ['immuno', 'Immuno burden'],
  ['liability', 'Seq liabilities (count)'],
  ['liability_detail', 'Liability sites'],
]

function AntibodyResultBody({ runId }: { runId: string }) {
  const status = useQuery({
    queryKey: ['antibody_design', 'status', runId],
    queryFn: () => api.antibodyDesignStatus(runId),
    refetchInterval: (q) => (q.state.data?.job_status === 'complete' ? false : 10000),
  })
  const topK = useQuery({
    queryKey: ['antibody_design', 'topk', runId],
    queryFn: () => api.antibodyDesignTopK(runId),
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

import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { DispatchSuccess } from '@/components/DispatchSuccess'
import { MolstarViewer } from '@/components/MolstarViewer'
import { RunSearchSection } from '@/components/RunSearchSection'
import type { DBRunRow } from '@/types/api'

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

export function AntibodyDesignTab() {
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
  const [searchToken, setSearchToken] = useState(0)

  // Prefill a working demo antigen (hen egg-white lysozyme) + a plausible epitope
  // so the form runs out of the box.
  const defaults = useQuery({ queryKey: ['antibody_design', 'defaults'], queryFn: api.antibodyDesignDefaults })
  useEffect(() => {
    if (defaults.data) {
      setAntigenPdb((v) => v || defaults.data!.antigen_pdb)
      setAntigenChain((v) => v || defaults.data!.antigen_chain)
      setEpitope((v) => v || defaults.data!.epitope_residues.join(','))
    }
  }, [defaults.data])

  const start = useMutation({
    mutationFn: () =>
      api.antibodyDesignStart({
        antigen_pdb: antigenPdb,
        epitope_residues: parseResidues(epitope),
        antigen_chain: antigenChain,
        vhh_length_min: lenMin,
        vhh_length_max: lenMax,
        num_samples: numSamples,
        num_iterations: numIterations,
        weights: {}, // backend fills DEFAULT_AXIS_WEIGHTS
        references: [],
        half_life_margin: 0.05,
        resampling_temperature: 0.1,
        strategy: 'resample',
        run_proteinmpnn: runMpnn,
        cofold_antigen: cofold,
        convergence_threshold: 0.01,
        convergence_window: 2,
        target_reward: null,
        best_k_target: null,
        best_k_threshold: null,
        mlflow_experiment: experiment,
        mlflow_run_name: runName,
      }),
    onSuccess: () => setSearchToken((t) => t + 1),
  })

  const canRun =
    !start.isPending &&
    antigenPdb.trim() &&
    antigenChain.trim() &&
    experiment.trim() &&
    runName.trim() &&
    lenMax >= lenMin

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Antibody Design (VHH / nanobody)</h3>
        <p className="text-xs text-muted-foreground">
          Generate single-domain (VHH) antibodies against an antigen epitope with{' '}
          <strong>RFD4-Proteina</strong> (loaded in-process on an H100), then optimize them over a
          reward-weighted loop scored on binding (Boltz ipTM), fold confidence (ESMFold pLDDT), and
          developability (solubility, half-life, Tm, low immunogenicity). Runs as a long GPU job —
          launch and review results under Search Past Runs.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(380px,520px)_1fr]">
        {/* Left: form */}
        <div className="space-y-3">
          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Antigen structure (PDB text)
            </span>
            <textarea
              value={antigenPdb}
              onChange={(e) => setAntigenPdb(e.target.value)}
              placeholder="Paste the antigen PDB (ATOM records for the target chain)…"
              rows={6}
              className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-[11px]"
            />
          </label>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Antigen chain</span>
              <input
                value={antigenChain}
                onChange={(e) => setAntigenChain(e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Epitope residues (CSV)
              </span>
              <input
                value={epitope}
                onChange={(e) => setEpitope(e.target.value)}
                placeholder="31,52,99"
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">VHH length min</span>
              <input
                type="number" min={80} max={160} value={lenMin}
                onChange={(e) => setLenMin(parseInt(e.target.value) || 110)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">VHH length max</span>
              <input
                type="number" min={80} max={160} value={lenMax}
                onChange={(e) => setLenMax(parseInt(e.target.value) || 130)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Candidates / iteration (K)
              </span>
              <input
                type="number" min={2} max={32} value={numSamples}
                onChange={(e) => setNumSamples(parseInt(e.target.value) || 8)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Iterations (N)</span>
              <input
                type="number" min={1} max={30} value={numIterations}
                onChange={(e) => setNumIterations(parseInt(e.target.value) || 6)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
          </div>

          <div className="flex flex-wrap gap-4 text-xs">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={runMpnn} onChange={(e) => setRunMpnn(e.target.checked)} />
              <span className="text-muted-foreground">ProteinMPNN framework redesign (fix CDRs)</span>
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={cofold} onChange={(e) => setCofold(e.target.checked)} />
              <span className="text-muted-foreground">Boltz co-fold antigen + VHH (binding)</span>
            </label>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">MLflow experiment</span>
              <input
                value={experiment}
                onChange={(e) => setExperiment(e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Run name</span>
              <input
                value={runName}
                onChange={(e) => setRunName(e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
          </div>

          <button
            onClick={() => start.mutate()}
            disabled={!canRun}
            className="w-full rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
          >
            {start.isPending ? 'Dispatching…' : 'Design VHH antibodies'}
          </button>

          {start.data && <DispatchSuccess jobRunId={start.data.job_run_id} runUrl={start.data.run_url} />}
          {start.error && <p className="text-[11px] text-destructive">{String(start.error)}</p>}
        </div>

        {/* Right: Search past runs */}
        <div>
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
        <div className="text-xs text-muted-foreground">
          Stage: {status.data?.job_status || '…'}
        </div>
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

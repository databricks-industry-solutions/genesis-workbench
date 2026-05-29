import { useMemo, useState } from 'react'

import { RealtimeProgress } from '@/components/RealtimeProgress'
import { useSseMutation } from '@/hooks/useSseMutation'
import type { AdmetResponse } from '@/types/api'
import { cn } from '@/lib/utils'

const EXAMPLE_SMILES = `COc(cc1)ccc1C#N
CC(=O)Oc1ccccc1C(=O)O
CC(C)NCC(O)c1ccc(O)c(O)c1
C1CCCCC1
c1ccc2[nH]ccc2c1`

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

function fmtPct(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '—'
  return `${(v * 100).toFixed(1)}%`
}

function fmtFloat(v: number | null | undefined, digits = 3): string {
  if (v == null || Number.isNaN(v)) return '—'
  return v.toFixed(digits)
}

function riskColor(v: number | null | undefined): string {
  if (v == null) return 'text-muted-foreground'
  if (v >= 0.7) return 'text-destructive'
  if (v >= 0.3) return 'text-amber-300'
  return 'text-success'
}

function riskLabel(v: number | null | undefined): string {
  if (v == null) return 'N/A'
  if (v >= 0.7) return 'High'
  if (v >= 0.3) return 'Medium'
  return 'Low'
}

export function AdmetSafetyTab() {
  const [smilesText, setSmilesText] = useState(EXAMPLE_SMILES)
  const [runBbbp, setRunBbbp] = useState(true)
  const [runClintox, setRunClintox] = useState(true)
  const [runAdmet, setRunAdmet] = useState(true)
  const [experiment, setExperiment] = useState('gwb_admet_safety')
  const [runName, setRunName] = useState(`admet_profiling_${ts()}`)
  const [expandedIdx, setExpandedIdx] = useState<number | null>(0)

  const profile = useSseMutation<
    {
      smiles: string[]
      run_bbbp: boolean
      run_clintox: boolean
      run_admet: boolean
      mlflow_experiment: string
      mlflow_run_name: string
    },
    AdmetResponse
  >('/api/small_molecules/admet/stream')

  const smilesList = useMemo(
    () =>
      smilesText
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean),
    [smilesText],
  )

  const canRun =
    !profile.isPending &&
    smilesList.length > 0 &&
    experiment.trim() &&
    runName.trim() &&
    (runBbbp || runClintox || runAdmet)

  const runProfile = () =>
    profile.start({
      smiles: smilesList,
      run_bbbp: runBbbp,
      run_clintox: runClintox,
      run_admet: runAdmet,
      mlflow_experiment: experiment,
      mlflow_run_name: runName,
    })

  const mlflowUrl = profile.data
    ? `${window.location.protocol}//${window.location.host.replace(/-\d+\.aws\.databricksapps\.com$/, '')}/ml/experiments/${profile.data.experiment_id}/runs/${profile.data.run_id}`
    : null

  const enabledStages = useMemo(() => {
    const stages: { label: string; pctEnd: number }[] = []
    const enabledCount = Number(runBbbp) + Number(runClintox) + Number(runAdmet)
    if (enabledCount === 0) return stages
    const step = 85 / enabledCount
    let acc = 10
    if (runBbbp) {
      acc += step
      stages.push({ label: 'BBB penetration (Chemprop)', pctEnd: Math.round(acc) })
    }
    if (runClintox) {
      acc += step
      stages.push({ label: 'Clinical toxicity (Chemprop)', pctEnd: Math.round(acc) })
    }
    if (runAdmet) {
      acc += step
      stages.push({ label: 'ADMET multi-task (Chemprop)', pctEnd: Math.round(acc) })
    }
    stages.push({ label: 'Logging to MLflow', pctEnd: 100 })
    return stages
  }, [runBbbp, runClintox, runAdmet])

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Profile ADMET and Safety Risks</h3>
        <p className="text-xs text-muted-foreground">
          Score one or more small molecules across absorption / distribution / metabolism /
          excretion / toxicity axes using Chemprop D-MPNN models. Predictors are independent —
          run any subset. Results log to MLflow alongside each prediction set.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(360px,460px)_1fr]">
        {/* Left form */}
        <div className="space-y-3">
          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              SMILES (one per line)
            </span>
            <textarea
              rows={8}
              value={smilesText}
              onChange={(e) => setSmilesText(e.target.value)}
              placeholder="COc(cc1)ccc1C#N"
              className="w-full rounded-md border border-border bg-background p-3 font-mono text-xs"
            />
            <span className="mt-1 block text-[10px] text-muted-foreground">
              {smilesList.length} molecule{smilesList.length === 1 ? '' : 's'} parsed
            </span>
          </label>

          <div className="space-y-2 rounded-md border border-border bg-card p-3 text-xs">
            <div className="mb-1 font-medium uppercase tracking-wide text-muted-foreground">
              Predictors
            </div>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={runBbbp}
                onChange={(e) => setRunBbbp(e.target.checked)}
              />
              <span>BBB penetration (blood-brain barrier probability)</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={runClintox}
                onChange={(e) => setRunClintox(e.target.checked)}
              />
              <span>Clinical toxicity (failure probability)</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={runAdmet}
                onChange={(e) => setRunAdmet(e.target.checked)}
              />
              <span>ADMET properties (multi-task regression)</span>
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

          <div className="flex gap-2">
            <button
              onClick={runProfile}
              disabled={!canRun}
              className="flex-1 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
            >
              {profile.isPending ? 'Profiling…' : 'Run ADMET Profiling'}
            </button>
            <button
              onClick={() => profile.reset()}
              disabled={!profile.data && !profile.error}
              className="rounded-md border border-border px-4 py-2 text-sm hover:bg-accent disabled:opacity-50"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Right results */}
        <div className="space-y-3">
          {profile.isPending && (
            <RealtimeProgress
              title={`Profiling ${smilesList.length} molecule${smilesList.length === 1 ? '' : 's'}`}
              pct={profile.progress?.pct ?? 0}
              msg={profile.progress?.msg ?? 'Starting…'}
              stages={enabledStages}
            />
          )}

          {profile.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
              {String(profile.error)}
            </div>
          )}

          {profile.data?.warnings?.map((w, i) => (
            <div
              key={i}
              className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-200"
            >
              {w}
            </div>
          ))}

          {profile.data && profile.data.smiles.length > 0 && (
            <>
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium">
                  Results — {profile.data.smiles.length} molecule
                  {profile.data.smiles.length === 1 ? '' : 's'}
                </h4>
                {mlflowUrl && (
                  <a
                    href={mlflowUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-primary hover:underline"
                  >
                    View MLflow run ↗
                  </a>
                )}
              </div>

              <div className="space-y-2">
                {profile.data.smiles.map((smi, idx) => {
                  const expanded = expandedIdx === idx
                  const bbbp = profile.data!.bbbp?.[idx] ?? null
                  const clintox = profile.data!.clintox?.[idx] ?? null
                  const admetRow = profile.data!.admet?.[idx] ?? null
                  const admetPropCount = admetRow
                    ? Object.values(admetRow).filter((v) => v != null && !Number.isNaN(v as number))
                        .length
                    : 0
                  return (
                    <div
                      key={idx}
                      className="rounded-md border border-border bg-card"
                    >
                      <button
                        type="button"
                        onClick={() => setExpandedIdx(expanded ? null : idx)}
                        className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left hover:bg-accent/40"
                      >
                        <code className="truncate font-mono text-xs">{smi}</code>
                        <span className="text-[10px] text-muted-foreground">
                          {expanded ? '▾' : '▸'}
                        </span>
                      </button>
                      {expanded && (
                        <div className="space-y-3 border-t border-border p-3">
                          <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
                            <div className="rounded-md border border-border bg-muted/30 p-2 text-xs">
                              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                                BBB penetration
                              </div>
                              <div className="text-sm font-medium">{fmtPct(bbbp)}</div>
                              {bbbp != null && (
                                <div className={cn('mt-0.5 text-[10px]', riskColor(bbbp))}>
                                  {bbbp >= 0.5 ? 'Permeable' : 'Non-permeable'}
                                </div>
                              )}
                            </div>
                            <div className="rounded-md border border-border bg-muted/30 p-2 text-xs">
                              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                                Toxicity risk
                              </div>
                              <div className="text-sm font-medium">{fmtPct(clintox)}</div>
                              {clintox != null && (
                                <div className={cn('mt-0.5 text-[10px]', riskColor(clintox))}>
                                  {riskLabel(clintox)}
                                </div>
                              )}
                            </div>
                            <div className="rounded-md border border-border bg-muted/30 p-2 text-xs">
                              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                                ADMET properties
                              </div>
                              <div className="text-sm font-medium">
                                {admetRow ? `${admetPropCount} predicted` : '—'}
                              </div>
                            </div>
                          </div>

                          {admetRow && (
                            <div>
                              <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                                ADMET breakdown
                              </div>
                              <div className="grid grid-cols-1 gap-1 text-xs sm:grid-cols-2">
                                {Object.entries(admetRow).map(([k, v]) => (
                                  <div
                                    key={k}
                                    className="flex justify-between gap-3 rounded-md bg-muted/20 px-2 py-1"
                                  >
                                    <span className="text-muted-foreground">{k}</span>
                                    <span className="font-mono">{fmtFloat(v as number)}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </>
          )}

          {!profile.isPending && !profile.data && !profile.error && (
            <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
              Paste one or more SMILES strings and pick which predictors to run. Each enabled
              Chemprop endpoint will score every molecule; collapsible cards below summarise
              BBB penetration, clinical toxicity, and the per-task ADMET breakdown.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

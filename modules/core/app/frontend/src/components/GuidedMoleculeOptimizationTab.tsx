import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { MaterialIcon } from '@/components/MaterialIcon'
import { useClipboard } from '@/stores/clipboard'
import type { MolOptStatus, MolOptTopKItem } from '@/types/api'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

export function GuidedMoleculeOptimizationTab() {
  const [seeds, setSeeds] = useState('')
  const [numIterations, setNumIterations] = useState(5)
  const [numSamples, setNumSamples] = useState(24)
  const [selectTop, setSelectTop] = useState(3)
  const [dockTopK, setDockTopK] = useState(5)
  const [wQed, setWQed] = useState(1.0)
  const [wAdmet, setWAdmet] = useState(1.0)
  const [runName, setRunName] = useState(`mol_opt_${ts()}`)
  const [runId, setRunId] = useState<string | null>(null)

  const clipAdd = useClipboard((s) => s.add)

  const seedList = seeds.split('\n').map((s) => s.trim()).filter(Boolean)

  const start = useMutation({
    mutationFn: () =>
      api.molOptStart({
        seed_smiles: seedList,
        num_samples: numSamples,
        num_iterations: numIterations,
        select_top: selectTop,
        dock_top_k: dockTopK,
        weights: { qed: wQed, admet: wAdmet },
        temperature: 1.2,
        randomness: 2.0,
        mlflow_run_name: runName,
      }),
    onSuccess: (d) => setRunId(d.mlflow_run_id),
  })

  const status = useQuery<MolOptStatus>({
    queryKey: ['molopt', 'status', runId],
    queryFn: () => api.molOptStatus(runId as string),
    enabled: !!runId,
    refetchInterval: (q) => {
      const js = q.state.data?.job_status
      return js === 'complete' || js === 'failed' ? false : 4000
    },
  })

  const done = status.data?.job_status === 'complete'
  const topk = useQuery({
    queryKey: ['molopt', 'topk', runId],
    queryFn: () => api.molOptTopK(runId as string),
    enabled: !!runId && done,
  })

  const search = useMutation({
    mutationFn: (text: string) => api.molOptSearch('run_name', text),
  })

  const fmt = (v: number | null | undefined, d = 3) =>
    v == null || Number.isNaN(v) ? '—' : v.toFixed(d)

  const topColumns: ColumnDef<MolOptTopKItem, unknown>[] = [
    { id: 'idx', header: '#', cell: ({ row }) => row.index + 1 },
    {
      id: 'smiles', header: 'SMILES', accessorKey: 'smiles',
      cell: ({ row }) => <span className="font-mono text-xs">{row.original.smiles}</span>,
    },
    { id: 'reward', header: 'Reward', cell: ({ row }) => fmt(row.original.reward) },
    { id: 'qed', header: 'QED', cell: ({ row }) => fmt(row.original.qed) },
    {
      id: 'tox', header: 'ClinTox', cell: ({ row }) => fmt(row.original.tox),
    },
    {
      id: 'dock', header: 'Dock', cell: ({ row }) => fmt(row.original.dock_confidence),
    },
    {
      id: 'clip', header: '',
      cell: ({ row }) => (
        <button
          type="button"
          onClick={() => clipAdd({ kind: 'molecule', value: row.original.smiles, source: 'Mol Opt' })}
          className="inline-flex items-center gap-1 rounded-md border border-primary/50 bg-primary/10 px-2 py-0.5 text-xs text-primary hover:bg-primary/20"
        >
          <MaterialIcon name="assignment" className="text-[14px] text-cyan-400" /> Clip
        </button>
      ),
    },
  ]

  const traj = status.data?.best_reward_history ?? []
  const meanTraj = status.data?.mean_reward_history ?? []
  const maxReward = Math.max(0.001, ...traj.map((p) => p.value))

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Guided Molecule Optimization</h3>
        <p className="text-xs text-muted-foreground">
          A design-make-test loop: GenMol grows the seed motif into candidates → each scored on{' '}
          <strong>QED</strong> (drug-likeness) + <strong>ADMET</strong> clinical-tox → the best
          reseed the next round → the top-K are docked into the target. Seed from a target’s binding
          motif (copy a scaffold from “Small Molecule Design”). Runs as a batch job — track it below.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(320px,420px)_1fr]">
        {/* Left: form */}
        <div className="space-y-3">
          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Seed scaffold SMILES — one per line
            </span>
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
              <input type="number" min={1} max={20} value={numIterations}
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
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Dock top-K</span>
              <input type="number" min={0} max={20} value={dockTopK}
                onChange={(e) => setDockTopK(parseInt(e.target.value) || 0)}
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

          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Run name</span>
            <input value={runName} onChange={(e) => setRunName(e.target.value)}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" />
          </label>

          <button
            onClick={() => start.mutate()}
            disabled={start.isPending || seedList.length === 0}
            className="w-full rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
          >
            {start.isPending ? 'Starting…' : 'Start optimization'}
          </button>
          {start.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive">
              {String(start.error)}
            </div>
          )}

          {/* Search past runs */}
          <div className="rounded-md border border-border bg-card p-3 text-xs">
            <div className="mb-1.5 font-medium uppercase tracking-wide text-muted-foreground">
              Search past runs
            </div>
            <div className="flex gap-2">
              <input
                placeholder="run name contains…"
                onKeyDown={(e) => e.key === 'Enter' && search.mutate((e.target as HTMLInputElement).value)}
                className="min-w-0 flex-1 rounded-md border border-border bg-background px-2 py-1.5 text-sm"
              />
            </div>
            {search.data?.runs?.map((r) => (
              <button key={r.run_id} type="button" onClick={() => setRunId(r.run_id)}
                className="mt-1 block w-full truncate rounded border border-border bg-background px-2 py-1 text-left hover:bg-accent">
                {r.run_name} <span className="text-[10px] text-muted-foreground">· {r.job_status}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Right: status + trajectory + top-K */}
        <div className="space-y-3">
          {!runId && (
            <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
              Seed a scaffold and Start. The loop runs as a batch job; progress and the top-K appear
              here, and you can reopen a past run via Search.
            </div>
          )}

          {runId && (
            <div className="rounded-md border border-border bg-card p-3 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-medium">{status.data?.run_name || runName}</span>
                <span className={
                  'rounded px-2 py-0.5 text-[11px] ' +
                  (status.data?.job_status === 'complete' ? 'bg-success/15 text-success'
                    : status.data?.job_status === 'failed' ? 'bg-destructive/15 text-destructive'
                    : 'bg-primary/10 text-primary')
                }>
                  {status.data?.job_status || 'submitted'}
                </span>
              </div>
              {/* trajectory */}
              {traj.length > 0 && (
                <div className="mt-2 space-y-1">
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
              {status.data && status.data.job_status !== 'complete' && status.data.job_status !== 'failed' && (
                <p className="mt-2 text-[11px] text-muted-foreground">
                  Running… first iteration appears once the job warms up (cluster + GenMol cold-start).
                </p>
              )}
            </div>
          )}

          {topk.data?.top_k && topk.data.top_k.length > 0 && (
            <>
              <h4 className="text-sm font-medium">Top {topk.data.top_k.length} optimized molecules</h4>
              <DataTable columns={topColumns} data={topk.data.top_k} emptyText="No molecules" />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

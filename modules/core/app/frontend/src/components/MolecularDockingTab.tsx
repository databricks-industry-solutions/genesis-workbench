import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { ClipboardPaste } from '@/components/ClipboardPaste'
import { DataTable } from '@/components/DataTable'
import { MolstarViewer } from '@/components/MolstarViewer'
import { RealtimeProgress } from '@/components/RealtimeProgress'
import { useSseMutation } from '@/hooks/useSseMutation'
import type { DockingPose, MolecularDockingResponse } from '@/types/api'
import { StructurePicker } from '@/components/StructurePicker'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

export function MolecularDockingTab() {
  // Defaults: SMILES + 50-residue PDB excerpt of chain A from 6agt, fetched
  // server-side so the React bundle doesn't carry a multi-KB string.
  const example = useQuery({
    queryKey: ['small_molecule', 'diffdock', 'example'],
    queryFn: api.diffdockExample,
    staleTime: Infinity,
  })

  const [smiles, setSmiles] = useState('')
  const [proteinPdb, setProteinPdb] = useState('')
  const [numSamples, setNumSamples] = useState(5)
  const [experiment, setExperiment] = useState('gwb_molecular_docking')
  const [runName, setRunName] = useState(`molecular_docking_${ts()}`)
  const [selectedRank, setSelectedRank] = useState<number | null>(null)

  // Seed the form once the example payload loads.
  useEffect(() => {
    if (!example.data) return
    setSmiles((cur) => cur || example.data!.smiles)
    setProteinPdb((cur) => cur || example.data!.pdb)
  }, [example.data])

  const dock = useSseMutation<
    {
      protein_pdb: string
      ligand_smiles: string
      num_samples: number
      mlflow_experiment: string
      mlflow_run_name: string
    },
    MolecularDockingResponse
  >('/api/small_molecule/diffdock/stream')

  // When a new result arrives, jump to the top-ranked pose so the viewer
  // shows something meaningful immediately.
  useEffect(() => {
    if (!dock.data?.poses?.length) {
      setSelectedRank(null)
      return
    }
    const firstOk = dock.data.poses.find((p) => !p.error) ?? dock.data.poses[0]
    setSelectedRank(firstOk.rank)
  }, [dock.data])

  const canRun = Boolean(
    smiles.trim() && proteinPdb.trim() && experiment.trim() && runName.trim() && !dock.isPending,
  )

  const runDocking = () =>
    dock.start({
      protein_pdb: proteinPdb,
      ligand_smiles: smiles,
      num_samples: numSamples,
      mlflow_experiment: experiment,
      mlflow_run_name: runName,
    })

  const selectedPose = useMemo<DockingPose | null>(() => {
    if (!dock.data || selectedRank == null) return null
    return dock.data.poses.find((p) => p.rank === selectedRank) ?? null
  }, [dock.data, selectedRank])

  const tableColumns = useMemo<ColumnDef<DockingPose, unknown>[]>(
    () => [
      { id: 'rank', header: 'Rank', accessorKey: 'rank' },
      {
        id: 'confidence',
        header: 'Confidence',
        accessorFn: (r) => r.confidence.toFixed(4),
      },
      {
        id: 'status',
        header: 'Status',
        cell: (ctx) =>
          ctx.row.original.error ? (
            <span className="text-destructive">Failed</span>
          ) : (
            <span className="text-success">OK</span>
          ),
      },
    ],
    [],
  )

  const mlflowUrl = dock.data
    ? `${window.location.protocol}//${window.location.host.replace(/-\d+\.aws\.databricksapps\.com$/, '')}/ml/experiments/${dock.data.experiment_id}/runs/${dock.data.run_id}`
    : null

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Simulate Ligand Binding to a Target</h3>
        <p className="text-xs text-muted-foreground">
          Predict 3D binding poses for a protein–ligand complex using{' '}
          <a
            href="https://github.com/gcorso/DiffDock"
            target="_blank"
            rel="noreferrer"
            className="text-primary hover:underline"
          >
            DiffDock
          </a>{' '}
          — a diffusion model that generates and ranks candidate poses with a confidence score.
          Computes ESM-2 embeddings of the target first, then runs the docking sampler with the
          embeddings pre-computed (split-endpoint pattern keeps each call under the proxy timeout).
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(320px,420px)_1fr]">
        {/* Left form */}
        <div className="space-y-3">
          <label className="block text-xs">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="block uppercase tracking-wide text-muted-foreground">
                Ligand (SMILES)
              </span>
              <ClipboardPaste kind="molecule" label="Paste molecule" onPick={(it) => setSmiles(it.value)} />
            </div>
            <input
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="COc(cc1)ccc1C#N"
              className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
            />
          </label>

          <label className="block text-xs">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="block uppercase tracking-wide text-muted-foreground">
                Target protein (PDB)
              </span>
              <StructurePicker onPick={setProteinPdb} />
            </div>
            <textarea
              rows={10}
              value={proteinPdb}
              onChange={(e) => setProteinPdb(e.target.value)}
              placeholder="ATOM…  — or use “Pick a structure from a prior run”"
              className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-[10px] leading-tight"
            />
          </label>

          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Number of poses
            </span>
            <input
              type="range"
              min={1}
              max={20}
              step={1}
              value={numSamples}
              onChange={(e) => setNumSamples(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="mt-1 text-muted-foreground">{numSamples}</div>
          </label>

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
              onClick={runDocking}
              disabled={!canRun}
              className="flex-1 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
            >
              {dock.isPending ? 'Docking…' : 'Run docking'}
            </button>
            <button
              onClick={() => dock.reset()}
              disabled={!dock.data && !dock.error}
              className="rounded-md border border-border px-4 py-2 text-sm hover:bg-accent disabled:opacity-50"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Right viewer + results */}
        <div className="space-y-3">
          {dock.isPending && (
            <RealtimeProgress
              title={`Generating ${numSamples} pose${numSamples > 1 ? 's' : ''}`}
              pct={dock.progress?.pct ?? 0}
              msg={dock.progress?.msg ?? 'Starting…'}
              stages={[
                { label: 'Computing ESM-2 embeddings', pctEnd: 25 },
                { label: 'Running DiffDock pose generation', pctEnd: 85 },
                { label: 'Building viewers + logging', pctEnd: 100 },
              ]}
            />
          )}

          {dock.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
              {String(dock.error)}
            </div>
          )}

          {dock.data && dock.data.poses.length === 0 && (
            <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
              DiffDock returned no poses.
            </div>
          )}

          {dock.data && dock.data.poses.length > 0 && (
            <>
              <div className="flex flex-wrap items-end gap-3">
                <label className="block text-xs">
                  <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                    Pose
                  </span>
                  <select
                    value={selectedRank ?? ''}
                    onChange={(e) => setSelectedRank(parseInt(e.target.value))}
                    className="rounded-md border border-border bg-background px-3 py-2 text-sm"
                  >
                    {dock.data.poses.map((p) => (
                      <option key={p.rank} value={p.rank}>
                        Rank {p.rank} — confidence {p.confidence.toFixed(4)}
                        {p.error ? ' (failed)' : ''}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="text-xs text-muted-foreground">
                  {dock.data.n_success}/{dock.data.poses.length} pose
                  {dock.data.poses.length === 1 ? '' : 's'} succeeded
                </div>
                {mlflowUrl && (
                  <a
                    href={mlflowUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="ml-auto text-xs text-primary hover:underline"
                  >
                    View MLflow run ↗
                  </a>
                )}
              </div>

              {selectedPose?.error ? (
                <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-200">
                  Pose generation reported an error: {selectedPose.error}
                </div>
              ) : null}

              <MolstarViewer viewerHtml={selectedPose?.viewer_html ?? null} height={520} />

              <details className="rounded-md border border-border">
                <summary className="cursor-pointer px-4 py-2 text-sm">
                  All docking results
                </summary>
                <div className="p-3">
                  <DataTable columns={tableColumns} data={dock.data.poses} />
                </div>
              </details>
            </>
          )}

          {!dock.isPending && !dock.data && !dock.error && (
            <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
              Submit the form to dock the selected ligand against the target protein.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

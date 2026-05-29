import { useEffect, useMemo, useState } from 'react'
import type { ColumnDef } from '@tanstack/react-table'

import { DataTable } from '@/components/DataTable'
import { MolstarViewer } from '@/components/MolstarViewer'
import { RealtimeProgress } from '@/components/RealtimeProgress'
import { useSseMutation } from '@/hooks/useSseMutation'
import type { MotifScaffold, MotifScaffoldingResponse } from '@/types/api'

const EXAMPLE_MOTIF_PDB = `ATOM      1  N   HIS B   1       5.123   8.456   2.345  1.00 15.00           N
ATOM      2  CA  HIS B   1       5.891   7.234   2.789  1.00 15.00           C
ATOM      3  C   HIS B   1       7.321   7.567   3.123  1.00 15.00           C
ATOM      4  O   HIS B   1       7.654   8.678   3.567  1.00 15.00           O
ATOM      5  CB  HIS B   1       5.456   6.123   3.678  1.00 15.00           C
ATOM      6  CG  HIS B   1       4.012   5.789   3.456  1.00 15.00           C
ATOM      7  ND1 HIS B   1       3.123   6.567   4.123  1.00 15.00           N
ATOM      8  CE1 HIS B   1       1.890   6.012   3.890  1.00 15.00           C
ATOM      9  NE2 HIS B   1       1.987   4.890   3.123  1.00 15.00           N
ATOM     10  CD2 HIS B   1       3.234   4.678   2.890  1.00 15.00           C
ATOM     11  N   ASP B   2       8.123   6.567   2.890  1.00 15.00           N
ATOM     12  CA  ASP B   2       9.543   6.789   3.234  1.00 15.00           C
ATOM     13  C   ASP B   2      10.234   5.567   3.890  1.00 15.00           C
ATOM     14  O   ASP B   2       9.678   4.456   4.012  1.00 15.00           O
ATOM     15  CB  ASP B   2      10.123   7.890   2.345  1.00 15.00           C
ATOM     16  CG  ASP B   2      11.567   8.123   2.678  1.00 15.00           C
ATOM     17  OD1 ASP B   2      12.234   7.234   3.123  1.00 15.00           O
ATOM     18  OD2 ASP B   2      11.890   9.234   2.345  1.00 15.00           O
ATOM     19  N   SER B   3      11.456   5.678   4.234  1.00 15.00           N
ATOM     20  CA  SER B   3      12.234   4.567   4.890  1.00 15.00           C
ATOM     21  C   SER B   3      13.678   4.890   5.234  1.00 15.00           C
ATOM     22  O   SER B   3      14.123   5.987   5.012  1.00 15.00           O
ATOM     23  CB  SER B   3      11.890   3.234   4.234  1.00 15.00           C
ATOM     24  OG  SER B   3      12.567   2.123   4.678  1.00 15.00           O
HETATM   25  C1  LIG B   1       6.500   3.200   5.100  1.00  5.00           C
HETATM   26  O1  LIG B   1       7.200   2.100   5.500  1.00  5.00           O
HETATM   27  N1  LIG B   1       5.300   3.500   5.800  1.00  5.00           N
END
`

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

export function MotifScaffoldingTab() {
  const [motifPdb, setMotifPdb] = useState(EXAMPLE_MOTIF_PDB)
  const [targetChain, setTargetChain] = useState('B')
  const [lenMin, setLenMin] = useState(50)
  const [lenMax, setLenMax] = useState(80)
  const [numSamples, setNumSamples] = useState(2)
  const [optimizeMpnn, setOptimizeMpnn] = useState(true)
  const [validateEsmfold, setValidateEsmfold] = useState(true)
  const [experiment, setExperiment] = useState('gwb_motif_scaffolding')
  const [runName, setRunName] = useState(`motif_scaffolding_${ts()}`)

  const [selectedIdx, setSelectedIdx] = useState(0)

  const job = useSseMutation<
    {
      motif_pdb: string
      target_chain: string
      scaffold_length_min: number
      scaffold_length_max: number
      num_samples: number
      optimize_mpnn: boolean
      validate_esmfold: boolean
      mlflow_experiment: string
      mlflow_run_name: string
    },
    MotifScaffoldingResponse
  >('/api/small_molecules/motif_scaffolding/stream')

  useEffect(() => {
    if (job.data?.scaffolds?.length) setSelectedIdx(0)
  }, [job.data])

  const canRun =
    !job.isPending &&
    motifPdb.trim().length > 0 &&
    targetChain.trim() &&
    experiment.trim() &&
    runName.trim() &&
    lenMin <= lenMax

  const runJob = () =>
    job.start({
      motif_pdb: motifPdb,
      target_chain: targetChain,
      scaffold_length_min: lenMin,
      scaffold_length_max: lenMax,
      num_samples: numSamples,
      optimize_mpnn: optimizeMpnn,
      validate_esmfold: validateEsmfold,
      mlflow_experiment: experiment,
      mlflow_run_name: runName,
    })

  const selected = useMemo<MotifScaffold | null>(() => {
    if (!job.data?.scaffolds?.length) return null
    return job.data.scaffolds[selectedIdx] ?? null
  }, [job.data, selectedIdx])

  const tableColumns = useMemo<ColumnDef<MotifScaffold, unknown>[]>(
    () => [
      { id: 'sample_id', header: 'Sample', accessorKey: 'sample_id' },
      {
        id: 'sequence',
        header: 'Sequence',
        cell: (ctx) => {
          const s = ctx.row.original.mpnn_sequence ?? ctx.row.original.sequence
          return s.length > 50 ? s.slice(0, 50) + '…' : s
        },
        meta: {
          thClass: 'min-w-[260px]',
          tdClass: 'whitespace-normal break-all font-mono text-[10px]',
        },
      },
      {
        id: 'rewards',
        header: 'Reward',
        accessorFn: (r) => r.rewards.toFixed(4),
      },
      {
        id: 'esmfold_validated',
        header: 'ESMFold',
        cell: (ctx) =>
          ctx.row.original.esmfold_validated ? (
            <span className="text-success">OK</span>
          ) : (
            <span className="text-muted-foreground">—</span>
          ),
      },
    ],
    [],
  )

  const mlflowUrl = job.data
    ? `${window.location.protocol}//${window.location.host.replace(/-\d+\.aws\.databricksapps\.com$/, '')}/ml/experiments/${job.data.experiment_id}/runs/${job.data.run_id}`
    : null

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Transplant a Motif into New Scaffolds</h3>
        <p className="text-xs text-muted-foreground">
          Generate stable protein scaffolds that preserve a functional motif (active site,
          binding loop, etc.) using Proteina-Complexa-AME. Optionally refine each scaffold's
          sequence with ProteinMPNN and validate folding with ESMFold.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(360px,460px)_1fr]">
        {/* Left form */}
        <div className="space-y-3">
          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Motif PDB (ATOM + optional HETATM)
            </span>
            <textarea
              rows={12}
              value={motifPdb}
              onChange={(e) => setMotifPdb(e.target.value)}
              className="w-full rounded-md border border-border bg-background p-3 font-mono text-[10px] leading-tight"
            />
          </label>

          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Motif chain
            </span>
            <input
              value={targetChain}
              onChange={(e) => setTargetChain(e.target.value)}
              className="w-24 rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
          </label>

          <div className="grid grid-cols-2 gap-3">
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Min scaffold length
              </span>
              <input
                type="number"
                min={20}
                max={200}
                value={lenMin}
                onChange={(e) => setLenMin(parseInt(e.target.value || '20'))}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Max scaffold length
              </span>
              <input
                type="number"
                min={20}
                max={300}
                value={lenMax}
                onChange={(e) => setLenMax(parseInt(e.target.value || '300'))}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
          </div>

          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Number of scaffolds
            </span>
            <input
              type="range"
              min={1}
              max={10}
              step={1}
              value={numSamples}
              onChange={(e) => setNumSamples(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="mt-1 text-muted-foreground">{numSamples}</div>
          </label>

          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={optimizeMpnn}
              onChange={(e) => setOptimizeMpnn(e.target.checked)}
            />
            <span>Optimise sequence with ProteinMPNN</span>
          </label>
          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={validateEsmfold}
              onChange={(e) => setValidateEsmfold(e.target.checked)}
            />
            <span>Validate each scaffold with ESMFold</span>
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
              onClick={runJob}
              disabled={!canRun}
              className="flex-1 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
            >
              {job.isPending ? 'Generating…' : 'Generate Scaffolds'}
            </button>
            <button
              onClick={() => job.reset()}
              disabled={!job.data && !job.error}
              className="rounded-md border border-border px-4 py-2 text-sm hover:bg-accent disabled:opacity-50"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Right viewer + results */}
        <div className="space-y-3">
          {job.isPending && (
            <RealtimeProgress
              title={`Generating ${numSamples} scaffold${numSamples > 1 ? 's' : ''}`}
              pct={job.progress?.pct ?? 0}
              msg={job.progress?.msg ?? 'Starting…'}
              stages={[
                { label: 'Generating scaffolds (Proteina-Complexa-AME)', pctEnd: 35 },
                { label: 'Optimising sequences (ProteinMPNN)', pctEnd: 60 },
                { label: 'Validating each scaffold (ESMFold)', pctEnd: 90 },
                { label: 'Building viewers + logging', pctEnd: 100 },
              ]}
            />
          )}

          {job.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
              {String(job.error)}
            </div>
          )}

          {job.data?.warnings?.map((w, i) => (
            <div
              key={i}
              className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-200"
            >
              {w}
            </div>
          ))}

          {job.data && job.data.scaffolds.length === 0 && (
            <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
              Proteina-Complexa-AME returned no scaffolds.
            </div>
          )}

          {job.data && job.data.scaffolds.length > 0 && (
            <>
              <div className="flex flex-wrap items-end gap-3">
                <label className="block text-xs">
                  <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                    Scaffold
                  </span>
                  <select
                    value={selectedIdx}
                    onChange={(e) => setSelectedIdx(parseInt(e.target.value))}
                    className="rounded-md border border-border bg-background px-3 py-2 text-sm"
                  >
                    {job.data.scaffolds.map((s, i) => (
                      <option key={i} value={i}>
                        Scaffold {s.sample_id || i + 1} — Reward {s.rewards.toFixed(4)}
                        {s.esmfold_validated ? '' : '  (ESMFold n/a)'}
                      </option>
                    ))}
                  </select>
                </label>

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

              <MolstarViewer viewerHtml={selected?.viewer_html ?? null} height={520} />

              {selected && (
                <div>
                  <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                    {selected.mpnn_sequence
                      ? 'Sequence (ProteinMPNN-optimised)'
                      : 'Sequence'}
                  </div>
                  <pre className="overflow-x-auto rounded-md border border-border bg-muted/30 p-3 font-mono text-[11px]">
                    {selected.mpnn_sequence ?? selected.sequence}
                  </pre>
                </div>
              )}

              <details className="rounded-md border border-border">
                <summary className="cursor-pointer px-4 py-2 text-sm">All scaffolds</summary>
                <div className="p-3">
                  <DataTable columns={tableColumns} data={job.data.scaffolds} />
                </div>
              </details>
            </>
          )}

          {!job.isPending && !job.data && !job.error && (
            <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
              Paste a motif PDB (with the motif residues on the specified chain) and run
              Generate Scaffolds. The pipeline keeps the motif geometry while sampling new
              surrounding scaffolds; the viewer overlays the original motif with each
              generated scaffold.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

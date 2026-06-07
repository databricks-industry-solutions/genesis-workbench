import { useEffect, useMemo, useState } from 'react'
import type { ColumnDef } from '@tanstack/react-table'

import { ClipboardPaste } from '@/components/ClipboardPaste'
import { DataTable } from '@/components/DataTable'
import { MolstarViewer } from '@/components/MolstarViewer'
import { RealtimeProgress } from '@/components/RealtimeProgress'
import { useSseMutation } from '@/hooks/useSseMutation'
import type { LigandBinderDesign, LigandBinderDesignResponse } from '@/types/api'
import { cn } from '@/lib/utils'

type InputMode = 'smiles' | 'pdb'

const EXAMPLE_SMILES = 'COc(cc1)ccc1C#N'
const EXAMPLE_LIGAND_PDB = `HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG A   1       1.394   0.000   0.000  1.00  0.00           C
HETATM    3  C3  LIG A   1       2.091   1.209   0.000  1.00  0.00           C
HETATM    4  C4  LIG A   1       1.394   2.418   0.000  1.00  0.00           C
HETATM    5  C5  LIG A   1       0.000   2.418   0.000  1.00  0.00           C
HETATM    6  C6  LIG A   1      -0.697   1.209   0.000  1.00  0.00           C
HETATM    7  O1  LIG A   1       3.461   1.209   0.000  1.00  0.00           O
HETATM    8  C7  LIG A   1       4.158   2.418   0.000  1.00  0.00           C
HETATM    9  C8  LIG A   1      -2.115   1.209   0.000  1.00  0.00           C
HETATM   10  N1  LIG A   1      -3.277   1.209   0.000  1.00  0.00           N
END
`

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

type ViewChoice =
  | 'ca_backbone'
  | 'esmfold'
  | 'ca_plus_dock'
  | 'esmfold_plus_dock'

export function LigandBinderDesignTab() {
  const [inputMode, setInputMode] = useState<InputMode>('smiles')
  const [smiles, setSmiles] = useState(EXAMPLE_SMILES)
  const [ligandPdb, setLigandPdb] = useState(EXAMPLE_LIGAND_PDB)
  const [lenMin, setLenMin] = useState(50)
  const [lenMax, setLenMax] = useState(80)
  const [numSamples, setNumSamples] = useState(2)
  const [validateEsmfold, setValidateEsmfold] = useState(true)
  const [validateDiffdock, setValidateDiffdock] = useState(true)
  const [experiment, setExperiment] = useState('gwb_ligand_binder_design')
  const [runName, setRunName] = useState(`ligand_binder_${ts()}`)

  const [selectedIdx, setSelectedIdx] = useState(0)
  const [viewChoice, setViewChoice] = useState<ViewChoice>('esmfold_plus_dock')

  const design = useSseMutation<
    {
      ligand_pdb?: string | null
      ligand_smiles?: string | null
      binder_length_min: number
      binder_length_max: number
      num_samples: number
      validate_esmfold: boolean
      validate_diffdock: boolean
      mlflow_experiment: string
      mlflow_run_name: string
    },
    LigandBinderDesignResponse
  >('/api/small_molecule/ligand_binder_design/stream')

  useEffect(() => {
    if (design.data?.designs?.length) setSelectedIdx(0)
  }, [design.data])

  const canRun =
    !design.isPending &&
    experiment.trim() &&
    runName.trim() &&
    lenMin <= lenMax &&
    (inputMode === 'smiles' ? smiles.trim().length > 0 : ligandPdb.trim().length > 0) &&
    // DiffDock validation needs a SMILES regardless of input mode.
    (!validateDiffdock || smiles.trim().length > 0)

  const runDesign = () =>
    design.start({
      ligand_pdb: inputMode === 'pdb' ? ligandPdb : null,
      ligand_smiles: smiles || null,
      binder_length_min: lenMin,
      binder_length_max: lenMax,
      num_samples: numSamples,
      validate_esmfold: validateEsmfold,
      validate_diffdock: validateDiffdock,
      mlflow_experiment: experiment,
      mlflow_run_name: runName,
    })

  const selectedDesign = useMemo<LigandBinderDesign | null>(() => {
    if (!design.data?.designs?.length) return null
    return design.data.designs[selectedIdx] ?? null
  }, [design.data, selectedIdx])

  const availableViews = useMemo<ViewChoice[]>(() => {
    if (!selectedDesign) return []
    const v: ViewChoice[] = []
    if (selectedDesign.viewer_html_ca_backbone) v.push('ca_backbone')
    if (selectedDesign.viewer_html_esmfold) v.push('esmfold')
    if (selectedDesign.viewer_html_ca_plus_dock) v.push('ca_plus_dock')
    if (selectedDesign.viewer_html_esmfold_plus_dock) v.push('esmfold_plus_dock')
    return v
  }, [selectedDesign])

  // When the selected design changes, pick the richest available view.
  useEffect(() => {
    if (!availableViews.length) return
    if (availableViews.includes(viewChoice)) return
    const preferred: ViewChoice[] = [
      'esmfold_plus_dock',
      'ca_plus_dock',
      'esmfold',
      'ca_backbone',
    ]
    setViewChoice(preferred.find((p) => availableViews.includes(p)) ?? availableViews[0])
  }, [availableViews, viewChoice])

  const viewerHtml = useMemo<string | null>(() => {
    if (!selectedDesign) return null
    switch (viewChoice) {
      case 'esmfold_plus_dock':
        return selectedDesign.viewer_html_esmfold_plus_dock
      case 'ca_plus_dock':
        return selectedDesign.viewer_html_ca_plus_dock
      case 'esmfold':
        return selectedDesign.viewer_html_esmfold
      case 'ca_backbone':
      default:
        return selectedDesign.viewer_html_ca_backbone
    }
  }, [selectedDesign, viewChoice])

  const tableColumns = useMemo<ColumnDef<LigandBinderDesign, unknown>[]>(
    () => [
      { id: 'sample_id', header: 'Sample', accessorKey: 'sample_id' },
      {
        id: 'sequence',
        header: 'Sequence',
        cell: (ctx) => {
          const s = ctx.row.original.sequence
          return s.length > 50 ? s.slice(0, 50) + '…' : s
        },
        meta: {
          thClass: 'min-w-[260px]',
          tdClass: 'whitespace-normal break-all font-mono text-[10px]',
        },
      },
      { id: 'rewards', header: 'Reward', accessorFn: (r) => r.rewards.toFixed(4) },
      {
        id: 'dock_confidence',
        header: 'Dock conf.',
        cell: (ctx) =>
          ctx.row.original.dock_confidence != null
            ? ctx.row.original.dock_confidence.toFixed(4)
            : <span className="text-muted-foreground">—</span>,
      },
    ],
    [],
  )

  const mlflowUrl = design.data
    ? `${window.location.protocol}//${window.location.host.replace(/-\d+\.aws\.databricksapps\.com$/, '')}/ml/experiments/${design.data.experiment_id}/runs/${design.data.run_id}`
    : null

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Design Proteins for a Small Molecule</h3>
        <p className="text-xs text-muted-foreground">
          Generate protein binders that bind a given small-molecule ligand. SMILES inputs are
          embedded in 3D via RDKit (ETKDGv3 + MMFF94) before Proteina-Complexa-Ligand. Optional
          ESMFold refines each design to a full-atom structure; optional DiffDock places the
          ligand back into the designed pocket so you can see the proposed binding pose.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(360px,460px)_1fr]">
        {/* Left form */}
        <div className="space-y-3">
          <div className="flex gap-1">
            {(['smiles', 'pdb'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setInputMode(m)}
                className={cn(
                  'rounded-md border px-3 py-2 text-xs transition-colors',
                  inputMode === m
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border text-muted-foreground hover:bg-accent',
                )}
              >
                {m === 'smiles' ? 'SMILES' : 'Ligand PDB'}
              </button>
            ))}
          </div>

          {inputMode === 'smiles' ? (
            <label className="block text-xs">
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="block uppercase tracking-wide text-muted-foreground">
                  Ligand SMILES
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
          ) : (
            <>
              <label className="block text-xs">
                <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                  Ligand PDB (HETATM)
                </span>
                <textarea
                  rows={10}
                  value={ligandPdb}
                  onChange={(e) => setLigandPdb(e.target.value)}
                  className="w-full rounded-md border border-border bg-background p-3 font-mono text-[10px] leading-tight"
                />
              </label>
              <label className="block text-xs">
                <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                  SMILES (for DiffDock validation)
                </span>
                <input
                  value={smiles}
                  onChange={(e) => setSmiles(e.target.value)}
                  placeholder="COc(cc1)ccc1C#N"
                  className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
                />
              </label>
            </>
          )}

          <div className="grid grid-cols-2 gap-3">
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Min protein length
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
                Max protein length
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
              Number of designs
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
              checked={validateEsmfold}
              onChange={(e) => setValidateEsmfold(e.target.checked)}
            />
            <span>Validate each design with ESMFold</span>
          </label>
          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={validateDiffdock}
              onChange={(e) => setValidateDiffdock(e.target.checked)}
            />
            <span>Validate ligand binding with DiffDock</span>
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
              onClick={runDesign}
              disabled={!canRun}
              className="flex-1 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
            >
              {design.isPending ? 'Designing…' : 'Design Ligand Binders'}
            </button>
            <button
              onClick={() => design.reset()}
              disabled={!design.data && !design.error}
              className="rounded-md border border-border px-4 py-2 text-sm hover:bg-accent disabled:opacity-50"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Right viewer + results */}
        <div className="space-y-3">
          {design.isPending && (
            <RealtimeProgress
              title={`Generating ${numSamples} binder${numSamples > 1 ? 's' : ''}`}
              pct={design.progress?.pct ?? 0}
              msg={design.progress?.msg ?? 'Starting…'}
              stages={[
                { label: 'SMILES → 3D coordinates (RDKit)', pctEnd: 15 },
                { label: 'Generating protein binders (Proteina-Complexa-Ligand)', pctEnd: 35 },
                { label: 'Validating designs (ESMFold)', pctEnd: 60 },
                { label: 'Docking ligand back to designs (DiffDock)', pctEnd: 90 },
                { label: 'Building viewers + logging', pctEnd: 100 },
              ]}
            />
          )}

          {design.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
              {String(design.error)}
            </div>
          )}

          {design.data?.warnings?.map((w, i) => (
            <div
              key={i}
              className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-200"
            >
              {w}
            </div>
          ))}

          {design.data && design.data.designs.length === 0 && (
            <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
              Proteina-Complexa-Ligand returned no designs.
            </div>
          )}

          {design.data && design.data.designs.length > 0 && (
            <>
              <div className="flex flex-wrap items-end gap-3">
                <label className="block text-xs">
                  <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                    Design
                  </span>
                  <select
                    value={selectedIdx}
                    onChange={(e) => setSelectedIdx(parseInt(e.target.value))}
                    className="rounded-md border border-border bg-background px-3 py-2 text-sm"
                  >
                    {design.data.designs.map((d, i) => (
                      <option key={i} value={i}>
                        Design {d.sample_id || i + 1} — Reward {d.rewards.toFixed(4)}
                        {d.dock_confidence != null ? ` · Dock ${d.dock_confidence.toFixed(3)}` : ''}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="block text-xs">
                  <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                    View
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {(
                      [
                        ['esmfold_plus_dock', 'Full + docked ligand'],
                        ['ca_plus_dock', 'CA + docked ligand'],
                        ['esmfold', 'Full protein'],
                        ['ca_backbone', 'CA backbone'],
                      ] as const
                    )
                      .filter(([id]) => availableViews.includes(id))
                      .map(([id, label]) => (
                        <button
                          key={id}
                          type="button"
                          onClick={() => setViewChoice(id)}
                          className={cn(
                            'rounded-md border px-3 py-2 text-xs transition-colors',
                            viewChoice === id
                              ? 'border-primary bg-primary/10 text-primary'
                              : 'border-border text-muted-foreground hover:bg-accent',
                          )}
                        >
                          {label}
                        </button>
                      ))}
                  </div>
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

              <MolstarViewer viewerHtml={viewerHtml} height={520} />

              {selectedDesign && (
                <div>
                  <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                    Designed sequence
                  </div>
                  <pre className="overflow-x-auto rounded-md border border-border bg-muted/30 p-3 font-mono text-[11px]">
                    {selectedDesign.sequence}
                  </pre>
                </div>
              )}

              <details className="rounded-md border border-border">
                <summary className="cursor-pointer px-4 py-2 text-sm">All designs</summary>
                <div className="p-3">
                  <DataTable columns={tableColumns} data={design.data.designs} />
                </div>
              </details>
            </>
          )}

          {!design.isPending && !design.data && !design.error && (
            <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
              Configure a ligand and run Design Ligand Binders. The pipeline generates ranked
              protein binders, optionally folds each one with ESMFold, then re-docks the ligand
              with DiffDock to show the predicted binding pose.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

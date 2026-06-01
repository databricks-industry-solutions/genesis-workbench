import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'

import { api } from '@/api/client'
import { DataTable } from '@/components/DataTable'
import { MolstarViewer } from '@/components/MolstarViewer'
import { RealtimeProgress } from '@/components/RealtimeProgress'
import { useSseMutation } from '@/hooks/useSseMutation'
import type { BinderDesign, BinderDesignResponse } from '@/types/api'
import { cn } from '@/lib/utils'

type InputMode = 'sequence' | 'pdb'
type ViewMode = 'with_target' | 'binder_only'

const EXAMPLE_SEQUENCE =
  'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDAATKTFTVTE'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

export function ProteinBinderDesignTab() {
  // Reuse the docking example endpoint for the default PDB — it serves a
  // 50-residue chain-A slice of 6agt that's also a sensible binder target.
  const example = useQuery({
    queryKey: ['small_molecule', 'diffdock', 'example'],
    queryFn: api.diffdockExample,
    staleTime: Infinity,
  })

  const [inputMode, setInputMode] = useState<InputMode>('sequence')
  const [sequence, setSequence] = useState(EXAMPLE_SEQUENCE)
  const [pdb, setPdb] = useState('')
  const [targetChain, setTargetChain] = useState('A')
  const [hotspots, setHotspots] = useState('')
  const [lenMin, setLenMin] = useState(50)
  const [lenMax, setLenMax] = useState(80)
  const [numSamples, setNumSamples] = useState(2)
  const [validateEsmfold, setValidateEsmfold] = useState(true)
  const [experiment, setExperiment] = useState('gwb_binder_design')
  const [runName, setRunName] = useState(`binder_design_${ts()}`)

  const [selectedIdx, setSelectedIdx] = useState(0)
  const [viewMode, setViewMode] = useState<ViewMode>('with_target')

  // Seed the PDB textarea once the example payload lands. Don't overwrite
  // anything the user has already typed.
  useEffect(() => {
    if (example.data?.pdb) setPdb((cur) => cur || example.data!.pdb)
  }, [example.data])

  const design = useSseMutation<
    {
      target_pdb?: string | null
      target_sequence?: string | null
      target_chain: string
      hotspot_residues: string
      binder_length_min: number
      binder_length_max: number
      num_samples: number
      validate_esmfold: boolean
      mlflow_experiment: string
      mlflow_run_name: string
    },
    BinderDesignResponse
  >('/api/large_molecule/binder_design/stream')

  useEffect(() => {
    if (design.data?.designs?.length) setSelectedIdx(0)
  }, [design.data])

  const canRun =
    !design.isPending &&
    targetChain.trim().length > 0 &&
    experiment.trim() &&
    runName.trim() &&
    lenMin <= lenMax &&
    (inputMode === 'sequence' ? sequence.trim().length > 0 : pdb.trim().length > 0)

  const runDesign = () =>
    design.start({
      target_pdb: inputMode === 'pdb' ? pdb : null,
      target_sequence: inputMode === 'sequence' ? sequence : null,
      target_chain: targetChain,
      hotspot_residues: hotspots,
      binder_length_min: lenMin,
      binder_length_max: lenMax,
      num_samples: numSamples,
      validate_esmfold: validateEsmfold,
      mlflow_experiment: experiment,
      mlflow_run_name: runName,
    })

  const selectedDesign = useMemo<BinderDesign | null>(() => {
    if (!design.data?.designs?.length) return null
    return design.data.designs[selectedIdx] ?? null
  }, [design.data, selectedIdx])

  const viewerHtml = useMemo<string | null>(() => {
    if (!selectedDesign) return design.data?.target_only_viewer_html ?? null
    if (viewMode === 'with_target') {
      return selectedDesign.viewer_html_with_target ?? selectedDesign.viewer_html_binder_only ?? null
    }
    return selectedDesign.viewer_html_binder_only ?? null
  }, [selectedDesign, viewMode, design.data])

  const tableColumns = useMemo<ColumnDef<BinderDesign, unknown>[]>(
    () => [
      { id: 'sample_id', header: 'Sample', accessorKey: 'sample_id' },
      {
        id: 'sequence',
        header: 'Sequence',
        cell: (ctx) => {
          const s = ctx.row.original.sequence
          return s.length > 50 ? s.slice(0, 50) + '…' : s
        },
        meta: { thClass: 'min-w-[260px]', tdClass: 'whitespace-normal break-all font-mono text-[10px]' },
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

  const mlflowUrl = design.data
    ? `${window.location.protocol}//${window.location.host.replace(/-\d+\.aws\.databricksapps\.com$/, '')}/ml/experiments/${design.data.experiment_id}/runs/${design.data.run_id}`
    : null

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Design Binders Against a Target</h3>
        <p className="text-xs text-muted-foreground">
          Design novel protein binders against a target. The pipeline takes a target PDB (or folds
          a target sequence first via ESMFold), generates binder candidates with Proteina-Complexa
          conditioned on the target + optional hotspot residues, then optionally re-folds each
          binder with ESMFold to verify the design folds.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(360px,460px)_1fr]">
        {/* Left form */}
        <div className="space-y-3">
          <div className="flex gap-1">
            {(['sequence', 'pdb'] as const).map((m) => (
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
                {m === 'sequence' ? 'Protein sequence' : 'Target PDB'}
              </button>
            ))}
          </div>

          {inputMode === 'sequence' ? (
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Target sequence
              </span>
              <textarea
                rows={4}
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder="MTYK…"
                className="w-full rounded-md border border-border bg-background p-3 font-mono text-xs"
              />
              <span className="mt-1 block text-[10px] text-muted-foreground">
                Will be folded by ESMFold to produce the target PDB.
              </span>
            </label>
          ) : (
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Target PDB
              </span>
              <textarea
                rows={10}
                value={pdb}
                onChange={(e) => setPdb(e.target.value)}
                placeholder="ATOM…"
                className="w-full rounded-md border border-border bg-background p-3 font-mono text-[10px] leading-tight"
              />
            </label>
          )}

          <div className="grid grid-cols-2 gap-3">
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Target chain
              </span>
              <input
                value={targetChain}
                onChange={(e) => setTargetChain(e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Hotspot residues
              </span>
              <input
                value={hotspots}
                onChange={(e) => setHotspots(e.target.value)}
                placeholder="e.g. 10,20,30"
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Min binder length
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
                Max binder length
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
              {design.isPending ? 'Designing…' : 'Design Binders'}
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
                {
                  label: 'Folding target sequence (ESMFold)',
                  pctEnd: 15,
                },
                { label: 'Generating binders (Proteina-Complexa)', pctEnd: 50 },
                { label: 'Validating each design (ESMFold)', pctEnd: 95 },
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
              Proteina-Complexa returned no binder designs.
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
                        {d.esmfold_validated ? '' : '  (ESMFold n/a)'}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="block text-xs">
                  <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                    View
                  </span>
                  <div className="flex gap-1">
                    {(['with_target', 'binder_only'] as const).map((m) => (
                      <button
                        key={m}
                        type="button"
                        onClick={() => setViewMode(m)}
                        className={cn(
                          'rounded-md border px-3 py-2 text-xs transition-colors',
                          viewMode === m
                            ? 'border-primary bg-primary/10 text-primary'
                            : 'border-border text-muted-foreground hover:bg-accent',
                        )}
                      >
                        {m === 'with_target' ? 'Binder + Target' : 'Binder only'}
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
              Configure the form and run Design Binders. Proteina-Complexa will return ranked
              binders; if ESMFold validation is enabled each candidate is re-folded so the
              viewer can show a full-atom structure instead of the CA-only backbone.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

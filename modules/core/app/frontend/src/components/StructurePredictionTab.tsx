import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'

import { api } from '@/api/client'
import { AlphaFoldSearchResults } from '@/components/AlphaFoldPanel'
import { MolstarViewer } from '@/components/MolstarViewer'
import { SequenceSourceControls } from '@/components/SequenceSourceControls'
import { WorkflowProgress } from '@/components/WorkflowProgress'
import { cn } from '@/lib/utils'

const MODELS = ['ESMFold', 'Boltz', 'AlphaFold2'] as const
type Model = (typeof MODELS)[number]

const DEFAULT_SEQUENCE = 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDAATKTFTVTE'
const BOLTZ_DEFAULT =
  'QVQLVESGGGLVQAGGSLRLACIASGRTFHSYVMAWFRQAPGKEREFVAAISWSSTPTYYGESVKGRFTISRDNAKNTVYLQMNRLKPEDTAVYFCAADRGESYYYTRPTEYEFWGQGTQVTVSS'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${(d.getMonth() + 1).toString().padStart(2, '0')}${d
    .getDate()
    .toString()
    .padStart(2, '0')}_${d.getHours().toString().padStart(2, '0')}${d
    .getMinutes()
    .toString()
    .padStart(2, '0')}`
}

export function StructurePredictionTab() {
  const qc = useQueryClient()
  const [model, setModel] = useState<Model>('ESMFold')
  const [sequence, setSequence] = useState<string>(DEFAULT_SEQUENCE)
  const [expName, setExpName] = useState('structure_prediction')
  const [runName, setRunName] = useState(`esmfold_${ts()}`)
  const [viewerHtml, setViewerHtml] = useState<string | null>(null)

  const isAsync = model === 'AlphaFold2'

  // ESMFold / Boltz fold synchronously (→ live viewer + MLflow log);
  // AlphaFold2 starts an async job (→ Search Past Runs on the right).
  const predict = useMutation({
    mutationFn: async () => {
      if (model === 'ESMFold') return api.esmfold(sequence, expName, runName)
      if (model === 'Boltz') return api.boltz(sequence, expName, runName)
      return api.alphafoldStart({ sequence, experiment_name: expName, run_name: runName })
    },
    onSuccess: (data) => {
      if ('viewer_html' in data) setViewerHtml(data.viewer_html)
      else qc.invalidateQueries({ queryKey: ['alphafold', 'search'] })
    },
  })

  const handleModel = (m: Model) => {
    setModel(m)
    setViewerHtml(null)
    predict.reset()
    setRunName(`${m === 'AlphaFold2' ? 'alphafold' : m.toLowerCase()}_${ts()}`)
    if (m === 'Boltz') setSequence(BOLTZ_DEFAULT)
    if (m === 'ESMFold') setSequence(DEFAULT_SEQUENCE)
  }

  const submitLabel = isAsync ? 'Start Job' : model === 'ESMFold' ? 'Predict & View' : 'Predict'
  const disabled =
    predict.isPending ||
    sequence.trim().length === 0 ||
    expName.trim().length === 0 ||
    runName.trim().length === 0

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold">Predict 3D Structure from Sequence</h3>

      <div className="flex flex-wrap gap-2">
        {MODELS.map((m) => (
          <button
            key={m}
            onClick={() => handleModel(m)}
            className={cn(
              'rounded-full border px-3 py-1 text-xs transition-colors',
              m === model
                ? 'border-primary bg-primary/10 text-primary'
                : 'border-border text-muted-foreground hover:bg-accent',
            )}
          >
            {m}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(340px,440px)_1fr]">
        {/* Left: shared input panel for all three models. */}
        <div className="space-y-3">
          <div className="text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Protein sequence
            </span>
            <SequenceSourceControls onSequence={setSequence} className="mb-1.5" />
            <textarea
              rows={8}
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
              placeholder="Paste a protein sequence (single-letter amino acid code), resolve from a gene, or paste from the Clipboard"
              className="w-full rounded-md border border-border bg-background p-3 font-mono text-xs"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                MLflow Experiment
              </span>
              <input
                value={expName}
                onChange={(e) => setExpName(e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
            <label className="block text-xs">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Run Name
              </span>
              <input
                value={runName}
                onChange={(e) => setRunName(e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => predict.mutate()}
              disabled={disabled}
              className="flex-1 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
            >
              {predict.isPending ? (isAsync ? 'Starting…' : 'Predicting…') : submitLabel}
            </button>
            {!isAsync && (
              <button
                onClick={() => {
                  setViewerHtml(null)
                  predict.reset()
                }}
                disabled={!viewerHtml && !predict.isError && !predict.isPending}
                className="rounded-md border border-border px-4 py-2 text-sm hover:bg-accent disabled:opacity-50"
              >
                Clear
              </button>
            )}
          </div>

          {predict.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
              {String(predict.error)}
            </div>
          )}
          {/* Async job started */}
          {isAsync && predict.data && 'job_run_id' in predict.data && (
            <div className="rounded-md border border-success/40 bg-success/10 p-3 text-sm">
              Job started — run id <code>{predict.data.job_run_id}</code>. Find it in Search Past
              Runs →
            </div>
          )}
          {/* Sync fold logged to MLflow */}
          {!isAsync && predict.data && 'run_id' in predict.data && predict.data.run_id && (
            <div className="text-[11px] text-muted-foreground">
              Logged to MLflow run{' '}
              {predict.data.run_url ? (
                <a
                  href={predict.data.run_url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary hover:underline"
                >
                  {runName}
                </a>
              ) : (
                <code>{runName}</code>
              )}
              .
            </div>
          )}
        </div>

        {/* Right: viewer (ESMFold/Boltz) or AlphaFold search + results. */}
        <div className="space-y-3">
          {isAsync ? (
            <AlphaFoldSearchResults />
          ) : (
            <>
              <WorkflowProgress
                active={predict.isPending}
                title={`${model} prediction`}
                stages={
                  model === 'ESMFold'
                    ? [{ label: 'Predicting structure (ESMFold)', estSeconds: 12 }]
                    : [
                        { label: 'Submitting input to Boltz', estSeconds: 2 },
                        { label: 'Predicting structure', estSeconds: 60 },
                      ]
                }
                note={
                  model === 'Boltz'
                    ? 'Boltz cold-start can exceed the proxy timeout (~60s) — pre-warm via Settings → Endpoint Management.'
                    : 'Endpoint cold-start can add 20–30s on the first call.'
                }
              />

              {viewerHtml ? (
                <MolstarViewer
                  viewerHtml={viewerHtml}
                  title={`${model} structure viewer`}
                  height={540}
                />
              ) : (
                !predict.isPending &&
                !predict.error && (
                  <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
                    Set up the sequence on the left and hit {submitLabel} to render the predicted
                    structure here.
                  </div>
                )
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

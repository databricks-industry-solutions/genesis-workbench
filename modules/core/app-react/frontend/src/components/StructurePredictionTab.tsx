import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'

import { api } from '@/api/client'
import { AlphaFoldPanel } from '@/components/AlphaFoldPanel'
import { MolstarViewer } from '@/components/MolstarViewer'
import { WorkflowProgress } from '@/components/WorkflowProgress'
import { cn } from '@/lib/utils'

const MODELS = ['ESMFold', 'Boltz', 'AlphaFold2'] as const
type Model = (typeof MODELS)[number]

const DEFAULT_SEQUENCE = 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDAATKTFTVTE'
const BOLTZ_DEFAULT =
  'QVQLVESGGGLVQAGGSLRLACIASGRTFHSYVMAWFRQAPGKEREFVAAISWSSTPTYYGESVKGRFTISRDNAKNTVYLQMNRLKPEDTAVYFCAADRGESYYYTRPTEYEFWGQGTQVTVSS'

export function StructurePredictionTab() {
  const [model, setModel] = useState<Model>('ESMFold')
  const [sequence, setSequence] = useState<string>(DEFAULT_SEQUENCE)
  const [viewerHtml, setViewerHtml] = useState<string | null>(null)

  const predict = useMutation({
    mutationFn: async () => {
      if (model === 'ESMFold') return api.esmfold(sequence)
      if (model === 'Boltz') return api.boltz(sequence)
      throw new Error('Use the AlphaFold2 panel below for async jobs.')
    },
    onSuccess: (data) => setViewerHtml(data.viewer_html),
  })

  const handleModel = (m: Model) => {
    setModel(m)
    setViewerHtml(null)
    if (m === 'Boltz') setSequence(BOLTZ_DEFAULT)
    if (m === 'ESMFold') setSequence(DEFAULT_SEQUENCE)
  }

  const isAsync = model === 'AlphaFold2'
  const disabled = isAsync || predict.isPending || sequence.trim().length === 0
  const buttonLabel = model === 'ESMFold' ? 'View' : 'Predict'

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold">Predict Protein Structure</h3>

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

      {!isAsync && (
        <>
          <textarea
            rows={4}
            value={sequence}
            onChange={(e) => setSequence(e.target.value)}
            placeholder="Paste a protein sequence (single-letter amino acid code)"
            className="w-full rounded-md border border-border bg-background p-3 font-mono text-xs"
          />

          <div className="flex items-center gap-3">
            <button
              onClick={() => predict.mutate()}
              disabled={disabled}
              className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
            >
              {predict.isPending ? 'Predicting…' : buttonLabel}
            </button>
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
          </div>

          {predict.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
              {String(predict.error)}
            </div>
          )}

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

          {viewerHtml && (
            <MolstarViewer viewerHtml={viewerHtml} title={`${model} structure viewer`} />
          )}
        </>
      )}

      {isAsync && <AlphaFoldPanel />}
    </div>
  )
}

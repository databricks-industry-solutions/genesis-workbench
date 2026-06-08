import { useState } from 'react'

import { MolstarViewer } from '@/components/MolstarViewer'
import { RealtimeProgress } from '@/components/RealtimeProgress'
import { SequenceSourceControls } from '@/components/SequenceSourceControls'
import { useSseMutation } from '@/hooks/useSseMutation'
import { useUserStore } from '@/stores/user'
import type { ProteinDesignResponse } from '@/types/api'

const DEFAULT_SEQUENCE =
  'MAQVKLQESGGGLVQPGGSLRLSCASSVPIFAITVMGWYRQAPGKQRELVAGIKRSGD[TNYADS]VKGRFTISRDDAKNTVFLQMNSLTTEDTAVYYCNAQILSWMGGTDYWGQGTQVTVSSGQAGQ'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

export function ProteinDesignTab() {
  const bootstrap = useUserStore((s) => s.bootstrap)
  const workspaceHost = bootstrap?.env ? new URL(window.location.origin).origin : ''

  const [sequence, setSequence] = useState(DEFAULT_SEQUENCE)
  const [experiment, setExperiment] = useState('gwb_protein_design')
  const [runName, setRunName] = useState(`protein_design_${ts()}`)
  const [nHits, setNHits] = useState(1)

  const design = useSseMutation<
    {
      sequence: string
      experiment_name: string
      run_name: string
      n_rfdiffusion_hits: number
    },
    ProteinDesignResponse
  >('/api/large_molecule/protein_design/stream')

  const runDesign = () =>
    design.start({
      sequence,
      experiment_name: experiment,
      run_name: runName,
      n_rfdiffusion_hits: nHits,
    })

  const bracketsOk = sequence.includes('[') && sequence.includes(']')
  const canRun =
    !design.isPending && bracketsOk && experiment.trim() && runName.trim()

  const mlflowUrl =
    design.data && bootstrap?.env
      ? // Databricks workspace MLflow UI deep link
        `${window.location.protocol}//${window.location.host.replace('-7474660003062283.aws.databricksapps.com', '')}/ml/experiments/${design.data.experiment_id}/runs/${design.data.run_id}`
      : null

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Redesign a Masked Sequence Region</h3>
        <p className="text-xs text-muted-foreground">
          Mask a region with <code>[brackets]</code>. The pipeline folds the original sequence,
          inpaints the masked region with RFDiffusion, then redesigns sequences with ProteinMPNN
          and validates each by re-folding with ESMFold. Logs every step to MLflow.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(320px,420px)_1fr]">
        {/* Left: form */}
        <div className="space-y-3">
          {/* Plain div (not <label>): a <label> proxies dead-area clicks to its first
              labelable descendant — here the SequenceSourceControls picker button — popping it open. */}
          <div className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Input sequence — masked region in [brackets]
            </span>
            <SequenceSourceControls onSequence={setSequence} className="mb-1.5" />
            <textarea
              aria-label="Input sequence — masked region in [brackets]"
              rows={8}
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
              placeholder="MAQV...[TNYADS]...SSGQ"
              className="w-full rounded-md border border-border bg-background p-3 font-mono text-xs"
            />
          </div>
          {!bracketsOk && sequence.trim() && (
            <div className="text-xs text-amber-500">
              Sequence must contain a [bracketed] region to redesign.
            </div>
          )}

          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              RFDiffusion scaffolds
            </span>
            <select
              value={nHits}
              onChange={(e) => setNHits(parseInt(e.target.value))}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            >
              {[1, 2, 3, 4].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
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
              {design.isPending ? 'Generating…' : 'Generate'}
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

        {/* Right: progress + viewer */}
        <div className="space-y-3">
          {design.isPending && (
            <RealtimeProgress
              title={`Designing ${nHits} scaffold${nHits > 1 ? 's' : ''}`}
              pct={design.progress?.pct ?? 0}
              msg={design.progress?.msg ?? 'Starting…'}
              stages={[
                { label: 'Folding original sequence (ESMFold)', pctEnd: 10 },
                { label: `RFDiffusion x ${nHits}`, pctEnd: 50 },
                { label: 'ProteinMPNN sequence design', pctEnd: 70 },
                { label: 'Folding each designed sequence (ESMFold)', pctEnd: 95 },
                { label: 'Aligning designed structures', pctEnd: 100 },
              ]}
            />
          )}

          {design.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
              {String(design.error)}
            </div>
          )}

          {design.data ? (
            <>
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium">
                  {design.data.n_designs} design{design.data.n_designs === 1 ? '' : 's'} (aligned
                  to initial)
                </h4>
                {mlflowUrl && (
                  <a
                    href={mlflowUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-primary hover:underline"
                    title={`workspace_host=${workspaceHost}`}
                  >
                    View MLflow run ↗
                  </a>
                )}
              </div>
              <MolstarViewer viewerHtml={design.data.viewer_html} height={560} />
            </>
          ) : (
            !design.isPending &&
            !design.error && (
              <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
                Submit the form to generate designs. The pipeline folds the original sequence,
                inpaints the masked region with RFDiffusion, redesigns each scaffold with
                ProteinMPNN, and re-folds every candidate with ESMFold.
              </div>
            )
          )}
        </div>
      </div>
    </div>
  )
}

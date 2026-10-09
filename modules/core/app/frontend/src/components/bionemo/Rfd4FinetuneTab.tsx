import { useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { RunSearchSection } from '@/components/RunSearchSection'
import type { DBRunRow } from '@/types/api'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

export function Rfd4FinetuneTab() {
  const [label, setLabel] = useState(`rfd4_ft_${ts()}`)
  const [pretrainCkpt, setPretrainCkpt] = useState('')
  const [trainData, setTrainData] = useState('')
  const [experimentPreset, setExperimentPreset] = useState('tutorial/finetune_test_dataset')
  const [maxEpochs, setMaxEpochs] = useState(5)
  const [stepsPerEpoch, setStepsPerEpoch] = useState(100)
  const [experiment, setExperiment] = useState('gwb_rfd4_finetune')
  const [runName, setRunName] = useState(`rfd4_ft_${ts()}`)
  const [searchToken, setSearchToken] = useState(0)

  const [deployFtId, setDeployFtId] = useState('')

  // Prefill the staged base checkpoint + tutorial preset. Training data stays
  // blank on purpose (blank = rfd4-train's bundled tutorial dataset).
  const defaults = useQuery({ queryKey: ['rfd4', 'defaults'], queryFn: api.rfd4Defaults })
  useEffect(() => {
    if (defaults.data) {
      setPretrainCkpt((v) => v || defaults.data!.pretrain_ckpt)
      setTrainData((v) => v || defaults.data!.train_data)
      setExperimentPreset((v) => v || defaults.data!.experiment_preset)
    }
  }, [defaults.data])

  const weights = useQuery({ queryKey: ['rfd4', 'weights'], queryFn: api.rfd4Weights })

  const start = useMutation({
    mutationFn: () =>
      api.rfd4Finetune({
        finetune_label: label,
        pretrain_ckpt: pretrainCkpt,
        train_data: trainData,
        experiment_preset: experimentPreset,
        max_epochs: maxEpochs,
        steps_per_epoch: stepsPerEpoch,
        experiment_name: experiment,
        run_name: runName,
      }),
    onSuccess: () => setSearchToken((t) => t + 1),
  })

  const deploy = useMutation({
    mutationFn: () => api.rfd4Deploy({ ft_id: deployFtId }),
  })

  const canRun =
    !start.isPending &&
    label.trim() &&
    experimentPreset.trim() &&
    experiment.trim() &&
    runName.trim()

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Fine-tune RFD4-Proteina (PEFT / LoRA)</h3>
        <p className="text-xs text-muted-foreground">
          Fine-tune the NVIDIA×Baker <strong>RFD4-Proteina</strong> flow-matching design model with a
          LoRA adapter on the released flow checkpoint, then deploy the fine-tuned version onto the
          RFD4-Proteina serving endpoint via Express. Runs on a serverless H100. Defaults to the
          bundled tutorial fine-tune dataset — leave the base checkpoint and training data blank to use
          the staged defaults.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(360px,460px)_1fr]">
        {/* Left: form */}
        <div className="space-y-3">
          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Fine-tune label
            </span>
            <input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
          </label>

          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Base checkpoint (UC volume; blank = staged flow checkpoint)
            </span>
            <input
              value={pretrainCkpt}
              onChange={(e) => setPretrainCkpt(e.target.value)}
              placeholder="/Volumes/…/rfd4_proteina/flow_checkpoints/…ema.ckpt"
              className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
            />
          </label>

          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              Training data dir (UC volume; parquet manifest + CIFs)
            </span>
            <input
              value={trainData}
              onChange={(e) => setTrainData(e.target.value)}
              placeholder="/Volumes/…/rfd4_proteina/ft_data"
              className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
            />
          </label>

          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              rfd4-train experiment preset
            </span>
            <input
              value={experimentPreset}
              onChange={(e) => setExperimentPreset(e.target.value)}
              placeholder="tutorial/finetune_test_dataset"
              className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
            />
          </label>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Max epochs</span>
              <input
                type="number"
                min={1}
                max={200}
                value={maxEpochs}
                onChange={(e) => setMaxEpochs(parseInt(e.target.value) || 5)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Steps / epoch</span>
              <input
                type="number"
                min={1}
                max={10000}
                step={10}
                value={stepsPerEpoch}
                onChange={(e) => setStepsPerEpoch(parseInt(e.target.value) || 100)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">MLflow Experiment</span>
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
            {start.isPending ? 'Dispatching…' : 'Fine-tune RFD4-Proteina'}
          </button>

          {start.data && (
            <p className="text-[11px] text-muted-foreground">
              ✓ Job dispatched (run {start.data.job_run_id}).{' '}
              <a href={start.data.run_url} target="_blank" rel="noreferrer" className="text-primary hover:underline">
                View job run ↗
              </a>{' '}
              — track progress in Search Past Runs below.
            </p>
          )}
          {start.error && (
            <p className="text-[11px] text-destructive">{String(start.error)}</p>
          )}
        </div>

        {/* Right: deploy a fine-tuned model + search */}
        <div className="space-y-4">
          <div className="rounded-md border border-border bg-card p-3 text-xs">
            <div className="mb-2 font-medium uppercase tracking-wide text-muted-foreground">
              Deploy a fine-tuned model → RFD4-Proteina endpoint
            </div>
            <p className="mb-2 text-[11px] text-muted-foreground">
              Express-registers the chosen fine-tuned adapter (base flow + LoRA) as a new version of the{' '}
              <code>rfd4_proteina</code> serving endpoint. (Endpoint build takes a while.)
            </p>
            <div className="flex gap-2">
              <select
                value={deployFtId}
                onChange={(e) => setDeployFtId(e.target.value)}
                className="min-w-0 flex-1 rounded-md border border-border bg-background px-3 py-2 text-sm"
              >
                <option value="">Select a fine-tuned model…</option>
                {(weights.data?.weights ?? []).map((w) => (
                  <option key={w.ft_id} value={w.ft_id}>
                    {w.ft_label} ({w.model_type}) — {w.created_datetime ?? ''}
                  </option>
                ))}
              </select>
              <button
                onClick={() => deploy.mutate()}
                disabled={deploy.isPending || !deployFtId}
                className="rounded-md bg-primary px-3 py-2 text-xs font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
              >
                {deploy.isPending ? 'Deploying…' : 'Deploy'}
              </button>
            </div>
            {deploy.data && (
              <p className="mt-2 text-[11px] text-muted-foreground">
                ✓ Deploy dispatched (run {deploy.data.job_run_id}).{' '}
                <a href={deploy.data.run_url} target="_blank" rel="noreferrer" className="text-primary hover:underline">
                  View job run ↗
                </a>
              </p>
            )}
            {deploy.error && <p className="mt-2 text-[11px] text-destructive">{String(deploy.error)}</p>}
          </div>

          <div className="border-t border-border pt-2">
            <h3 className="mb-2 text-sm font-semibold">Search past runs</h3>
            <RunSearchSection
              searchKey={['rfd4', 'finetune', 'search'] as const}
              searchFn={api.rfd4FinetuneSearch}
              detailLabel="Model"
              initialText="rfd4"
              viewableStatuses={['complete']}
              detailColClass="w-48"
              searchToken={searchToken}
              renderDialog={(run: DBRunRow) => (
                <div className="space-y-3 text-sm">
                  <div>
                    <div className="text-xs font-medium text-muted-foreground">Run</div>
                    <div>{run.run_name}</div>
                    <div className="text-xs text-muted-foreground">{run.detail}</div>
                  </div>
                  {run.run_url && (
                    <a href={run.run_url} target="_blank" rel="noreferrer" className="text-primary hover:underline">
                      View in MLflow ↗
                    </a>
                  )}
                  <p className="text-xs text-muted-foreground">
                    When complete, deploy this model from the panel above to serve it on the
                    RFD4-Proteina endpoint.
                  </p>
                </div>
              )}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

import { useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { RunSearchSection } from '@/components/RunSearchSection'
import type { DBRunRow } from '@/types/api'

function ts(): string {
  const d = new Date()
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
}

export function KermtFinetuneTab() {
  const [label, setLabel] = useState(`kermt_ft_${ts()}`)
  const [trainData, setTrainData] = useState('')
  const [valData, setValData] = useState('')
  const [testData, setTestData] = useState('')
  const [targetNames, setTargetNames] = useState('toxicity')
  const [datasetType, setDatasetType] = useState('classification')
  const [epochs, setEpochs] = useState(20)
  const [batchSize, setBatchSize] = useState(16)
  const [ffnHidden, setFfnHidden] = useState(700)
  const [experiment, setExperiment] = useState('gwb_kermt_finetune')
  const [runName, setRunName] = useState(`kermt_ft_${ts()}`)
  const [searchToken, setSearchToken] = useState(0)

  const [deployFtId, setDeployFtId] = useState('')

  // Prefill the TDC ClinTox sample the kermt module stages.
  const defaults = useQuery({ queryKey: ['kermt', 'defaults'], queryFn: api.kermtDefaults })
  useEffect(() => {
    if (defaults.data) {
      setTrainData((v) => v || defaults.data!.train_data)
      setValData((v) => v || defaults.data!.validation_data)
      setTestData((v) => v || defaults.data!.test_data)
    }
  }, [defaults.data])

  const weights = useQuery({ queryKey: ['kermt', 'weights'], queryFn: api.kermtWeights })

  const start = useMutation({
    mutationFn: () =>
      api.kermtFinetune({
        finetune_label: label,
        train_data: trainData,
        validation_data: valData,
        test_data: testData,
        target_names: targetNames,
        dataset_type: datasetType,
        epochs,
        batch_size: batchSize,
        ffn_hidden_size: ffnHidden,
        experiment_name: experiment,
        run_name: runName,
      }),
    onSuccess: () => setSearchToken((t) => t + 1),
  })

  const deploy = useMutation({
    mutationFn: () => api.kermtDeploy({ ft_id: deployFtId }),
  })

  const canRun =
    !start.isPending &&
    label.trim() &&
    trainData.trim() &&
    valData.trim() &&
    testData.trim() &&
    targetNames.trim() &&
    experiment.trim() &&
    runName.trim()

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Fine-tune KERMT for ADMET / Toxicity</h3>
        <p className="text-xs text-muted-foreground">
          Fine-tune NVIDIA-BioNeMo <strong>KERMT</strong> (Kinetic GROVER Multi-Task, a graph
          neural network for small-molecule property prediction) on a SMILES + target dataset, then
          deploy it as a serving endpoint that the ADMET &amp; Safety tab can call side-by-side with
          Chemprop. Defaults to the bundled TDC ClinTox sample.
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

          {(['train', 'val', 'test'] as const).map((k) => {
            const v = k === 'train' ? trainData : k === 'val' ? valData : testData
            const set = k === 'train' ? setTrainData : k === 'val' ? setValData : setTestData
            const lbl = k === 'train' ? 'Train CSV' : k === 'val' ? 'Validation CSV' : 'Test CSV'
            return (
              <label key={k} className="block text-xs">
                <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                  {lbl} (UC volume; columns: smiles, &lt;target&gt;)
                </span>
                <input
                  value={v}
                  onChange={(e) => set(e.target.value)}
                  placeholder="/Volumes/…/kermt/ft_data/clintox_train.csv"
                  className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs"
                />
              </label>
            )
          })}

          <div className="grid grid-cols-2 gap-3 text-xs">
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Target column(s)
              </span>
              <input
                value={targetNames}
                onChange={(e) => setTargetNames(e.target.value)}
                placeholder="toxicity"
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                Task type
              </span>
              <select
                value={datasetType}
                onChange={(e) => setDatasetType(e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              >
                <option value="classification">classification</option>
                <option value="regression">regression</option>
              </select>
            </label>
          </div>

          <div className="grid grid-cols-3 gap-3 text-xs">
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Epochs</span>
              <input
                type="number"
                min={2}
                max={200}
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 20)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">Batch</span>
              <input
                type="number"
                min={1}
                max={64}
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 16)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </label>
            <label className="block">
              <span className="mb-1 block uppercase tracking-wide text-muted-foreground">FFN hidden</span>
              <input
                type="number"
                min={100}
                max={2000}
                step={100}
                value={ffnHidden}
                onChange={(e) => setFfnHidden(parseInt(e.target.value) || 700)}
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
            {start.isPending ? 'Dispatching…' : 'Fine-tune KERMT'}
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
              Deploy a fine-tuned model → ADMET endpoint
            </div>
            <p className="mb-2 text-[11px] text-muted-foreground">
              Registers the chosen fine-tuned KERMT as the <code>kermt_admet</code> serving endpoint
              the ADMET &amp; Safety tab queries. (Endpoint build takes a while.)
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
                    {w.ft_label} ({w.dataset_type}: {w.task_names}) — {w.created_datetime ?? ''}
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
              searchKey={['kermt', 'finetune', 'search'] as const}
              searchFn={api.kermtFinetuneSearch}
              detailLabel="Model"
              initialText="kermt"
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
                    When complete, deploy this model from the panel above to make it servable in the
                    ADMET &amp; Safety tab.
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

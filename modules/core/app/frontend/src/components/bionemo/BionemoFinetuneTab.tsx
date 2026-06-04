// NVIDIA BioNeMo — ESM2 fine-tuning form. Port of the Streamlit finetune tab.
import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'

const TASK_TYPES = ['regression', 'classification']
const PRECISIONS = ['bf16-mixed', 'fp16', 'bf16', 'fp32', 'fp32-mixed', '16-mixed', 'fp16-mixed']

export function BionemoFinetuneTab() {
  const variants = useQuery({ queryKey: ['bionemo', 'variants'], queryFn: api.bionemoVariants })
  const esm2 = variants.data?.esm2 ?? []

  // User's explicit choice (empty until they pick); the effective variant
  // falls back to the first option once the list loads — derived, not stored,
  // so we never setState during render/effect.
  const [esmVariantSel, setEsmVariant] = useState('')
  const esmVariant = esmVariantSel || esm2[0] || ''
  const [trainData, setTrainData] = useState('')
  const [evalData, setEvalData] = useState('')
  const [useLora, setUseLora] = useState(false)
  const [label, setLabel] = useState('')
  const [experiment, setExperiment] = useState('')
  const [taskType, setTaskType] = useState('regression')
  const [numSteps, setNumSteps] = useState(50)
  const [microBatch, setMicroBatch] = useState(2)
  const [precision, setPrecision] = useState('bf16-mixed')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [dropout, setDropout] = useState(0.25)
  const [hiddenSize, setHiddenSize] = useState(256)
  const [targetSize, setTargetSize] = useState(1)
  const [lr, setLr] = useState(5e-3)
  const [lrMult, setLrMult] = useState(1e2)

  const start = useMutation({
    mutationFn: () =>
      api.bionemoFinetune({
        esm_variant: esmVariant,
        train_data: trainData,
        evaluation_data: evalData,
        finetune_label: label,
        experiment_name: experiment,
        should_use_lora: useLora,
        task_type: taskType,
        num_steps: numSteps,
        micro_batch_size: microBatch,
        precision,
        mlp_ft_dropout: dropout,
        mlp_hidden_size: hiddenSize,
        mlp_target_size: targetSize,
        mlp_lr: lr,
        mlp_lr_multiplier: lrMult,
      }),
  })

  const csvOk = (p: string) => p.trim().startsWith('/Volumes') && p.trim().endsWith('.csv')
  const canStart =
    !!esmVariant && csvOk(trainData) && csvOk(evalData) && !!label.trim() && !!experiment.trim()

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Fine-tune NVIDIA's ESM-2 protein language model on your own labeled sequences. Train and
        evaluation data must be CSV files in a UC Volume with a <code>sequence</code> column and a{' '}
        <code>target</code> column.
      </p>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {/* Column 1 — data + identity */}
        <Section title="Data & run">
          <Field label="ESM variant">
            <Select value={esmVariant} onChange={setEsmVariant} options={esm2} />
          </Field>
          <Field label="Train data (UC Volume *.csv)" error={trainData && !csvOk(trainData) ? 'Must be /Volumes/…/*.csv' : ''}>
            <Text value={trainData} onChange={setTrainData} placeholder="/Volumes/cat/schema/vol/train.csv" />
          </Field>
          <Field label="Evaluation data (UC Volume *.csv)" error={evalData && !csvOk(evalData) ? 'Must be /Volumes/…/*.csv' : ''}>
            <Text value={evalData} onChange={setEvalData} placeholder="/Volumes/cat/schema/vol/eval.csv" />
          </Field>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={useLora} onChange={(e) => setUseLora(e.target.checked)} />
            Use LoRA
          </label>
          <Field label="Fine-tuning label">
            <Text value={label} onChange={setLabel} placeholder="my-stability-ft" />
          </Field>
          <Field label="MLflow experiment name">
            <Text value={experiment} onChange={setExperiment} placeholder="esm2_stability" />
          </Field>
        </Section>

        {/* Column 2 — core hyperparams */}
        <Section title="Training">
          <Field label="Task type">
            <Select value={taskType} onChange={setTaskType} options={TASK_TYPES} />
          </Field>
          <Field label="Number of steps">
            <Num value={numSteps} onChange={setNumSteps} />
          </Field>
          <Field label="Micro batch size">
            <Num value={microBatch} onChange={setMicroBatch} />
          </Field>
          <Field label="Precision">
            <Select value={precision} onChange={setPrecision} options={PRECISIONS} />
          </Field>
        </Section>

        {/* Column 3 — advanced */}
        <Section title="Advanced">
          <button
            type="button"
            onClick={() => setShowAdvanced((s) => !s)}
            className="text-xs text-primary hover:underline"
          >
            {showAdvanced ? 'Hide' : 'Show'} advanced parameters
          </button>
          {showAdvanced && (
            <div className="space-y-3 pt-1">
              <Field label="Dropout"><Num value={dropout} onChange={setDropout} step="any" /></Field>
              <Field label="Hidden size"><Num value={hiddenSize} onChange={setHiddenSize} /></Field>
              <Field label="Target size"><Num value={targetSize} onChange={setTargetSize} /></Field>
              <Field label="Learning rate"><Num value={lr} onChange={setLr} step="any" /></Field>
              <Field label="LR multiplier"><Num value={lrMult} onChange={setLrMult} step="any" /></Field>
            </div>
          )}
        </Section>
      </div>

      <button
        onClick={() => start.mutate()}
        disabled={!canStart || start.isPending}
        className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
      >
        {start.isPending ? 'Launching…' : 'Start fine-tuning run'}
      </button>

      {start.data && (
        <div className="rounded-md border border-success/40 bg-success/10 p-3 text-xs">
          <span className="text-success">✓ Fine-tuning run started.</span> Run ID{' '}
          <code className="rounded bg-muted px-1">{start.data.job_run_id}</code>{' '}
          {start.data.run_url && (
            <a href={start.data.run_url} target="_blank" rel="noreferrer" className="text-primary hover:underline">
              View in Databricks ↗
            </a>
          )}
        </div>
      )}
      {start.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
          {String(start.error)}
        </div>
      )}
    </div>
  )
}

// ── small shared form controls ───────────────────────────────────────────────
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <fieldset className="space-y-3 rounded-md border border-border bg-card p-4">
      <legend className="px-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </legend>
      {children}
    </fieldset>
  )
}
function Field({ label, error, children }: { label: string; error?: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <div className="mb-1 text-xs font-medium text-foreground">{label}</div>
      {children}
      {error && <div className="mt-1 text-[10px] text-destructive">{error}</div>}
    </label>
  )
}
function Text({ value, onChange, placeholder }: { value: string; onChange: (v: string) => void; placeholder?: string }) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full rounded-md border border-border bg-background px-2 py-1 text-sm"
    />
  )
}
function Num({ value, onChange, step }: { value: number; onChange: (v: number) => void; step?: string }) {
  return (
    <input
      type="number"
      step={step}
      value={value}
      onChange={(e) => onChange(e.target.value === '' ? 0 : Number(e.target.value))}
      className="w-full rounded-md border border-border bg-background px-2 py-1 text-sm"
    />
  )
}
function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: string[] }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-md border border-border bg-background px-2 py-1 text-sm"
    >
      {options.map((o) => (
        <option key={o} value={o}>{o}</option>
      ))}
    </select>
  )
}

export { Section, Field, Text, Num, Select }

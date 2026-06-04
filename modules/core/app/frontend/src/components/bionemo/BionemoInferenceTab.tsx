// NVIDIA BioNeMo — ESM2 inference form. Port of the Streamlit inference tab.
import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { cn } from '@/lib/utils'
import { Field, Section, Select, Text } from './BionemoFinetuneTab'

const TASK_TYPES = ['regression', 'classification']

export function BionemoInferenceTab() {
  const variants = useQuery({ queryKey: ['bionemo', 'variants'], queryFn: api.bionemoVariants })
  const esm2 = variants.data?.esm2 ?? []
  const weights = useQuery({ queryKey: ['bionemo', 'weights'], queryFn: api.bionemoWeights })

  const [esmVariantSel, setEsmVariant] = useState('')
  const esmVariant = esmVariantSel || esm2[0] || ''
  const [useBaseModel, setUseBaseModel] = useState(true)
  const [selectedFtId, setSelectedFtId] = useState<number | null>(null)
  const [taskType, setTaskType] = useState('regression')
  const [dataLocation, setDataLocation] = useState('')
  const [seqColumn, setSeqColumn] = useState('')
  const [resultLocation, setResultLocation] = useState('')

  const start = useMutation({
    mutationFn: () =>
      api.bionemoInference({
        esm_variant: esmVariant,
        is_base_model: useBaseModel,
        finetune_run_id: useBaseModel ? 0 : selectedFtId ?? 0,
        task_type: taskType,
        data_location: dataLocation,
        sequence_column_name: seqColumn,
        result_location: resultLocation,
      }),
  })

  const csvOk = dataLocation.trim().startsWith('/Volumes') && dataLocation.trim().endsWith('.csv')
  const resultOk = resultLocation.trim().startsWith('/Volumes')
  const weightOk = useBaseModel || selectedFtId !== null
  const canStart = !!esmVariant && csvOk && resultOk && !!seqColumn.trim() && weightOk

  const rows = weights.data?.weights ?? []

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Run inference with the ESM-2 base model or one of your fine-tuned weights. Input must be a
        CSV file in a UC Volume; results are written as <code>results.csv</code> into the result
        folder.
      </p>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Section title="Model">
          <Field label="ESM variant">
            <Select value={esmVariant} onChange={setEsmVariant} options={esm2} />
          </Field>
          <div className="flex gap-2">
            {[
              { k: true, label: 'Base Model' },
              { k: false, label: 'Fine-tuned Weight' },
            ].map((o) => (
              <button
                key={o.label}
                type="button"
                onClick={() => setUseBaseModel(o.k)}
                className={cn(
                  'rounded-full border px-3 py-1 text-xs',
                  useBaseModel === o.k
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border text-muted-foreground hover:bg-accent',
                )}
              >
                {o.label}
              </button>
            ))}
          </div>

          {!useBaseModel && (
            <div>
              <div className="mb-1 text-xs font-medium text-foreground">Select a fine-tuned weight</div>
              {weights.isLoading ? (
                <div className="text-xs text-muted-foreground">Loading weights…</div>
              ) : rows.length === 0 ? (
                <div className="rounded-md border border-border bg-muted/30 p-3 text-xs text-muted-foreground">
                  No fine-tuned weights available yet — run a fine-tuning job first.
                </div>
              ) : (
                <div className="max-h-56 overflow-auto rounded-md border border-border">
                  <table className="w-full text-xs">
                    <thead className="bg-muted/50 uppercase text-muted-foreground">
                      <tr>
                        <th className="px-2 py-1"></th>
                        <th className="px-2 py-1 text-left">Label</th>
                        <th className="px-2 py-1 text-left">Variant</th>
                        <th className="px-2 py-1 text-left">Created by</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((w) => (
                        <tr
                          key={w.ft_id}
                          onClick={() => setSelectedFtId(w.ft_id)}
                          className={cn(
                            'cursor-pointer border-t border-border hover:bg-accent/20',
                            selectedFtId === w.ft_id && 'bg-primary/10',
                          )}
                        >
                          <td className="px-2 py-1">
                            <input type="radio" readOnly checked={selectedFtId === w.ft_id} />
                          </td>
                          <td className="px-2 py-1">{w.ft_label}</td>
                          <td className="px-2 py-1">{w.variant}</td>
                          <td className="px-2 py-1">{w.created_by ?? '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </Section>

        <Section title="Inference">
          <Field label="Task type">
            <Select value={taskType} onChange={setTaskType} options={TASK_TYPES} />
          </Field>
          <Field label="Data location (UC Volume *.csv)" error={dataLocation && !csvOk ? 'Must be /Volumes/…/*.csv' : ''}>
            <Text value={dataLocation} onChange={setDataLocation} placeholder="/Volumes/cat/schema/vol/input.csv" />
          </Field>
          <Field label="Sequence column name">
            <Text value={seqColumn} onChange={setSeqColumn} placeholder="sequence" />
          </Field>
          <Field label="Result location (UC Volume folder)" error={resultLocation && !resultOk ? 'Must be a /Volumes/… folder' : ''}>
            <Text value={resultLocation} onChange={setResultLocation} placeholder="/Volumes/cat/schema/vol/results" />
          </Field>
        </Section>
      </div>

      <button
        onClick={() => start.mutate()}
        disabled={!canStart || start.isPending}
        className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
      >
        {start.isPending ? 'Launching…' : 'Run inference'}
      </button>

      {start.data && (
        <div className="rounded-md border border-success/40 bg-success/10 p-3 text-xs">
          <span className="text-success">✓ Inference run started.</span> Run ID{' '}
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

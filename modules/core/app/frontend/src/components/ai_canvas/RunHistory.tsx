// Vortex (ai_canvas) — "Past Runs" sub-tab: browse past runs, inspect their
// workflow / inputs / outputs, and re-run with edited inputs.
import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '@/api/client'
import { Dialog } from '@/components/Dialog'
import { Drawer } from '@/components/Drawer'
import { cn } from '@/lib/utils'
import type { CanvasGraph, CanvasGraphNode, CanvasNodeType } from '@/types/api'
import { ResultGraph } from './ResultGraph'
import { InputField, ParamInput } from './NodeParamPanel'

const STATUS_BADGE: Record<string, string> = {
  submitted: 'bg-muted text-muted-foreground',
  started: 'bg-amber-500/15 text-amber-600',
  running: 'bg-amber-500/15 text-amber-600',
  complete: 'bg-emerald-500/15 text-emerald-600',
  failed: 'bg-destructive/15 text-destructive',
}

// Backend sends an ISO-8601 (UTC) start time; render it in the viewer's local
// timezone. Fall back to the raw value if it isn't a parseable date.
function fmtLocalTime(s: string): string {
  if (!s) return ''
  const d = new Date(s)
  return isNaN(d.getTime()) ? s : d.toLocaleString()
}

// Ports of a node that are fed by an upstream edge (so their inline value is
// overridden at run time and shouldn't be edited here).
function wiredInputsOf(graph: CanvasGraph | null | undefined, nodeId: string): Set<string> {
  const s = new Set<string>()
  for (const e of graph?.edges ?? []) {
    if (e.target === nodeId && e.targetHandle) s.add(e.targetHandle)
  }
  return s
}

export function PastRunsTab() {
  const [text, setText] = useState('')
  const [page, setPage] = useState(1)
  const [resultRunId, setResultRunId] = useState<string | null>(null)
  const [resultName, setResultName] = useState<string>('')
  const [resultTab, setResultTab] = useState<'workflow' | 'inputs' | 'outputs'>('workflow')
  const [rerunOpen, setRerunOpen] = useState(false)

  const runs = useQuery({
    queryKey: ['ai_canvas', 'runs', text, page],
    queryFn: () => api.aiCanvasRuns(text, page),
  })

  // Catalog gives each node's param schema (type/options/labels) so the inputs
  // view + re-run editor render proper controls, not raw key/values.
  const catalog = useQuery({
    queryKey: ['ai_canvas', 'catalog'],
    queryFn: () => api.aiCanvasCatalog(),
  })
  const byType = useMemo(
    () => new Map<string, CanvasNodeType>((catalog.data?.nodes ?? []).map((n) => [n.type, n])),
    [catalog.data],
  )

  const result = useMutation({
    mutationFn: (runId: string) => api.aiCanvasRunResult(runId),
  })

  return (
    <div>
      <div className="mb-3 flex items-center gap-2">
        <input
          value={text}
          onChange={(e) => {
            setText(e.target.value)
            setPage(1) // a new filter restarts from the first page
          }}
          placeholder="Filter by run name…"
          className="w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
        />
        {/* Status is read fresh from MLflow on each fetch; live runs change, so a
            manual refresh re-pulls the latest. */}
        <button
          onClick={() => runs.refetch()}
          disabled={runs.isFetching}
          title="Refresh run status"
          className="flex shrink-0 items-center gap-1 rounded-md border border-border px-2.5 py-1.5 text-xs hover:bg-accent disabled:opacity-40"
        >
          <span className={cn('inline-block', runs.isFetching && 'animate-spin')}>↻</span>
          Refresh
        </button>
      </div>

      {runs.isFetching && (
        <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
          <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-blue-500/30 border-t-blue-500" />
          Fetching latest status…
        </div>
      )}
      {runs.isLoading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : !runs.data || runs.data.runs.length === 0 ? (
        <p className="text-sm text-muted-foreground">No runs yet.</p>
      ) : (
        <ul className="divide-y divide-border rounded-md border border-border">
          {runs.data.runs.map((r) => (
            <li key={r.run_id} className="flex items-center gap-3 px-3 py-2">
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium">{r.run_name || r.run_id}</div>
                <div className="text-[10px] text-muted-foreground">
                  {r.node_count ?? '?'} nodes · {fmtLocalTime(r.start_time)}
                </div>
              </div>
              <span
                className={cn(
                  'rounded px-1.5 py-0.5 text-[10px] font-medium',
                  STATUS_BADGE[r.job_status] ?? 'bg-muted text-muted-foreground',
                )}
              >
                {r.job_status || 'unknown'}
              </span>
              {r.run_url && (
                <a
                  href={r.run_url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-[11px] text-primary hover:underline"
                >
                  Job ↗
                </a>
              )}
              <button
                onClick={() => {
                  setResultRunId(r.run_id)
                  setResultName(r.run_name || r.run_id)
                  setResultTab('workflow')
                  result.mutate(r.run_id)
                }}
                className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-accent"
              >
                Open
              </button>
            </li>
          ))}
        </ul>
      )}

      {/* Pager — 20 most-recent per page. */}
      {(page > 1 || runs.data?.has_more) && (
        <div className="mt-3 flex items-center justify-between text-xs">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1 || runs.isFetching}
            className="rounded-md border border-border px-2.5 py-1 hover:bg-accent disabled:opacity-40"
          >
            ← Newer
          </button>
          <span className="text-muted-foreground">Page {page}</span>
          <button
            onClick={() => setPage((p) => p + 1)}
            disabled={!runs.data?.has_more || runs.isFetching}
            className="rounded-md border border-border px-2.5 py-1 hover:bg-accent disabled:opacity-40"
          >
            Older →
          </button>
        </div>
      )}

      {/* Result viewer — a large dialog with Workflow / Inputs / Outputs tabs. */}
      <Dialog
        open={resultRunId !== null}
        onClose={() => setResultRunId(null)}
        title={resultName ? `Result · ${resultName}` : 'Workflow result'}
        width="max-w-6xl"
      >
        {result.isPending ? (
          <p className="text-sm text-muted-foreground">Loading result…</p>
        ) : (
          <>
            {/* Failure write-up — shown above the tabs whenever a step failed, so
                the "why" is the first thing you see regardless of the active tab. */}
            <FailureSummary
              runId={resultRunId}
              graph={result.data?.graph ?? null}
              nodeStatus={result.data?.node_status ?? {}}
              nodeError={result.data?.node_error ?? {}}
            />
            <div className="mb-3 flex items-center gap-1 border-b border-border">
              {(['workflow', 'inputs', 'outputs'] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setResultTab(t)}
                  className={cn(
                    '-mb-px border-b-2 px-3 py-1.5 text-xs font-medium capitalize',
                    resultTab === t
                      ? 'border-primary text-foreground'
                      : 'border-transparent text-muted-foreground hover:text-foreground',
                  )}
                >
                  {t}
                </button>
              ))}
              <div className="ml-auto flex items-center gap-3 pb-1">
                <button
                  onClick={() => setRerunOpen(true)}
                  disabled={!result.data?.graph}
                  title="Edit inputs and dispatch a new run"
                  className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-accent disabled:opacity-40"
                >
                  ↻ Re-run…
                </button>
                <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                  <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-emerald-500" /> passed</span>
                  <span className="flex items-center gap-1"><span className="h-2 w-2 animate-pulse rounded-full bg-amber-500" /> running</span>
                  <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-red-500" /> failed</span>
                  <span className="flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full bg-slate-400" />{' '}
                    {Object.values(result.data?.node_status ?? {}).includes('failed') ||
                    Object.keys(result.data?.node_error ?? {}).length > 0
                      ? 'skipped'
                      : 'pending'}
                  </span>
                </div>
              </div>
            </div>

            {resultTab === 'workflow' ? (
              result.data?.graph ? (
                <ResultGraph
                  graph={result.data.graph as CanvasGraph}
                  nodeStatus={result.data.node_status ?? {}}
                />
              ) : (
                <p className="text-sm text-muted-foreground">
                  No saved graph for this run (older run, or it failed before the graph was logged).
                </p>
              )
            ) : resultTab === 'inputs' ? (
              <InputsTab graph={result.data?.graph ?? null} byType={byType} />
            ) : (
              <OutputsTab data={result.data} />
            )}
          </>
        )}
      </Dialog>

      {/* Re-run drawer — edit the run's inputs/params, then dispatch a fresh run. */}
      <RerunDrawer
        open={rerunOpen}
        onClose={() => setRerunOpen(false)}
        graph={(result.data?.graph as CanvasGraph) ?? null}
        byType={byType}
        runName={resultName}
        onSubmitted={() => {
          setRerunOpen(false)
          setResultRunId(null)
        }}
      />
    </div>
  )
}

// Read-only view of what fed the run: every input-bearing node's params + inline
// (unwired) input values, in graph order.
function InputsTab({
  graph,
  byType,
}: {
  graph: CanvasGraph | null
  byType: Map<string, CanvasNodeType>
}) {
  if (!graph) return <p className="text-sm text-muted-foreground">No saved graph for this run.</p>
  const rows: { node: string; field: string; value: unknown }[] = []
  for (const n of graph.nodes) {
    const cat = byType.get(n.type)
    const label = n.label || cat?.label || n.type
    const wired = wiredInputsOf(graph, n.id)
    for (const p of cat?.params ?? []) {
      const v = n.params?.[p.name]
      if (v !== undefined && v !== '' && v !== null) rows.push({ node: label, field: p.label || p.name, value: v })
    }
    for (const ip of cat?.inputs ?? []) {
      if (wired.has(ip.name)) continue
      const v = n.inputs?.[ip.name]
      if (v !== undefined && v !== '' && v !== null) rows.push({ node: label, field: ip.label || ip.name, value: v })
    }
  }
  if (rows.length === 0)
    return <p className="text-sm text-muted-foreground">No editable inputs captured for this run.</p>
  return (
    <div className="max-h-[64vh] space-y-2 overflow-auto">
      {rows.map((r, i) => (
        <div key={i} className="rounded-md border border-border">
          <div className="border-b border-border bg-muted/30 px-3 py-1.5 text-[11px] font-medium">
            {r.node} · <span className="text-muted-foreground">{r.field}</span>
          </div>
          <pre className="max-h-48 overflow-auto px-3 py-2 text-[11px] leading-snug whitespace-pre-wrap break-words">
            {typeof r.value === 'string' ? r.value : JSON.stringify(r.value, null, 2)}
          </pre>
        </div>
      ))}
    </div>
  )
}

// Side drawer: edit the run's inputs/params (seeded from the run's graph), then
// dispatch a brand-new run with the edited graph. Lands in gwb_ai_canvas.
function RerunDrawer({
  open,
  onClose,
  graph,
  byType,
  runName,
  onSubmitted,
}: {
  open: boolean
  onClose: () => void
  graph: CanvasGraph | null
  byType: Map<string, CanvasNodeType>
  runName: string
  onSubmitted: () => void
}) {
  // Editable deep copy, reseeded each time the drawer opens.
  const [g, setG] = useState<CanvasGraph | null>(graph)
  useEffect(() => {
    if (open && graph) setG(JSON.parse(JSON.stringify(graph)) as CanvasGraph)
  }, [open, graph])

  const qc = useQueryClient()
  const run = useMutation({
    mutationFn: () =>
      api.aiCanvasRun({
        graph: g as CanvasGraph,
        experiment_name: 'gwb_ai_canvas',
        run_name: `${runName || 'ai_canvas'} (re-run)`,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['ai_canvas', 'runs'] })
      onSubmitted()
    },
  })

  const setParam = (nodeId: string, name: string, value: unknown) =>
    setG((prev) =>
      prev
        ? {
            ...prev,
            nodes: prev.nodes.map((n) =>
              n.id === nodeId ? { ...n, params: { ...n.params, [name]: value } } : n,
            ),
          }
        : prev,
    )
  const setInput = (nodeId: string, name: string, value: unknown) =>
    setG((prev) =>
      prev
        ? {
            ...prev,
            nodes: prev.nodes.map((n) =>
              n.id === nodeId ? { ...n, inputs: { ...(n.inputs ?? {}), [name]: value } } : n,
            ),
          }
        : prev,
    )

  // Nodes that have something to edit: params, or an unwired input port.
  const editable: { node: CanvasGraphNode; cat: CanvasNodeType; unwired: CanvasNodeType['inputs'] }[] =
    []
  for (const n of g?.nodes ?? []) {
    const cat = byType.get(n.type)
    if (!cat) continue
    const wired = wiredInputsOf(g, n.id)
    const unwired = cat.inputs.filter((p) => !wired.has(p.name))
    if (cat.params.length > 0 || unwired.length > 0) editable.push({ node: n, cat, unwired })
  }

  return (
    <Drawer open={open} onClose={onClose} title="Re-run with edited inputs" width="max-w-xl">
      {!g ? (
        <p className="text-sm text-muted-foreground">No graph to re-run.</p>
      ) : editable.length === 0 ? (
        <p className="text-sm text-muted-foreground">This workflow has no editable inputs.</p>
      ) : (
        <div className="space-y-4">
          <p className="text-xs text-muted-foreground">
            Edit any input or parameter below, then dispatch a fresh run. The original run is
            unchanged.
          </p>
          {editable.map(({ node, cat, unwired }) => (
            <div key={node.id} className="rounded-md border border-border p-3">
              <div className="mb-2 text-xs font-semibold">{node.label || cat.label}</div>
              <div className="space-y-3">
                {unwired.map((p) => (
                  <InputField
                    key={`in-${p.name}`}
                    port={p}
                    value={node.inputs?.[p.name]}
                    wired={false}
                    onChange={(v) => setInput(node.id, p.name, v)}
                  />
                ))}
                {cat.params.map((p) => (
                  <ParamInput
                    key={`pm-${p.name}`}
                    param={p}
                    value={node.params?.[p.name]}
                    onChange={(v) => setParam(node.id, p.name, v)}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="sticky bottom-0 -mx-5 mt-4 flex items-center justify-end gap-2 border-t border-border bg-card px-5 py-3">
        {run.isError && (
          <span className="mr-auto text-xs text-destructive">
            {(run.error as Error)?.message || 'Re-run failed.'}
          </span>
        )}
        <button
          onClick={onClose}
          className="rounded-md border border-border px-3 py-1.5 text-xs hover:bg-accent"
        >
          Cancel
        </button>
        <button
          onClick={() => run.mutate()}
          disabled={!g || run.isPending}
          className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-40"
        >
          {run.isPending ? 'Dispatching…' : 'Submit re-run'}
        </button>
      </div>
    </Drawer>
  )
}

// Right column: AI triage of a failed step — a data-vs-system verdict + root cause
// + suggested fix, fetched once and revealed when it settles.
function ErrorInterpretation({ errorText, context }: { errorText: string; context: string }) {
  const q = useQuery({
    queryKey: ['ai_canvas', 'interpret-error', context, errorText.slice(0, 300)],
    queryFn: () => api.aiCanvasInterpretError({ error_trace: errorText, context }),
    enabled: !!errorText.trim(),
    staleTime: Infinity,
    retry: false,
  })
  const cls = q.data?.classification
  const badge =
    cls === 'data'
      ? { label: 'Data error', cn: 'bg-amber-500/15 text-amber-600 dark:text-amber-400' }
      : cls === 'system'
        ? { label: 'System error', cn: 'bg-purple-500/15 text-purple-600 dark:text-purple-400' }
        : { label: 'Needs review', cn: 'bg-muted text-muted-foreground' }
  return (
    <div className="rounded border border-border bg-card/60 p-2">
      <div className="mb-1.5 flex items-center gap-1.5 font-medium text-foreground">
        <span aria-hidden className="material-symbols-outlined text-[16px] leading-none">
          bolt
        </span>
        AI analysis
      </div>
      {q.isFetching ? (
        <span className="text-muted-foreground">Analyzing the failure…</span>
      ) : q.isError ? (
        <span className="text-destructive">Couldn’t analyze this error.</span>
      ) : (
        <div className="space-y-2">
          <span
            className={cn('inline-block rounded px-1.5 py-0.5 text-[10px] font-semibold', badge.cn)}
          >
            {badge.label}
          </span>
          {q.data?.root_cause && (
            <div>
              <div className="font-medium text-foreground">Root cause</div>
              <p className="leading-snug text-muted-foreground">{q.data.root_cause}</p>
            </div>
          )}
          {q.data?.fix && (
            <div>
              <div className="font-medium text-foreground">Suggested fix</div>
              <p className="leading-snug text-muted-foreground">{q.data.fix}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Lazily fetches the ORIGINATING Databricks job behind a failed node and shows its
// real task error/stack trace + a link to the job run page — "dig deeper".
function JobErrorDigger({ runId, nodeId }: { runId: string; nodeId: string }) {
  const [open, setOpen] = useState(false)
  const q = useQuery({
    queryKey: ['ai_canvas', 'node-job-error', runId, nodeId],
    queryFn: () => api.aiCanvasNodeJobError(runId, nodeId),
    enabled: open,
  })
  return (
    <div className="mt-1.5">
      <button
        onClick={() => setOpen((v) => !v)}
        className="text-[11px] font-medium text-primary hover:underline"
      >
        {open ? '▾' : '▸'} Examine the child job
      </button>
      {open && (
        <div className="mt-1 rounded border border-border bg-background/60 p-2 text-[11px]">
          {q.isFetching ? (
            <span className="text-muted-foreground">Fetching the originating job’s error…</span>
          ) : q.isError ? (
            <span className="text-destructive">Couldn’t load the job error.</span>
          ) : !q.data?.found ? (
            <span className="text-muted-foreground">
              {q.data?.message || 'No originating job to dig into.'}
            </span>
          ) : (
            <div className="space-y-2">
              {q.data.run_page_url && (
                <a
                  href={q.data.run_page_url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary hover:underline"
                >
                  Open job run {q.data.job_run_id} ↗
                </a>
              )}
              {q.data.tasks.length === 0 && (
                <p className="text-muted-foreground">No task-level error captured on this job run.</p>
              )}
              {q.data.tasks.map((t, i) => {
                const trace = t.error_trace || t.error || ''
                return (
                  <div key={i}>
                    <div className="font-medium text-foreground">
                      {t.task_key} · {t.result_state}
                    </div>
                    {t.state_message && <div className="text-muted-foreground">{t.state_message}</div>}
                    {trace && (
                      // Two columns: raw error (left) + AI root-cause/fix (right).
                      <div className="mt-1 grid grid-cols-1 gap-3 md:grid-cols-2">
                        <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/40 p-2 leading-snug text-destructive">
                          {trace}
                        </pre>
                        <ErrorInterpretation errorText={trace} context={`${nodeId} · ${t.task_key}`} />
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// A prominent write-up of why a run failed: which step, the error, a likely cause,
// and how far the pipeline got. Renders nothing when no step failed.
function FailureSummary({
  runId,
  graph,
  nodeStatus,
  nodeError,
}: {
  runId: string | null
  graph: CanvasGraph | null
  nodeStatus: Record<string, string>
  nodeError: Record<string, string>
}) {
  const labels = new Map<string, { label: string; type: string }>(
    (graph?.nodes ?? []).map((n) => [n.id, { label: n.label || n.type, type: n.type }]),
  )
  const failedIds = [
    ...new Set([
      ...Object.keys(nodeStatus).filter((id) => nodeStatus[id] === 'failed'),
      ...Object.keys(nodeError),
    ]),
  ]
  if (failedIds.length === 0) return null

  const total = graph?.nodes?.length ?? Object.keys(nodeStatus).length
  const completed = Object.values(nodeStatus).filter((s) => s === 'complete').length
  const didNotRun = (graph?.nodes ?? [])
    .filter((n) => {
      const s = nodeStatus[n.id]
      return s !== 'complete' && s !== 'failed' && !(n.id in nodeError)
    })
    .map((n) => n.label || n.type)

  return (
    <div className="mb-3 rounded-md border border-destructive/40 bg-destructive/5 p-3">
      <div className="mb-1 text-sm font-semibold text-destructive">
        ⚠ This run failed{total ? ` — ${completed} of ${total} step${total === 1 ? '' : 's'} completed` : ''}
      </div>
      <div className="space-y-2">
        {failedIds.map((id) => {
          const meta = labels.get(id)
          const err = nodeError[id] || 'No error message was captured for this step.'
          const ctx = meta?.type ? `${meta.label} (${meta.type})` : id
          // A step backed by a Databricks job ("Job run N …") has a deeper trace to
          // fetch → offer "Examine the child job". An in-process failure (validation,
          // a chain) IS the whole error → show it + AI interpretation inline.
          const hasChildJob = runId != null && /job run \d+/i.test(err)
          return (
            <div key={id} className="text-xs">
              <div className="font-medium text-foreground">
                Failed at: {meta?.label ?? id}
                {meta?.type && <span className="text-muted-foreground"> ({meta.type})</span>}
              </div>
              {hasChildJob ? (
                <JobErrorDigger runId={runId as string} nodeId={id} />
              ) : (
                <div className="mt-1 grid grid-cols-1 gap-3 md:grid-cols-2">
                  <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded bg-background/60 p-2 text-[11px] leading-snug text-destructive">
                    {err}
                  </pre>
                  <ErrorInterpretation errorText={err} context={ctx} />
                </div>
              )}
            </div>
          )
        })}
      </div>
      {didNotRun.length > 0 && (
        <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
          Did not run (blocked by the failure): {didNotRun.join(', ')}.
        </p>
      )}
    </div>
  )
}

// Per-node outputs + errors as collapsible expanders.
function OutputsTab({
  data,
}: {
  data:
    | {
        result: Record<string, unknown>
        graph: CanvasGraph | null
        node_status: Record<string, string>
        node_error: Record<string, string>
      }
    | undefined
}) {
  if (!data) return <p className="text-sm text-muted-foreground">No result.</p>
  const res = (data.result ?? {}) as {
    node_outputs?: Record<string, unknown>
    final_outputs?: Record<string, unknown>
  }
  const labels = new Map<string, string>(
    (data.graph?.nodes ?? []).map((n) => [n.id, n.label || n.type]),
  )
  const nodeOutputs = res.node_outputs ?? {}
  const finalOutputs = res.final_outputs ?? {}
  const errors = data.node_error ?? {}
  const status = data.node_status ?? {}
  // A not-run node means "pending" only while the run is still going; once the run
  // has FAILED (terminal) those nodes were "skipped" — they'll never run.
  const runFailed = Object.values(status).includes('failed') || Object.keys(errors).length > 0

  // Show EVERY node (in graph order), not just ones with outputs — a failed run
  // writes no node_outputs, so keying off outputs alone left only the failed node.
  const graphIds = (data.graph?.nodes ?? []).map((n) => n.id)
  // output_sink nodes just COLLECT — their node output is {} and the real value is
  // rendered below as an "Output · <label>" row. Drop the empty duplicate row for a
  // successful sink (keep it if it errored/failed, so the failure is still visible).
  const sinkIds = new Set(
    (data.graph?.nodes ?? []).filter((n) => n.type === 'output_sink').map((n) => n.id),
  )
  const ids = [
    ...new Set([...graphIds, ...Object.keys(status), ...Object.keys(nodeOutputs), ...Object.keys(errors)]),
  ].filter((id) => !(sinkIds.has(id) && status[id] !== 'failed' && !(id in errors)))

  if (ids.length === 0 && Object.keys(finalOutputs).length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No outputs captured — the run may have produced no result.
      </p>
    )
  }

  const row = (key: string, title: string, badge: string | undefined, body: unknown, err?: string) => (
    <details key={key} className="rounded-md border border-border">
      <summary className="cursor-pointer px-3 py-2 text-xs font-medium">
        {title}
        {badge && (
          <span
            className={cn(
              'ml-2 rounded px-1.5 py-0.5 text-[10px]',
              badge === 'complete'
                ? 'bg-emerald-500/15 text-emerald-600'
                : badge === 'failed'
                  ? 'bg-destructive/15 text-destructive'
                  : badge === 'running'
                    ? 'bg-amber-500/15 text-amber-600'
                    : 'bg-muted text-muted-foreground',
            )}
          >
            {badge === 'complete'
              ? 'passed'
              : badge === 'failed'
                ? 'failed'
                : badge === 'running'
                  ? 'running'
                  : runFailed
                    ? 'skipped'
                    : 'pending'}
          </span>
        )}
      </summary>
      <div className="border-t border-border p-2">
        {err && <p className="mb-2 text-xs text-destructive">⚠ {err}</p>}
        {body !== undefined ? (
          <pre className="max-h-72 overflow-auto rounded bg-muted/30 p-2 text-[11px] leading-snug">
            {typeof body === 'string' ? body : JSON.stringify(body, null, 2)}
          </pre>
        ) : (
          !err && <p className="text-xs italic text-muted-foreground">No output captured for this node.</p>
        )}
      </div>
    </details>
  )

  return (
    <div className="max-h-[64vh] space-y-2 overflow-auto">
      {ids.map((id) =>
        row(id, labels.get(id) ?? id, status[id], id in nodeOutputs ? nodeOutputs[id] : undefined, errors[id]),
      )}
      {Object.entries(finalOutputs).map(([name, val]) => row(`final-${name}`, `Output · ${name}`, 'complete', val))}
    </div>
  )
}

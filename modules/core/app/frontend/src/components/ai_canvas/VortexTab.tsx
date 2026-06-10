// Vortex (ai_canvas) — AI-assisted drag-and-drop workflow canvas.
//
// Visible name "Vortex"; feature id `ai_canvas`. This tab lets a user compose a
// workflow from deployed endpoints, batch jobs, and data-IO nodes, then run it
// (run dispatch + AI generation + save/load arrive in later increments).
import { useCallback, useMemo, useRef, useState } from 'react'
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  addEdge,
  useEdgesState,
  useNodesState,
} from '@xyflow/react'
import type { Connection, NodeTypes, ReactFlowInstance } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useMutation, useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { useThemeStore } from '@/stores/theme'
import { CanvasNode } from './CanvasNode'
import { NodePalette } from './NodePalette'
import { NodeParamPanel } from './NodeParamPanel'
import { RunHistory } from './RunHistory'
import { WorkflowLibrary } from './WorkflowLibrary'
import {
  autoLayout,
  defaultParams,
  fromCanvasGraph,
  graphValidationErrors,
  ioTypeForDtype,
  nextNodeId,
  portsCompatible,
  toCanvasGraph,
} from './graph'
import type { VortexEdge, VortexNode } from './graph'
import type { CanvasNodeType } from '@/types/api'

const nodeTypes: NodeTypes = { vortex: CanvasNode }

// Fun, catalog-valid example goals for the ✨ "Show me how" button.
const EXAMPLE_GOALS = [
  'Fold a protein sequence and predict its solubility',
  'Dock a small molecule into a predicted protein structure',
  'Design a protein binder around a target and validate it',
  'Run an ADMET screen on a candidate molecule',
  'Predict structure with AlphaFold and check thermostability',
  'Annotate genetic variants from a VCF and flag pathogenic ones',
  'Generate molecules and screen them for toxicity',
  'Embed a protein sequence and predict its half-life',
]

// Timestamp slug for default names, e.g. "20260609_135501".
function ts(): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
}

// Auto-generated default workflow name, e.g. "vortex_20260609_1355".
function defaultWorkflowName(): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `vortex_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`
}

function VortexCanvas() {
  const theme = useThemeStore((s) => s.theme)
  const wrapperRef = useRef<HTMLDivElement>(null)
  const rfRef = useRef<ReactFlowInstance<VortexNode, VortexEdge> | null>(null)

  const [nodes, setNodes, onNodesChange] = useNodesState<VortexNode>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<VortexEdge>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  // Yellow banner: reserved for critical warnings / errors that persist until
  // dismissed. Routine confirmations go to the auto-disappearing canvas toast.
  const [notice, setNotice] = useState<string | null>(null)
  // The banner message the user dismissed (hidden until the message changes).
  const [dismissed, setDismissed] = useState<string | null>(null)
  // Transient toast on top of the canvas — auto-clears after a few seconds.
  const [toast, setToast] = useState<string | null>(null)
  const toastTimer = useRef<number | undefined>(undefined)
  const showToast = useCallback((msg: string) => {
    setToast(msg)
    if (toastTimer.current) window.clearTimeout(toastTimer.current)
    toastTimer.current = window.setTimeout(() => setToast(null), 3500)
  }, [])
  // Transient "Finding a transform…" overlay (with spinner) during the AI lookup.
  const [suggesting, setSuggesting] = useState<string | null>(null)
  const [goal, setGoal] = useState('')
  // Track the loaded workflow so Save updates it (vs. always creating a copy).
  const [loadedId, setLoadedId] = useState<string | null>(null)
  // Editable workflow name, auto-generated on start (shown in the right panel).
  const [workflowName, setWorkflowName] = useState(defaultWorkflowName)
  // MLflow tracking inputs (defaulted like other screens; experiment lives in
  // the user's workspace folder).
  const [experimentName, setExperimentName] = useState('gwb_ai_canvas')
  const [runName, setRunName] = useState(() => `vortex_${ts()}`)

  const catalogQuery = useQuery({
    queryKey: ['ai_canvas', 'catalog'],
    queryFn: api.aiCanvasCatalog,
    staleTime: 5 * 60 * 1000,
  })

  const catalog = useMemo(() => catalogQuery.data?.nodes ?? [], [catalogQuery.data])
  const catalogByType = useMemo(
    () => new Map(catalog.map((n) => [n.type, n])),
    [catalog],
  )

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedId) ?? null,
    [nodes, selectedId],
  )

  // Replace the canvas with a graph (from AI generation or a loaded workflow).
  const loadGraph = useCallback(
    (graph: Parameters<typeof fromCanvasGraph>[0]) => {
      const { nodes: ns, edges: es } = fromCanvasGraph(graph, catalogByType)
      const styledEdges = es.map((e) => ({ ...e, animated: false }))
      setSelectedId(null)
      setNodes(ns)
      // Apply edges only AFTER the new nodes have mounted + measured their
      // handles — otherwise React Flow silently drops edges that reference
      // not-yet-measured handles (the "missing edges on load" bug).
      setEdges([])
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          setEdges(styledEdges)
          rfRef.current?.fitView({ maxZoom: 1, duration: 300 })
        }),
      )
    },
    [catalogByType, setNodes, setEdges],
  )

  const generate = useMutation({
    mutationFn: (g: string) => api.aiCanvasGenerate({ goal: g }),
    onSuccess: (res) => {
      loadGraph(res.graph)
      setLoadedId(null) // a generated graph is new/unsaved (keep the current name)
      showToast(
        res.graph.nodes.length
          ? `Generated a ${res.graph.nodes.length}-node workflow — edit it, then Run.`
          : 'No nodes generated. Try rephrasing your goal.',
      )
    },
    onError: (err: Error) => setNotice(err.message),
  })

  // ✨ "Show me how" — drop in a random fun goal and generate it.
  const tryRandom = useCallback(() => {
    const g = EXAMPLE_GOALS[Math.floor(Math.random() * EXAMPLE_GOALS.length)]
    setGoal(g)
    generate.mutate(g)
  }, [generate])

  // ── add a node ─────────────────────────────────────────────────────────────
  // Prebuilt workflows (category `batch` — chains + jobs) land pre-wired: every
  // input port gets a matching IO source (Text/Volume/Delta input) and every
  // output a collecting Output sink, so the graph is runnable on drop. The user
  // just fills in the IO values. Other node kinds drop bare (they're meant to be
  // chained onto an existing graph).
  const makeNode = (cat: CanvasNodeType, position: { x: number; y: number }, id?: string): VortexNode => ({
    id: id ?? nextNodeId(cat.type),
    type: 'vortex',
    position,
    data: { typeKey: cat.type, label: cat.label, params: defaultParams(cat), catalog: cat },
  })

  const addNode = useCallback(
    (cat: CanvasNodeType, position: { x: number; y: number }) => {
      const mainId = nextNodeId(cat.type)
      const main = makeNode(cat, position, mainId)

      if (cat.category !== 'batch' || (cat.inputs.length === 0 && cat.outputs.length === 0)) {
        setNodes((nds) => [...nds, main])
        setSelectedId(mainId)
        return
      }

      // Auto-create + wire the surrounding IO nodes.
      const extraNodes: VortexNode[] = []
      const conns: Connection[] = []
      cat.inputs.forEach((port, i) => {
        const ioCat = catalogByType.get(ioTypeForDtype(port.dtype))
        if (!ioCat) return
        const io = makeNode(ioCat, { x: position.x - 260, y: position.y + i * 120 })
        extraNodes.push(io)
        conns.push({
          source: io.id,
          sourceHandle: ioCat.outputs[0]?.name ?? null,
          target: mainId,
          targetHandle: port.name,
        })
      })
      cat.outputs.forEach((port, i) => {
        const sinkCat = catalogByType.get('output_sink')
        if (!sinkCat) return
        const sink = makeNode(sinkCat, { x: position.x + 260, y: position.y + i * 120 })
        extraNodes.push(sink)
        conns.push({
          source: mainId,
          sourceHandle: port.name,
          target: sink.id,
          targetHandle: sinkCat.inputs[0]?.name ?? null,
        })
      })

      const nextNodes = [...nodes, main, ...extraNodes]
      const nextEdges = conns.reduce<VortexEdge[]>(
        (eds, c) => addEdge({ ...c, animated: false }, eds),
        edges,
      )
      setNodes(autoLayout(nextNodes, nextEdges))
      setEdges(nextEdges)
      setSelectedId(mainId)
      setTimeout(() => rfRef.current?.fitView({ maxZoom: 1, duration: 300 }), 50)
    },
    [nodes, edges, catalogByType, setNodes, setEdges],
  )

  const addNodeAtCenter = useCallback(
    (cat: CanvasNodeType) => {
      const inst = rfRef.current
      const center = inst
        ? inst.screenToFlowPosition({
            x: (wrapperRef.current?.clientWidth ?? 800) / 2,
            y: (wrapperRef.current?.clientHeight ?? 500) / 2,
          })
        : { x: 200, y: 120 }
      addNode(cat, center)
    },
    [addNode],
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const typeKey = e.dataTransfer.getData('application/vortex-node')
      const cat = catalogByType.get(typeKey)
      if (!cat || !rfRef.current) return
      const position = rfRef.current.screenToFlowPosition({ x: e.clientX, y: e.clientY })
      addNode(cat, position)
    },
    [addNode, catalogByType],
  )

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  // ── insert an AI-picked transform between source → target (rewires through it) ──
  const insertTransform = useCallback(
    (conn: Connection, tcat: CanvasNodeType, params: Record<string, unknown>) => {
      const src = nodes.find((n) => n.id === conn.source)
      const dst = nodes.find((n) => n.id === conn.target)
      const mid =
        src && dst
          ? { x: (src.position.x + dst.position.x) / 2, y: (src.position.y + dst.position.y) / 2 + 30 }
          : { x: 200, y: 160 }
      const id = nextNodeId(tcat.type)
      const tNode: VortexNode = {
        id,
        type: 'vortex',
        position: mid,
        data: { typeKey: tcat.type, label: tcat.label, params: { ...defaultParams(tcat), ...params }, catalog: tcat },
      }
      const inPort = tcat.inputs[0]?.name
      const outPort = tcat.outputs[0]?.name
      const nextNodes = [...nodes, tNode]
      const nextEdges = addEdge(
        { source: id, sourceHandle: outPort, target: conn.target, targetHandle: conn.targetHandle, animated: false },
        addEdge(
          { source: conn.source, sourceHandle: conn.sourceHandle, target: id, targetHandle: inPort, animated: false },
          edges,
        ),
      )
      // Re-tidy the graph so the inserted node doesn't overlap, then refit.
      setNodes(autoLayout(nextNodes, nextEdges))
      setEdges(nextEdges)
      setTimeout(() => rfRef.current?.fitView({ maxZoom: 1, duration: 300 }), 50)
    },
    [nodes, edges, setNodes, setEdges],
  )

  // ── connect edges; on a dtype mismatch, let AI auto-insert a bridging transform ──
  const onConnect = useCallback(
    (conn: Connection) => {
      const src = nodes.find((n) => n.id === conn.source)
      const dst = nodes.find((n) => n.id === conn.target)
      const srcPort = src?.data.catalog?.outputs.find((p) => p.name === conn.sourceHandle)
      const dstPort = dst?.data.catalog?.inputs.find((p) => p.name === conn.targetHandle)
      if (srcPort && dstPort && !portsCompatible(srcPort.dtype, dstPort.dtype)) {
        const fail = `Can't connect ${srcPort.dtype} → ${dstPort.dtype}. Outputs must match the target input type.`
        setNotice(null)
        setSuggesting(`Finding a transform for ${srcPort.dtype} → ${dstPort.dtype}…`)
        api
          .aiCanvasSuggestTransform({
            source_dtype: srcPort.dtype,
            target_dtype: dstPort.dtype,
            source_label: src?.data.label ?? '',
            target_label: dst?.data.label ?? '',
          })
          .then((res) => {
            setSuggesting(null)
            const tcat = res.type ? catalogByType.get(res.type) : undefined
            if (tcat) {
              insertTransform(conn, tcat, res.params ?? {})
              showToast(`Inserted “${tcat.label}” to convert ${srcPort.dtype} → ${dstPort.dtype}.`)
            } else {
              setNotice(fail)
            }
          })
          .catch(() => {
            setSuggesting(null)
            setNotice(fail)
          })
        return
      }
      setNotice(null)
      setEdges((eds) => addEdge({ ...conn, animated: false }, eds))
    },
    [nodes, setEdges, catalogByType, insertTransform],
  )

  // ── param panel callbacks ───────────────────────────────────────────────────
  const patchSelected = useCallback(
    (mutate: (n: VortexNode) => VortexNode) => {
      setNodes((nds) => nds.map((n) => (n.id === selectedId ? mutate(n) : n)))
    },
    [selectedId, setNodes],
  )

  const onChangeParam = useCallback(
    (name: string, value: unknown) =>
      patchSelected((n) => ({ ...n, data: { ...n.data, params: { ...n.data.params, [name]: value } } })),
    [patchSelected],
  )
  const onRename = useCallback(
    (label: string) => patchSelected((n) => ({ ...n, data: { ...n.data, label } })),
    [patchSelected],
  )
  const onDeleteSelected = useCallback(() => {
    if (!selectedId) return
    setNodes((nds) => nds.filter((n) => n.id !== selectedId))
    setEdges((eds) => eds.filter((e) => e.source !== selectedId && e.target !== selectedId))
    setSelectedId(null)
  }, [selectedId, setNodes, setEdges])

  const clearCanvas = useCallback(() => {
    setNodes([])
    setEdges([])
    setSelectedId(null)
    setLoadedId(null)
    setWorkflowName(defaultWorkflowName())
  }, [setNodes, setEdges])

  // Tidy the whole graph left→right and refit (manual button + after auto-insert).
  const autoArrange = useCallback(() => {
    setNodes(autoLayout(nodes, edges))
    setTimeout(() => rfRef.current?.fitView({ maxZoom: 1, duration: 300 }), 50)
  }, [nodes, edges, setNodes])

  // ── run dispatch ─────────────────────────────────────────────────────────────
  // Dispatch is fire-and-forget: the orchestrator runs as a serverless job that
  // can take minutes-to-hours (it blocks on any batch child jobs), so we don't
  // poll live status here — we point the user to the Past Runs dialog and show a
  // confirmation banner. Past Runs is the single source of run status.
  const run = useMutation({
    mutationFn: () =>
      api.aiCanvasRun({
        graph: toCanvasGraph(nodes, edges),
        experiment_name: experimentName.trim() || 'gwb_ai_canvas',
        run_name: runName.trim() || workflowName || 'ai_canvas_run',
      }),
    onSuccess: () => {
      showToast('▶ Workflow started — track its progress in the “Past Runs” dialog.')
    },
    onError: (err: Error) => setNotice(err.message),
  })

  // Run is gated on the graph validating — inputs wired, required values filled.
  const validationErrors = useMemo(() => graphValidationErrors(nodes, edges), [nodes, edges])
  const runnable = nodes.length > 0 && validationErrors.length === 0
  const runDisabledReason =
    nodes.length === 0
      ? 'Add a node to begin'
      : validationErrors.length > 0
        ? `Fix ${validationErrors.length} issue(s) before running`
        : ''

  // Centered spinner popup — AI generation / transform lookup only (runs are
  // tracked in Past Runs, not with a live spinner).
  const popupMessage = suggesting ?? (generate.isPending ? 'Generating workflow…' : null)

  // Yellow banner: the latest notice (dispatch confirmation, errors, warnings).
  const banner = notice

  return (
    <div className="flex h-[70vh] min-h-[520px] flex-col overflow-hidden rounded-md border border-border">
      {/* Toolbar */}
      <div className="flex flex-col gap-2 border-b border-border bg-card/60 px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">
            Describe a goal and let AI draft the workflow, or build it by hand.
          </span>
          <div className="ml-auto flex items-center gap-2">
            <WorkflowLibrary
              graph={toCanvasGraph(nodes, edges)}
              loadedId={loadedId}
              loadedName={workflowName}
              disabled={nodes.length === 0}
              onLoad={(detail) => {
                loadGraph(detail.graph)
                setLoadedId(detail.workflow_id)
                setWorkflowName(detail.name)
                showToast(`Loaded "${detail.name}".`)
              }}
              onSaved={(id, name) => {
                setLoadedId(id)
                setWorkflowName(name)
                showToast(`Saved "${name}".`)
              }}
            />
            <RunHistory />
            <button
              onClick={autoArrange}
              disabled={nodes.length === 0}
              className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-accent disabled:opacity-40"
            >
              Auto arrange
            </button>
            <button
              onClick={clearCanvas}
              disabled={nodes.length === 0}
              className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-accent disabled:opacity-40"
            >
              Clear
            </button>
            <button
              onClick={() => run.mutate()}
              disabled={!runnable || run.isPending}
              title={runDisabledReason}
              className="rounded-md bg-primary px-3 py-1 text-xs font-medium text-primary-foreground hover:opacity-90 disabled:opacity-40"
            >
              {run.isPending ? 'Dispatching…' : '▶ Run'}
            </button>
          </div>
        </div>
        <form
          className="flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            if (goal.trim()) generate.mutate(goal.trim())
          }}
        >
          <button
            type="button"
            onClick={tryRandom}
            disabled={generate.isPending}
            title="Show me how"
            aria-label="Show me how"
            className="shrink-0 rounded-md px-1 text-base transition-transform hover:scale-125 disabled:opacity-40"
          >
            ✨
          </button>
          <input
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="e.g. fold a protein sequence and predict its solubility"
            className="min-w-0 flex-1 rounded-md border border-border bg-background px-2.5 py-1.5 text-xs"
          />
          <button
            type="submit"
            disabled={!goal.trim() || generate.isPending}
            className="shrink-0 rounded-md border border-primary/50 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary hover:bg-primary/20 disabled:opacity-40"
          >
            {generate.isPending ? 'Generating…' : 'Generate workflow'}
          </button>
        </form>
      </div>

      {banner && banner !== dismissed && (
        <div className="flex items-start gap-2 border-b border-amber-500/40 bg-amber-500/10 px-3 py-1.5 text-xs text-amber-700 dark:text-amber-400">
          <span className="min-w-0 flex-1">{banner}</span>
          <button
            onClick={() => setDismissed(banner)}
            aria-label="Dismiss message"
            title="Dismiss"
            className="shrink-0 rounded px-1 leading-none hover:bg-amber-500/20"
          >
            ✕
          </button>
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        {catalogQuery.isLoading ? (
          <div className="flex w-56 items-center justify-center gap-2 border-r border-border text-xs text-muted-foreground">
            <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-blue-500/30 border-t-blue-500" />
            Loading nodes…
          </div>
        ) : (
          <NodePalette catalog={catalog} onAdd={addNodeAtCenter} />
        )}

        <div ref={wrapperRef} className="relative min-w-0 flex-1" onDrop={onDrop} onDragOver={onDragOver}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={(inst) => (rfRef.current = inst)}
            nodeTypes={nodeTypes}
            onNodeClick={(_, n) => setSelectedId(n.id)}
            onPaneClick={() => setSelectedId(null)}
            colorMode={theme}
            fitView
            // Cap the fit zoom at 1× so node text renders at its true px size
            // (matching the left-panel font); without this, a sparse graph
            // fit-to-view magnifies the nodes and the text looks oversized.
            fitViewOptions={{ maxZoom: 1 }}
            minZoom={0.3}
            maxZoom={1.5}
            proOptions={{ hideAttribution: true }}
          >
            <Background />
            <Controls />
            <MiniMap pannable zoomable />
          </ReactFlow>

          {nodes.length === 0 && !catalogQuery.isLoading && !popupMessage && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <div className="rounded-md border border-dashed border-border bg-card/70 px-4 py-3 text-center text-xs text-muted-foreground">
                Drag a node from the left palette onto the canvas, or double-click one to add it.
              </div>
            </div>
          )}

          {/* Validation checklist (top-left) — why Run is disabled. One line per
              unmet requirement: unconnected inputs, empty values, bad paths. */}
          {nodes.length > 0 && validationErrors.length > 0 && (
            <div className="absolute left-2 top-2 z-10 max-h-[40%] w-max max-w-[28rem] overflow-auto rounded-md border border-red-500/40 bg-card/95 p-2 text-xs shadow-md">
              <div className="mb-1 whitespace-nowrap font-medium text-red-600 dark:text-red-400">
                ⚠ {validationErrors.length} issue{validationErrors.length > 1 ? 's' : ''} to fix before running
              </div>
              <ul className="space-y-0.5 text-muted-foreground">
                {validationErrors.map((e, i) => (
                  <li key={i} className="whitespace-nowrap leading-snug">
                    <span className="font-medium text-foreground">{e.node}</span> · {e.message}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Auto-disappearing toast (top-center) for transient confirmations.
              Critical warnings use the persistent yellow banner instead. */}
          {toast && (
            <div className="pointer-events-none absolute left-1/2 top-3 z-20 -translate-x-1/2">
              <div className="rounded-full border border-border bg-foreground/90 px-3.5 py-1.5 text-xs text-background shadow-lg">
                {toast}
              </div>
            </div>
          )}

          {/* Transient AI spinner popup — generating a workflow or finding a transform.
              Sits a bit above center (pb-28) so it doesn't collide with centered content. */}
          {popupMessage && (
            <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center pb-28">
              <div className="flex items-center gap-2.5 rounded-lg border border-border bg-card px-4 py-2.5 text-xs text-foreground shadow-lg">
                <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-blue-500/30 border-t-blue-500" />
                {popupMessage}
              </div>
            </div>
          )}
        </div>

        <NodeParamPanel
          node={selectedNode}
          workflowName={workflowName}
          onChangeWorkflowName={setWorkflowName}
          experimentName={experimentName}
          onChangeExperimentName={setExperimentName}
          runName={runName}
          onChangeRunName={setRunName}
          onChangeParam={onChangeParam}
          onRename={onRename}
          onDelete={onDeleteSelected}
        />
      </div>
    </div>
  )
}

export function VortexTab() {
  // ReactFlowProvider gives child components access to the flow instance.
  return (
    <ReactFlowProvider>
      <VortexCanvas />
    </ReactFlowProvider>
  )
}

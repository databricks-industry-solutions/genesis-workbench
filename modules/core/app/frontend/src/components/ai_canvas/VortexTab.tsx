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
  nextNodeId,
  portsCompatible,
  toCanvasGraph,
} from './graph'
import type { VortexEdge, VortexNode } from './graph'
import type { CanvasGraph, CanvasNodeType } from '@/types/api'

const nodeTypes: NodeTypes = { vortex: CanvasNode }

// Fun, catalog-valid example goals for the ✨ "Show me how" button. Each leans on
// a Prebuilt Workflow (Guided Enzyme/Molecule Optimization, Protein Design, ADMET
// Screen, AlphaFold, GWAS) so the generated graph reads as a real pipeline.
const EXAMPLE_GOALS = [
  'Guided enzyme optimization on a motif, then fold the best candidate and check solubility',
  'Design a protein around a target with Protein Design, then predict thermostability',
  'Run an ADMET screen on a molecule, then guided molecule optimization on the winners',
  'Optimize an enzyme for a substrate and validate the top design with AlphaFold',
  'Predict structure with AlphaFold, dock a ligand, and screen it for toxicity',
  'Call variants from a FASTQ, run GWAS, and annotate the pathogenic hits',
  'Generate molecules, ADMET-screen them, then optimize the best for a target',
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
  // `error` toasts render red (e.g. an invalid connection attempt).
  const [toast, setToast] = useState<{ msg: string; error: boolean } | null>(null)
  const toastTimer = useRef<number | undefined>(undefined)
  const showToast = useCallback((msg: string, error = false) => {
    setToast({ msg, error })
    if (toastTimer.current) window.clearTimeout(toastTimer.current)
    toastTimer.current = window.setTimeout(() => setToast(null), error ? 4500 : 3500)
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
  // Input ports of the selected node that are fed by an edge (their inline editor
  // is disabled — the upstream value wins).
  const wiredInputs = useMemo(
    () =>
      new Set(
        edges.filter((e) => e.target === selectedId).map((e) => e.targetHandle ?? ''),
      ),
    [edges, selectedId],
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

  // Streamed generation: the model's plan arrives as `thought` events (shown as a
  // live feed) then a `result` event carrying the graph. fetch + manual SSE framing
  // (EventSource is GET-only) — mirrors useSseMutation.
  const [generating, setGenerating] = useState(false)
  const [genThoughts, setGenThoughts] = useState<string[]>([])
  const genCtrl = useRef<AbortController | null>(null)
  const runGenerate = useCallback(
    (goalText: string) => {
      const g = goalText.trim()
      if (!g) return
      genCtrl.current?.abort()
      const ctrl = new AbortController()
      genCtrl.current = ctrl
      setGenerating(true)
      setGenThoughts([])
      setNotice(null)
      ;(async () => {
        try {
          const res = await fetch('/api/ai_canvas/generate/stream', {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
            body: JSON.stringify({ goal: g }),
            signal: ctrl.signal,
          })
          if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)
          const reader = res.body.getReader()
          const dec = new TextDecoder()
          let buf = ''
          while (true) {
            const { done, value } = await reader.read()
            if (done) break
            buf += dec.decode(value, { stream: true })
            let sep
            while ((sep = buf.indexOf('\n\n')) !== -1) {
              const frame = buf.slice(0, sep)
              buf = buf.slice(sep + 2)
              if (!frame || frame.startsWith(':')) continue
              let event = 'message'
              let dataLine = ''
              for (const line of frame.split('\n')) {
                if (line.startsWith('event:')) event = line.slice(6).trim()
                else if (line.startsWith('data:')) dataLine += line.slice(5).trim()
              }
              if (!dataLine) continue
              let payload: { text?: string; message?: string; nodes?: unknown[] }
              try {
                payload = JSON.parse(dataLine)
              } catch {
                continue
              }
              if (event === 'thought' && payload.text) {
                setGenThoughts((t) => [...t, payload.text as string])
              } else if (event === 'result') {
                loadGraph(payload as CanvasGraph)
                setLoadedId(null)
                setGenerating(false)
                genCtrl.current = null
                const n = (payload.nodes as unknown[] | undefined)?.length ?? 0
                showToast(
                  n
                    ? `Generated a ${n}-node workflow — edit it, then Run.`
                    : 'No nodes generated. Try rephrasing your goal.',
                )
                return
              } else if (event === 'error') {
                setNotice(payload.message ?? 'Generation failed.')
                setGenerating(false)
                genCtrl.current = null
                return
              }
            }
          }
          if (genCtrl.current === ctrl) {
            setGenerating(false)
            genCtrl.current = null
          }
        } catch (e) {
          if ((e as Error).name === 'AbortError') return
          setNotice((e as Error).message)
          setGenerating(false)
          genCtrl.current = null
        }
      })()
    },
    [loadGraph, showToast],
  )

  // ✨ "Show me how" — drop in a random fun goal and generate it.
  const tryRandom = useCallback(() => {
    const g = EXAMPLE_GOALS[Math.floor(Math.random() * EXAMPLE_GOALS.length)]
    setGoal(g)
    runGenerate(g)
  }, [runGenerate])

  // ── add a node ─────────────────────────────────────────────────────────────
  // Convertible fields: a node's input ports are editable inline (right panel) or
  // wired from upstream — so a dropped node is self-contained, no auto-spawned IO
  // nodes. Seed inline `inputs` empty; the user types values or draws edges.
  const addNode = useCallback(
    (cat: CanvasNodeType, position: { x: number; y: number }) => {
      const id = nextNodeId(cat.type)
      const node: VortexNode = {
        id,
        type: 'vortex',
        position,
        data: { typeKey: cat.type, label: cat.label, params: defaultParams(cat), inputs: {}, catalog: cat },
      }
      setNodes((nds) => [...nds, node])
      setSelectedId(id)
    },
    [setNodes],
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
              showToast(fail, true)
            }
          })
          .catch(() => {
            setSuggesting(null)
            showToast(fail, true)
          })
        return
      }
      setNotice(null)
      setEdges((eds) => addEdge({ ...conn, animated: false }, eds))
      // If the source is a still-unnamed IO node (Text/Volume/Delta Input), name
      // it after the field it now feeds — so wired inputs are self-describing.
      if (
        src?.data.catalog?.category === 'io' &&
        dstPort &&
        src.data.label === src.data.catalog?.label
      ) {
        const newLabel = dstPort.label || dstPort.name
        setNodes((nds) =>
          nds.map((n) => (n.id === conn.source ? { ...n, data: { ...n.data, label: newLabel } } : n)),
        )
      }
    },
    [nodes, setNodes, setEdges, catalogByType, insertTransform],
  )

  // Double-click an edge to remove it (plus Backspace/Delete on a selected edge,
  // enabled via deleteKeyCode on the canvas).
  const onEdgeDoubleClick = useCallback(
    (_: React.MouseEvent, edge: VortexEdge) => {
      setEdges((eds) => eds.filter((e) => e.id !== edge.id))
    },
    [setEdges],
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
  // Inline value for an input port (convertible fields).
  const onChangeInput = useCallback(
    (name: string, value: unknown) =>
      patchSelected((n) => ({ ...n, data: { ...n.data, inputs: { ...n.data.inputs, [name]: value } } })),
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
  // Per-node validity drives the red/green border. Inject as derived node data so
  // CanvasNode can render it without us mutating the canonical `nodes` state.
  const invalidIds = useMemo(
    () => new Set(validationErrors.map((e) => e.nodeId)),
    [validationErrors],
  )
  // Per-node set of input ports fed by an edge (so CanvasNode shows a handle only
  // while a field is empty AND unwired).
  const connectedByNode = useMemo(() => {
    const m = new Map<string, string[]>()
    for (const e of edges) {
      if (!e.target) continue
      const arr = m.get(e.target) ?? []
      arr.push(e.targetHandle ?? '')
      m.set(e.target, arr)
    }
    return m
  }, [edges])
  const flowNodes = useMemo(
    () =>
      nodes.map((n) => ({
        ...n,
        data: {
          ...n.data,
          invalid: invalidIds.has(n.id),
          connectedInputs: connectedByNode.get(n.id) ?? [],
        },
      })),
    [nodes, invalidIds, connectedByNode],
  )
  const runDisabledReason =
    nodes.length === 0
      ? 'Add a node to begin'
      : validationErrors.length > 0
        ? `Fix ${validationErrors.length} issue(s) before running`
        : ''

  // Centered spinner popup — AI generation / transform lookup only (runs are
  // tracked in Past Runs, not with a live spinner).
  // Centered spinner popup: transform-lookup only. Generation has its own live
  // thoughts panel (rendered separately while `generating`).
  const popupMessage = suggesting

  // Yellow banner: the latest notice (dispatch confirmation, errors, warnings).
  const banner = notice

  return (
    <div className="flex h-[86vh] min-h-[640px] flex-col overflow-hidden rounded-md border border-border">
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
            runGenerate(goal)
          }}
        >
          <button
            type="button"
            onClick={tryRandom}
            disabled={generating}
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
            disabled={!goal.trim() || generating}
            className="shrink-0 rounded-md border border-primary/50 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary hover:bg-primary/20 disabled:opacity-40"
          >
            {generating ? 'Generating…' : 'Generate workflow'}
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
            nodes={flowNodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onEdgeDoubleClick={onEdgeDoubleClick}
            deleteKeyCode={['Backspace', 'Delete']}
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

          {nodes.length === 0 && !catalogQuery.isLoading && !popupMessage && !generating && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <div className="rounded-md border border-dashed border-border bg-card/70 px-4 py-3 text-center text-xs text-muted-foreground">
                Drag a node from the left palette onto the canvas, or double-click one to add it.
              </div>
            </div>
          )}

          {/* Live "thinking" feed while the AI designs the workflow — the model's
              plan streams in as bullets, then the graph renders. */}
          {generating && (
            <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center pb-28">
              <div className="w-80 max-w-[80%] rounded-lg border border-border bg-card px-4 py-3 text-xs shadow-lg">
                <div className="mb-2 flex items-center gap-2 font-medium text-foreground">
                  <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-blue-500/30 border-t-blue-500" />
                  ✨ Designing workflow…
                </div>
                {genThoughts.length === 0 ? (
                  <div className="italic text-muted-foreground">Thinking through the goal…</div>
                ) : (
                  <ul className="space-y-1 text-muted-foreground">
                    {genThoughts.map((t, i) => (
                      <li key={i} className="leading-snug">• {t}</li>
                    ))}
                  </ul>
                )}
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
              <div
                className={
                  'rounded-full px-3.5 py-1.5 text-xs shadow-lg ' +
                  (toast.error
                    ? 'bg-red-600 font-medium text-white'
                    : 'border border-border bg-foreground/90 text-background')
                }
              >
                {toast.msg}
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
          onChangeInput={onChangeInput}
          wiredInputs={wiredInputs}
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

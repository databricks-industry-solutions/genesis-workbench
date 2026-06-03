// Vortex (ai_canvas) — AI-assisted drag-and-drop workflow canvas.
//
// Visible name "Vortex"; feature id `ai_canvas`. This tab lets a user compose a
// workflow from deployed endpoints, batch jobs, and data-IO nodes, then run it
// (run dispatch + AI generation + save/load arrive in later increments).
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
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
  defaultParams,
  fromCanvasGraph,
  nextNodeId,
  portsCompatible,
  toCanvasGraph,
} from './graph'
import type { VortexEdge, VortexNode, VortexNodeData } from './graph'
import type { CanvasNodeType } from '@/types/api'

const nodeTypes: NodeTypes = { vortex: CanvasNode }

function VortexCanvas() {
  const theme = useThemeStore((s) => s.theme)
  const wrapperRef = useRef<HTMLDivElement>(null)
  const rfRef = useRef<ReactFlowInstance<VortexNode, VortexEdge> | null>(null)

  const [nodes, setNodes, onNodesChange] = useNodesState<VortexNode>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<VortexEdge>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [goal, setGoal] = useState('')
  // Track the loaded workflow so Save updates it (vs. always creating a copy).
  const [loadedId, setLoadedId] = useState<number | null>(null)
  const [loadedName, setLoadedName] = useState('')
  // MLflow run id of the in-flight dispatch (drives the live status overlay).
  const [activeRunId, setActiveRunId] = useState<string | null>(null)

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
      setNodes(ns)
      setEdges(es.map((e) => ({ ...e, animated: true })))
      setSelectedId(null)
    },
    [catalogByType, setNodes, setEdges],
  )

  const generate = useMutation({
    mutationFn: (g: string) => api.aiCanvasGenerate({ goal: g }),
    onSuccess: (res) => {
      loadGraph(res.graph)
      setLoadedId(null) // a generated graph is new/unsaved
      setLoadedName('')
      setNotice(
        res.graph.nodes.length
          ? `Generated a ${res.graph.nodes.length}-node workflow — edit it, then Run.`
          : 'No nodes generated. Try rephrasing your goal.',
      )
    },
    onError: (err: Error) => setNotice(err.message),
  })

  // ── add a node ─────────────────────────────────────────────────────────────
  const addNode = useCallback(
    (cat: CanvasNodeType, position: { x: number; y: number }) => {
      const id = nextNodeId(cat.type)
      const node: VortexNode = {
        id,
        type: 'vortex',
        position,
        data: { typeKey: cat.type, label: cat.label, params: defaultParams(cat), catalog: cat },
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

  // ── connect edges with dtype validation ─────────────────────────────────────
  const onConnect = useCallback(
    (conn: Connection) => {
      const src = nodes.find((n) => n.id === conn.source)
      const dst = nodes.find((n) => n.id === conn.target)
      const srcPort = src?.data.catalog?.outputs.find((p) => p.name === conn.sourceHandle)
      const dstPort = dst?.data.catalog?.inputs.find((p) => p.name === conn.targetHandle)
      if (srcPort && dstPort && !portsCompatible(srcPort.dtype, dstPort.dtype)) {
        setNotice(
          `Can't connect ${srcPort.dtype} → ${dstPort.dtype}. Outputs must match the target input type.`,
        )
        return
      }
      setNotice(null)
      setEdges((eds) => addEdge({ ...conn, animated: true }, eds))
    },
    [nodes, setEdges],
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
    setLoadedName('')
  }, [setNodes, setEdges])

  // ── run dispatch + live per-node status overlay ──────────────────────────────
  const run = useMutation({
    mutationFn: () =>
      api.aiCanvasRun({
        graph: toCanvasGraph(nodes, edges),
        run_name: loadedName || 'ai_canvas_run',
      }),
    onSuccess: (res) => {
      setActiveRunId(res.mlflow_run_id)
      setNotice('Workflow dispatched — running…')
    },
    onError: (err: Error) => setNotice(err.message),
  })

  const runStatus = useQuery({
    queryKey: ['ai_canvas', 'run-status', activeRunId],
    queryFn: () => api.aiCanvasRunStatus(activeRunId as string),
    enabled: activeRunId !== null,
    // Poll while the orchestrator is in flight; stop once terminal.
    refetchInterval: (q) => {
      const s = q.state.data?.job_status
      return s === 'complete' || s === 'failed' ? false : 4000
    },
  })

  // Sync per-node statuses from the poll onto the canvas nodes (external → React).
  const statusData = runStatus.data
  useEffect(() => {
    if (!statusData) return
    setNodes((nds) =>
      nds.map((n) => {
        const s = statusData.node_status[n.id] as VortexNodeData['status'] | undefined
        return s ? { ...n, data: { ...n.data, status: s } } : n
      }),
    )
  }, [statusData, setNodes])

  // Run banner is derived during render (not stored) so we don't setState in the effect.
  const runMessage =
    activeRunId && statusData
      ? statusData.job_status === 'complete'
        ? '✅ Workflow complete.'
        : statusData.job_status === 'failed'
          ? '❌ Workflow failed — see Past runs for details.'
          : `Running… ${Object.values(statusData.node_status).filter((s) => s === 'complete').length}/${nodes.length} nodes done`
      : null

  return (
    <div className="flex h-[70vh] min-h-[520px] flex-col overflow-hidden rounded-md border border-border">
      {/* Toolbar */}
      <div className="flex flex-col gap-2 border-b border-border bg-card/60 px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">Vortex</span>
          <span className="text-xs text-muted-foreground">
            Describe a goal and let AI draft the workflow, or build it by hand.
          </span>
          <div className="ml-auto flex items-center gap-2">
            <WorkflowLibrary
              graph={toCanvasGraph(nodes, edges)}
              loadedId={loadedId}
              loadedName={loadedName}
              disabled={nodes.length === 0}
              onLoad={(detail) => {
                loadGraph(detail.graph)
                setLoadedId(detail.workflow_id)
                setLoadedName(detail.name)
                setNotice(`Loaded "${detail.name}".`)
              }}
              onSaved={(id, name) => {
                setLoadedId(id)
                setLoadedName(name)
                setNotice(`Saved "${name}".`)
              }}
            />
            <RunHistory />
            <button
              onClick={clearCanvas}
              disabled={nodes.length === 0}
              className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-accent disabled:opacity-40"
            >
              Clear
            </button>
            <button
              onClick={() => run.mutate()}
              disabled={nodes.length === 0 || run.isPending}
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
          <span className="text-sm">✨</span>
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

      {(runMessage ?? notice) && (
        <div className="border-b border-amber-500/40 bg-amber-500/10 px-3 py-1.5 text-xs text-amber-700 dark:text-amber-400">
          {runMessage ?? notice}
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        {catalogQuery.isLoading ? (
          <div className="flex w-56 items-center justify-center border-r border-border text-xs text-muted-foreground">
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
            proOptions={{ hideAttribution: true }}
          >
            <Background />
            <Controls />
            <MiniMap pannable zoomable />
          </ReactFlow>

          {nodes.length === 0 && !catalogQuery.isLoading && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <div className="rounded-md border border-dashed border-border bg-card/70 px-4 py-3 text-center text-xs text-muted-foreground">
                Drag a node from the left palette onto the canvas, or double-click one to add it.
              </div>
            </div>
          )}
        </div>

        <NodeParamPanel
          node={selectedNode}
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

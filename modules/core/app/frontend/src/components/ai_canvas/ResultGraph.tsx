// Vortex (ai_canvas) — read-only canvas for a past run's result: renders the
// saved graph with each node tinted by its run status (green=complete,
// red=failed, grey=pending/skipped). Pan/zoom only; not editable.
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Background,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesInitialized,
  useNodesState,
} from '@xyflow/react'
import type { NodeTypes, ReactFlowInstance } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { useThemeStore } from '@/stores/theme'
import { CanvasNode } from './CanvasNode'
import { GraphJson } from './GraphJson'
import { fromCanvasGraph } from './graph'
import type { NodeStatus, VortexEdge, VortexNode } from './graph'
import type { CanvasGraph, CanvasNodeType } from '@/types/api'

const nodeTypes: NodeTypes = { vortex: CanvasNode }

function Inner({
  graph,
  nodeStatus,
}: {
  graph: CanvasGraph
  nodeStatus: Record<string, string>
}) {
  const theme = useThemeStore((s) => s.theme)
  const catalog = useQuery({ queryKey: ['ai_canvas', 'catalog'], queryFn: () => api.aiCanvasCatalog() })
  const catalogByType = useMemo(
    () => new Map<string, CanvasNodeType>((catalog.data?.nodes ?? []).map((n) => [n.type, n])),
    [catalog.data],
  )

  // useNodesState (with onNodesChange) so React Flow measures the custom nodes —
  // a static `nodes` prop inside this modal Dialog mis-measures and collapses the
  // graph to ~one node. Edges are applied after nodes are measured (same fix as
  // the editor canvas), then we fit the view.
  const [nodes, setNodes, onNodesChange] = useNodesState<VortexNode>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<VortexEdge>([])
  const [pendingEdges, setPendingEdges] = useState<VortexEdge[] | null>(null)
  const rfRef = useRef<ReactFlowInstance<VortexNode, VortexEdge> | null>(null)

  useEffect(() => {
    const g = fromCanvasGraph(graph, catalogByType)
    // A node with no status tag never ran (downstream of a failure) → "pending"
    // renders grey via CanvasNode's status ring.
    const ns = g.nodes.map((n) => ({
      ...n,
      data: { ...n.data, status: (nodeStatus[n.id] as NodeStatus | undefined) ?? 'pending' },
    }))
    setNodes(ns)
    setEdges([])
    setPendingEdges(g.edges)
  }, [graph, catalogByType, nodeStatus, setNodes, setEdges])

  const nodesInitialized = useNodesInitialized()
  useEffect(() => {
    if (nodesInitialized && pendingEdges) {
      setEdges(pendingEdges)
      setPendingEdges(null)
      rfRef.current?.fitView({ maxZoom: 1, duration: 200 })
    }
  }, [nodesInitialized, pendingEdges, setEdges])

  return (
    <div className="relative h-[55vh] w-full overflow-hidden rounded-md border border-border">
      {/* View / copy this run's graph JSON — floating top-right, same as the editor canvas. */}
      <div className="absolute right-2 top-2 z-10">
        <GraphJson graph={graph} />
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onInit={(inst) => (rfRef.current = inst)}
        nodeTypes={nodeTypes}
        colorMode={theme}
        fitView
        fitViewOptions={{ maxZoom: 1 }}
        minZoom={0.2}
        maxZoom={1.5}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}

export function ResultGraph(props: { graph: CanvasGraph; nodeStatus: Record<string, string> }) {
  return (
    <ReactFlowProvider>
      <Inner {...props} />
    </ReactFlowProvider>
  )
}

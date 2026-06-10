// Vortex (ai_canvas) — read-only canvas for a past run's result: renders the
// saved graph with each node tinted by its run status (green=complete,
// red=failed, grey=pending/skipped). Pan/zoom only; not editable.
import { useMemo } from 'react'
import { ReactFlow, ReactFlowProvider, Background, Controls } from '@xyflow/react'
import type { NodeTypes } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { useThemeStore } from '@/stores/theme'
import { CanvasNode } from './CanvasNode'
import { fromCanvasGraph } from './graph'
import type { NodeStatus } from './graph'
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
  const { nodes, edges } = useMemo(() => {
    const g = fromCanvasGraph(graph, catalogByType)
    // A node with no status tag never ran (downstream of a failure) → "pending"
    // renders grey via CanvasNode's status ring.
    const ns = g.nodes.map((n) => ({
      ...n,
      data: { ...n.data, status: (nodeStatus[n.id] as NodeStatus | undefined) ?? 'pending' },
    }))
    return { nodes: ns, edges: g.edges }
  }, [graph, catalogByType, nodeStatus])

  return (
    <div className="h-[55vh] w-full overflow-hidden rounded-md border border-border">
      <ReactFlow
        nodes={nodes}
        edges={edges}
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

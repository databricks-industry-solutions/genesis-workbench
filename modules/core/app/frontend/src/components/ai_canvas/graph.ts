// Vortex (ai_canvas) — shared types + graph <-> React Flow conversion helpers.
import type { Edge, Node } from '@xyflow/react'

import type {
  CanvasGraph,
  CanvasNodeType,
} from '@/types/api'

// Data carried on each React Flow node. `catalog` is the node-type definition
// (ports, params) so the custom node + param panel can render without a lookup.
export type VortexNodeData = {
  typeKey: string
  label: string
  params: Record<string, unknown>
  catalog: CanvasNodeType | null
  status?: NodeStatus
}

export type VortexNode = Node<VortexNodeData>
export type VortexEdge = Edge

export type NodeStatus = 'pending' | 'running' | 'complete' | 'failed'

// Visual accent per node category — reuses the app's semantic tokens.
export const CATEGORY_STYLE: Record<
  CanvasNodeType['category'],
  { ring: string; chip: string; label: string }
> = {
  endpoint: { ring: 'border-blue-500/60', chip: 'bg-blue-500/15 text-blue-500', label: 'Endpoint' },
  batch: { ring: 'border-amber-500/60', chip: 'bg-amber-500/15 text-amber-600', label: 'Batch job' },
  io: { ring: 'border-emerald-500/60', chip: 'bg-emerald-500/15 text-emerald-600', label: 'Data I/O' },
}

// Coarse dtype color so handles of compatible ports read as the same color.
export function dtypeColor(dtype: string): string {
  switch (dtype) {
    case 'sequence':
    case 'sequences':
      return '#3b82f6' // blue
    case 'pdb':
      return '#a855f7' // purple
    case 'smiles':
      return '#f59e0b' // amber
    case 'table':
    case 'path':
      return '#10b981' // emerald
    case 'embedding':
    case 'score':
      return '#ec4899' // pink
    default:
      return '#94a3b8' // slate / any
  }
}

// An edge is valid when the source-output dtype matches the target-input dtype,
// or either side is `any`. Mirrors the backend PortType compatibility note.
export function portsCompatible(srcDtype: string, dstDtype: string): boolean {
  if (srcDtype === 'any' || dstDtype === 'any') return true
  if (srcDtype === dstDtype) return true
  // a single value flows into a list-typed port and vice-versa
  const norm = (d: string) => d.replace(/s$/, '')
  return norm(srcDtype) === norm(dstDtype)
}

let _counter = 0
export function nextNodeId(typeKey: string): string {
  _counter += 1
  return `${typeKey}-${_counter}`
}

// Default param map for a freshly-dropped node, seeded from catalog defaults.
export function defaultParams(cat: CanvasNodeType): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const p of cat.params) {
    if (p.default !== null && p.default !== undefined) out[p.name] = p.default
  }
  return out
}

// React Flow nodes/edges -> the persisted/executable graph JSON.
export function toCanvasGraph(nodes: VortexNode[], edges: VortexEdge[]): CanvasGraph {
  return {
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.data.typeKey,
      label: n.data.label,
      params: n.data.params ?? {},
      position: { x: Math.round(n.position.x), y: Math.round(n.position.y) },
    })),
    edges: edges.map((e) => ({
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle ?? null,
      targetHandle: e.targetHandle ?? null,
    })),
  }
}

// Persisted/generated graph JSON -> React Flow nodes/edges.
export function fromCanvasGraph(
  graph: CanvasGraph,
  catalogByType: Map<string, CanvasNodeType>,
): { nodes: VortexNode[]; edges: VortexEdge[] } {
  const nodes: VortexNode[] = graph.nodes.map((n) => {
    const cat = catalogByType.get(n.type) ?? null
    return {
      id: n.id,
      type: 'vortex',
      position: n.position ?? { x: 0, y: 0 },
      data: {
        typeKey: n.type,
        label: n.label || cat?.label || n.type,
        params: n.params ?? {},
        catalog: cat,
      },
    }
  })
  const edges: VortexEdge[] = graph.edges.map((e, i) => ({
    id: `e-${i}-${e.source}-${e.target}`,
    source: e.source,
    target: e.target,
    sourceHandle: e.sourceHandle ?? undefined,
    targetHandle: e.targetHandle ?? undefined,
  }))
  return { nodes, edges }
}

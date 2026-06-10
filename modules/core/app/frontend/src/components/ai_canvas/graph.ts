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

// Visual accent per node group (category). One source of truth for the palette
// section headers AND the on-canvas node color + icon. `icon` is a Material
// Symbols name — keep it in the `icon_names=` subset in index.html.
export const CATEGORY_STYLE: Record<
  CanvasNodeType['category'],
  { label: string; icon: string; border: string; band: string; ring: string; iconColor: string }
> = {
  io: {
    label: 'Data',
    icon: 'dataset',
    border: 'border-emerald-500/50',
    band: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
    ring: 'border-emerald-500/60',
    iconColor: 'text-emerald-600 dark:text-emerald-400',
  },
  endpoint: {
    label: 'Serving Endpoints',
    icon: 'bolt',
    border: 'border-blue-500/50',
    band: 'bg-blue-500/15 text-blue-700 dark:text-blue-300',
    ring: 'border-blue-500/60',
    iconColor: 'text-blue-600 dark:text-blue-400',
  },
  batch: {
    label: 'Prebuilt Workflows',
    icon: 'account_tree',
    border: 'border-amber-500/50',
    band: 'bg-amber-500/15 text-amber-700 dark:text-amber-300',
    ring: 'border-amber-500/60',
    iconColor: 'text-amber-600 dark:text-amber-400',
  },
  transform: {
    label: 'Transforms',
    icon: 'transform',
    border: 'border-violet-500/50',
    band: 'bg-violet-500/15 text-violet-700 dark:text-violet-300',
    ring: 'border-violet-500/60',
    iconColor: 'text-violet-600 dark:text-violet-400',
  },
}

// Format a param value for the compact on-canvas body (single line, no wrap).
export function formatParamValue(v: unknown): string {
  if (v === null || v === undefined || v === '') return '—'
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  if (Array.isArray(v)) return v.join(', ')
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
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

// Left→right auto-layout by longest-path depth (mirrors the backend
// `_auto_layout`): x by topological depth, y stacked per depth. Dependency-free.
export function autoLayout(nodes: VortexNode[], edges: VortexEdge[]): VortexNode[] {
  const ids = new Set(nodes.map((n) => n.id))
  const succ = new Map<string, string[]>()
  const indeg = new Map<string, number>()
  nodes.forEach((n) => {
    succ.set(n.id, [])
    indeg.set(n.id, 0)
  })
  edges.forEach((e) => {
    if (ids.has(e.source) && ids.has(e.target)) {
      succ.get(e.source)!.push(e.target)
      indeg.set(e.target, (indeg.get(e.target) ?? 0) + 1)
    }
  })
  const depth = new Map<string, number>(nodes.map((n) => [n.id, 0]))
  const queue = nodes.filter((n) => (indeg.get(n.id) ?? 0) === 0).map((n) => n.id)
  while (queue.length) {
    const id = queue.shift()!
    for (const s of succ.get(id) ?? []) {
      depth.set(s, Math.max(depth.get(s) ?? 0, (depth.get(id) ?? 0) + 1))
      indeg.set(s, (indeg.get(s) ?? 0) - 1)
      if ((indeg.get(s) ?? 0) === 0) queue.push(s)
    }
  }
  const rowByDepth = new Map<number, number>()
  return nodes.map((n) => {
    const d = depth.get(n.id) ?? 0
    const row = rowByDepth.get(d) ?? 0
    rowByDepth.set(d, row + 1)
    return { ...n, position: { x: d * 240, y: row * 120 + 20 } }
  })
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

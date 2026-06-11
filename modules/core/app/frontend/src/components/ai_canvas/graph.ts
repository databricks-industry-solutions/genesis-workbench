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
  // Inline values for input ports (convertible fields). A wired edge to a port
  // overrides its inline value at run time.
  inputs?: Record<string, unknown>
  // Derived for display: input-port names currently fed by an edge. An input's
  // connection handle shows only while the field is empty AND unwired.
  connectedInputs?: string[]
  catalog: CanvasNodeType | null
  status?: NodeStatus
  // Derived for display: true when this node has unmet validation requirements
  // (unconnected input / empty required value). Drives the red/green node border.
  invalid?: boolean
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

// A UC Volume `path` is a file reference and can supply any file-backed input
// (a .pdb → PDB port, .fasta → sequence, .json/.csv → json/table). Mirrors the
// backend _PATH_FEEDS.
const PATH_FEEDS = new Set(['pdb', 'sequence', 'sequences', 'fasta', 'json', 'table'])

// An edge is valid when the source-output dtype matches the target-input dtype,
// or either side is `any`. Mirrors the backend PortType compatibility note.
export function portsCompatible(srcDtype: string, dstDtype: string): boolean {
  if (srcDtype === 'any' || dstDtype === 'any') return true
  if (srcDtype === dstDtype) return true
  if (srcDtype === 'path' && PATH_FEEDS.has(dstDtype)) return true
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

// The IO source node-type that feeds a given input dtype — mirrors the
// orchestrator's io_kind handling: `path` reads from a UC Volume, `table` from
// a Delta table, everything else is a literal the user types into a Text Input.
export function ioTypeForDtype(dtype: string): string {
  if (dtype === 'path') return 'volume_input'
  if (dtype === 'table') return 'delta_input'
  return 'text_input'
}

// A single reason the graph can't run yet, attributed to a node (id + label).
export type ValidationIssue = { nodeId: string; node: string; message: string }

function isBlank(v: unknown): boolean {
  return v === null || v === undefined || (typeof v === 'string' && v.trim() === '')
}

// Everything that must be fixed before a workflow can run, in order: each node's
// input ports must be connected, required params filled, and IO sources must hold
// a valid value (non-empty literal, /Volumes/ path, catalog.schema.table). Empty
// list ⇒ runnable. Drives both the Run-button gate and the on-canvas error list.
export function graphValidationErrors(nodes: VortexNode[], edges: VortexEdge[]): ValidationIssue[] {
  const issues: ValidationIssue[] = []
  for (const n of nodes) {
    const cat = n.data.catalog
    const label = n.data.label
    const params = n.data.params ?? {}

    // 1. Every input field must be satisfied: wired from upstream OR filled inline
    //    (convertible fields — a value can be typed on the node or piped in).
    for (const p of cat?.inputs ?? []) {
      const connected = edges.some((e) => e.target === n.id && e.targetHandle === p.name)
      const inlineSet = !isBlank(n.data.inputs?.[p.name])
      if (!connected && !inlineSet) {
        issues.push({ nodeId: n.id, node: label, message: `${p.label || p.name} is empty (type a value or wire it)` })
      }
    }

    // 2. Required params must be filled (covers IO value/path/table — all required).
    for (const pf of cat?.params ?? []) {
      if (pf.required && isBlank(params[pf.name])) {
        issues.push({ nodeId: n.id, node: label, message: `${pf.label || pf.name} is empty` })
      }
    }

    // 3. IO source format checks (only when a value is present).
    if (n.data.typeKey === 'volume_input' && !isBlank(params.path) && !String(params.path).startsWith('/Volumes/')) {
      issues.push({ nodeId: n.id, node: label, message: 'path should start with /Volumes/' })
    }
    if (n.data.typeKey === 'delta_input' && !isBlank(params.table) && String(params.table).split('.').length !== 3) {
      issues.push({ nodeId: n.id, node: label, message: 'table should be catalog.schema.table' })
    }
  }
  return issues
}

// Default param map for a freshly-dropped node, seeded from catalog defaults.
export function defaultParams(cat: CanvasNodeType): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const p of cat.params) {
    if (p.default !== null && p.default !== undefined) out[p.name] = p.default
  }
  return out
}

// Inline editor kind for an input port, derived from its dtype. Big text blobs
// (PDB / JSON / multi-sequence) get a textarea; everything else a single line.
export function inputEditorIsTextarea(dtype: string): boolean {
  return dtype === 'pdb' || dtype === 'json' || dtype === 'sequences'
}

// True when a field value is effectively empty (no inline value set).
export function fieldIsBlank(v: unknown): boolean {
  return v === null || v === undefined || (typeof v === 'string' && v.trim() === '')
}

// React Flow nodes/edges -> the persisted/executable graph JSON.
export function toCanvasGraph(nodes: VortexNode[], edges: VortexEdge[]): CanvasGraph {
  return {
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.data.typeKey,
      label: n.data.label,
      params: n.data.params ?? {},
      inputs: n.data.inputs ?? {},
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
        inputs: n.inputs ?? {},
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

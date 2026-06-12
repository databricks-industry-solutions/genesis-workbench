// Vortex (ai_canvas) — custom React Flow node.
//
// Layout: a fixed-width rounded card with a colored title band (label left,
// group icon right) and a body that lists the node's set param values — one
// truncated line each, auto-growing as more are set. The node is tinted by its
// group (category) color. Full description shows on hover (title attr).
import { Handle, Position } from '@xyflow/react'
import type { NodeProps } from '@xyflow/react'

import { cn } from '@/lib/utils'
import { MaterialIcon } from '@/components/MaterialIcon'
import { CATEGORY_STYLE, dtypeColor, fieldIsBlank, formatParamValue } from './graph'
import type { VortexNodeData } from './graph'

const STATUS_DOT: Record<string, string> = {
  pending: 'bg-muted-foreground/40',
  running: 'bg-amber-500 animate-pulse',
  complete: 'bg-emerald-500',
  failed: 'bg-destructive',
}

export function CanvasNode({ data, selected }: NodeProps) {
  const d = data as VortexNodeData
  const cat = d.catalog
  const category = cat?.category ?? 'endpoint'
  const style = CATEGORY_STYLE[category]
  const inputs = cat?.inputs ?? []
  const outputs = cat?.outputs ?? []
  const unavailable = cat ? !cat.available : false
  const connected = new Set(d.connectedInputs ?? [])

  // Only *input* ports are convertible (inline value or wired). Params stay inline
  // config. An input's handle shows while it's an "open slot" — blank inline value
  // AND not wired. Type a value → handle hides; clear it → it returns; a wired
  // input keeps its handle.
  const showInputHandle = (name: string) =>
    connected.has(name) || fieldIsBlank(d.inputs?.[name])

  // Set inline input values render in the body (like params).
  const inputRows = inputs
    .map((p) => ({ label: p.label || p.name, value: d.inputs?.[p.name] }))
    .filter((r) => !fieldIsBlank(r.value))
    .map((r) => ({ label: r.label, text: formatParamValue(r.value) }))

  // "Main param values" = params with a non-empty effective value (set or
  // defaulted). Each renders as one no-wrap, ellipsis-truncated line.
  const paramRows = (cat?.params ?? [])
    .map((p) => ({ label: p.label || p.name, value: d.params?.[p.name] ?? p.default }))
    .map((r) => ({ label: r.label, text: formatParamValue(r.value) }))
    .filter((r) => r.text !== '—')

  // Spread handles evenly along the left (inputs) / right (outputs) edges.
  const handleTop = (i: number, n: number) => `${((i + 1) / (n + 1)) * 100}%`

  // Read-only result view sets data.status → ring by run outcome (green/red/grey).
  // The editor leaves status unset → fall back to the validation ring.
  const statusRing =
    d.status === 'complete'
      ? 'ring-2 ring-emerald-500'
      : d.status === 'failed'
        ? 'ring-2 ring-red-500'
        : d.status === 'running'
          ? 'ring-2 ring-amber-500 animate-pulse' // currently executing — stand out
          : d.status
            ? 'ring-2 ring-slate-400/70' // pending (not yet run)
            : null

  return (
    <div
      title={cat?.description || d.label}
      className={cn(
        'w-32 overflow-hidden rounded-md border bg-card shadow-sm',
        style.border,
        // Run-status ring (result viewer) takes precedence; otherwise the
        // validation ring (red = unmet requirements, green = good to go).
        statusRing ?? (d.invalid ? 'ring-2 ring-red-500' : 'ring-1 ring-emerald-500/60'),
        selected && 'outline outline-2 outline-primary outline-offset-1',
        unavailable && 'opacity-60',
      )}
    >
      {/* Input handles — only for open slots (empty + unwired) or already-wired. */}
      {inputs.map((p, i) =>
        showInputHandle(p.name) ? (
          <Handle
            key={`in-${p.name}`}
            id={p.name}
            type="target"
            position={Position.Left}
            style={{ top: handleTop(i, inputs.length), background: dtypeColor(p.dtype), width: 7, height: 7 }}
            title={`${p.label} (${p.dtype})`}
          />
        ) : null,
      )}

      {/* Title band */}
      <div className={cn('flex items-center gap-1 px-1.5 py-0.5', style.band)}>
        {d.status && (
          <span className={cn('h-1.5 w-1.5 shrink-0 rounded-full', STATUS_DOT[d.status] ?? 'bg-muted')} />
        )}
        <span className="min-w-0 flex-1 truncate text-xs font-bold leading-tight">{d.label}</span>
        <MaterialIcon name={style.icon} className={cn('shrink-0 text-[12px]', style.iconColor)} />
      </div>

      {/* Body — main param values, one truncated line each */}
      <div className="px-1.5 py-0.5">
        {unavailable && (
          <div className="mb-px inline-block rounded bg-destructive/15 px-1 py-px text-[10px] font-medium text-destructive">
            not deployed
          </div>
        )}
        {inputRows.map((r) => (
          <div key={`in-${r.label}`} className="truncate text-xs leading-tight text-foreground">
            <span className="text-muted-foreground">{r.label}:</span> {r.text}
          </div>
        ))}
        {paramRows.map((r) => (
          <div key={r.label} className="truncate text-xs leading-tight text-foreground">
            <span className="text-muted-foreground">{r.label}:</span> {r.text}
          </div>
        ))}
        {inputRows.length === 0 && paramRows.length === 0 && (
          <div className="truncate text-xs italic leading-tight text-muted-foreground">
            {style.label}
          </div>
        )}
      </div>

      {/* Output handles */}
      {outputs.map((p, i) => (
        <Handle
          key={`out-${p.name}`}
          id={p.name}
          type="source"
          position={Position.Right}
          style={{ top: handleTop(i, outputs.length), background: dtypeColor(p.dtype), width: 7, height: 7 }}
          title={`${p.label} (${p.dtype})`}
        />
      ))}
    </div>
  )
}

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
import { CATEGORY_STYLE, dtypeColor, formatParamValue } from './graph'
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

  // "Main param values" = params with a non-empty effective value (set or
  // defaulted). Each renders as one no-wrap, ellipsis-truncated line.
  const paramRows = (cat?.params ?? [])
    .map((p) => ({ label: p.label || p.name, value: d.params?.[p.name] ?? p.default }))
    .map((r) => ({ label: r.label, text: formatParamValue(r.value) }))
    .filter((r) => r.text !== '—')

  // Spread handles evenly along the left (inputs) / right (outputs) edges.
  const handleTop = (i: number, n: number) => `${((i + 1) / (n + 1)) * 100}%`

  return (
    <div
      title={cat?.description || d.label}
      className={cn(
        'w-32 overflow-hidden rounded-md border bg-card shadow-sm',
        style.border,
        // Validation status ring: red when this node has unmet requirements,
        // subtle green when it's good to go. Selection uses a separate outline
        // (different CSS property) so both can show at once.
        d.invalid ? 'ring-2 ring-red-500' : 'ring-1 ring-emerald-500/60',
        selected && 'outline outline-2 outline-primary outline-offset-1',
        unavailable && 'opacity-60',
      )}
    >
      {/* Input handles */}
      {inputs.map((p, i) => (
        <Handle
          key={`in-${p.name}`}
          id={p.name}
          type="target"
          position={Position.Left}
          style={{ top: handleTop(i, inputs.length), background: dtypeColor(p.dtype), width: 7, height: 7 }}
          title={`${p.label} (${p.dtype})`}
        />
      ))}

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
        {paramRows.length > 0 ? (
          paramRows.map((r) => (
            <div key={r.label} className="truncate text-xs leading-tight text-foreground">
              <span className="text-muted-foreground">{r.label}:</span> {r.text}
            </div>
          ))
        ) : (
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

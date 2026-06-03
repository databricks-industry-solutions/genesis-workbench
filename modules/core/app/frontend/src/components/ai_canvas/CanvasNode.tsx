// Vortex (ai_canvas) — custom React Flow node with one typed handle per port.
import { Handle, Position } from '@xyflow/react'
import type { NodeProps } from '@xyflow/react'

import { cn } from '@/lib/utils'
import { CATEGORY_STYLE, dtypeColor } from './graph'
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

  // Spread handles evenly along the left (inputs) / right (outputs) edges.
  const handleTop = (i: number, n: number) => `${((i + 1) / (n + 1)) * 100}%`

  return (
    <div
      className={cn(
        'min-w-[170px] rounded-md border-2 bg-card shadow-sm',
        style.ring,
        selected && 'ring-2 ring-primary ring-offset-1 ring-offset-background',
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
          style={{ top: handleTop(i, inputs.length), background: dtypeColor(p.dtype), width: 10, height: 10 }}
          title={`${p.label} (${p.dtype})`}
        />
      ))}

      <div className="px-3 py-2">
        <div className="flex items-center gap-2">
          {d.status && (
            <span className={cn('h-2 w-2 shrink-0 rounded-full', STATUS_DOT[d.status] ?? 'bg-muted')} />
          )}
          <span className="truncate text-sm font-semibold text-foreground">{d.label}</span>
        </div>
        <div className="mt-1 flex items-center gap-1.5">
          <span className={cn('rounded px-1.5 py-0.5 text-[10px] font-medium', style.chip)}>
            {style.label}
          </span>
          {unavailable && (
            <span className="rounded bg-destructive/15 px-1.5 py-0.5 text-[10px] font-medium text-destructive">
              not deployed
            </span>
          )}
        </div>
      </div>

      {/* Output handles */}
      {outputs.map((p, i) => (
        <Handle
          key={`out-${p.name}`}
          id={p.name}
          type="source"
          position={Position.Right}
          style={{ top: handleTop(i, outputs.length), background: dtypeColor(p.dtype), width: 10, height: 10 }}
          title={`${p.label} (${p.dtype})`}
        />
      ))}
    </div>
  )
}

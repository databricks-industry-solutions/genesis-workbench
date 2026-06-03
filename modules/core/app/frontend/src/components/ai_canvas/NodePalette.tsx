// Vortex (ai_canvas) — left palette. Drag a node type onto the canvas, or
// double-click to drop it at the center.
import { useMemo, useState } from 'react'

import { cn } from '@/lib/utils'
import { CATEGORY_STYLE } from './graph'
import type { CanvasNodeType } from '@/types/api'

const CATEGORY_ORDER: CanvasNodeType['category'][] = ['io', 'endpoint', 'batch']

export function NodePalette({
  catalog,
  onAdd,
}: {
  catalog: CanvasNodeType[]
  onAdd: (nodeType: CanvasNodeType) => void
}) {
  const [filter, setFilter] = useState('')

  const grouped = useMemo(() => {
    const f = filter.trim().toLowerCase()
    const groups: Record<string, CanvasNodeType[]> = { io: [], endpoint: [], batch: [] }
    for (const n of catalog) {
      if (f && !n.label.toLowerCase().includes(f) && !n.description.toLowerCase().includes(f)) continue
      groups[n.category]?.push(n)
    }
    return groups
  }, [catalog, filter])

  return (
    <div className="flex h-full w-56 shrink-0 flex-col border-r border-border bg-card/40">
      <div className="border-b border-border p-2">
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter nodes…"
          className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs"
        />
      </div>
      <div className="flex-1 overflow-auto p-2">
        {CATEGORY_ORDER.map((cat) => {
          const items = grouped[cat]
          if (!items || items.length === 0) return null
          return (
            <div key={cat} className="mb-3">
              <div className="mb-1 px-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                {CATEGORY_STYLE[cat].label}
              </div>
              <div className="space-y-1">
                {items.map((n) => (
                  <button
                    key={n.type}
                    draggable
                    onDragStart={(e) => {
                      e.dataTransfer.setData('application/vortex-node', n.type)
                      e.dataTransfer.effectAllowed = 'move'
                    }}
                    onDoubleClick={() => onAdd(n)}
                    title={n.description + (n.available ? '' : ' — not currently deployed')}
                    className={cn(
                      'w-full cursor-grab rounded-md border bg-background px-2 py-1.5 text-left text-xs transition-colors hover:bg-accent',
                      CATEGORY_STYLE[cat].ring,
                      !n.available && 'opacity-50',
                    )}
                  >
                    <div className="truncate font-medium text-foreground">{n.label}</div>
                    {!n.available && (
                      <div className="text-[10px] text-destructive">not deployed</div>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

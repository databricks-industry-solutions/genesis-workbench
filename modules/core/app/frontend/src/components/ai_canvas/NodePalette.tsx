// Vortex (ai_canvas) — left palette. Groups (Data / Live Models / Prebuilt
// Workflows) are collapsible (expanded by default). Drag a node onto the
// canvas, or double-click to drop it at center.
import { useMemo, useState } from 'react'

import { cn } from '@/lib/utils'
import { MaterialIcon } from '@/components/MaterialIcon'
import { CATEGORY_STYLE } from './graph'
import type { CanvasNodeType } from '@/types/api'

// Prebuilt Workflows first (headline), then Serving Endpoints, Transforms, Data.
const CATEGORY_ORDER: CanvasNodeType['category'][] = ['batch', 'endpoint', 'transform', 'io']

export function NodePalette({
  catalog,
  onAdd,
}: {
  catalog: CanvasNodeType[]
  onAdd: (nodeType: CanvasNodeType) => void
}) {
  const [filter, setFilter] = useState('')
  // Collapsed groups (empty = all expanded by default).
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const toggle = (cat: string) =>
    setCollapsed((prev) => {
      const next = new Set(prev)
      next.has(cat) ? next.delete(cat) : next.add(cat)
      return next
    })

  const grouped = useMemo(() => {
    const f = filter.trim().toLowerCase()
    const groups: Record<string, CanvasNodeType[]> = { io: [], transform: [], endpoint: [], batch: [] }
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
          const style = CATEGORY_STYLE[cat]
          const isCollapsed = collapsed.has(cat)
          return (
            <div key={cat} className="mb-3">
              <button
                onClick={() => toggle(cat)}
                className="mb-1 flex w-full items-center gap-1.5 rounded px-1 py-0.5 text-left hover:bg-accent/40"
              >
                <MaterialIcon
                  name="expand_more"
                  className={cn(
                    'text-[16px] text-muted-foreground transition-transform',
                    isCollapsed && '-rotate-90',
                  )}
                />
                <MaterialIcon name={style.icon} className={cn('text-[15px]', style.iconColor)} />
                <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                  {style.label}
                </span>
                <span className="ml-auto text-[10px] text-muted-foreground/70">{items.length}</span>
              </button>
              {!isCollapsed && (
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
                        'flex w-full cursor-grab items-center gap-1.5 rounded-md border-l-2 border border-border bg-background px-2 py-1.5 text-left text-xs transition-colors hover:bg-accent',
                        style.ring,
                        !n.available && 'opacity-50',
                      )}
                    >
                      <MaterialIcon name={style.icon} className={cn('shrink-0 text-[14px]', style.iconColor)} />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate font-medium text-foreground">{n.label}</span>
                        {!n.available && (
                          <span className="block text-[10px] text-destructive">not deployed</span>
                        )}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

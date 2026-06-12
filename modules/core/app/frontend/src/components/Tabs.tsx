import { useState } from 'react'
import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

// `align: 'right'` pushes a tab to the right end of the tab bar (it still switches
// content like any other tab) — e.g. "Vortex | AI Assistant ……[Past Vortex Runs]".
type Tab = { id: string; label: string; content: ReactNode; align?: 'left' | 'right' }

export function Tabs({
  tabs,
  initial,
  rightAccessory,
}: {
  tabs: Tab[]
  initial?: string
  rightAccessory?: ReactNode
}) {
  const [active, setActive] = useState(initial ?? tabs[0]?.id)
  const renderTab = (t: Tab) => (
    <button
      key={t.id}
      onClick={() => setActive(t.id)}
      className={cn(
        'rounded-t-md px-4 py-2 text-sm transition-colors',
        active === t.id
          ? 'border-b-2 border-red-600 font-bold text-red-600 dark:border-red-500 dark:text-red-500'
          : 'text-muted-foreground hover:bg-muted hover:text-foreground',
      )}
    >
      {t.label}
    </button>
  )
  const leftTabs = tabs.filter((t) => t.align !== 'right')
  const rightTabs = tabs.filter((t) => t.align === 'right')
  return (
    <div>
      <div className="mb-4 flex items-end justify-between gap-3 border-b border-border">
        <div className="flex gap-1">{leftTabs.map(renderTab)}</div>
        <div className="flex items-center gap-1 pb-px">
          {rightTabs.map(renderTab)}
          {rightAccessory && <div className="pb-1">{rightAccessory}</div>}
        </div>
      </div>
      <div>{tabs.find((t) => t.id === active)?.content}</div>
    </div>
  )
}

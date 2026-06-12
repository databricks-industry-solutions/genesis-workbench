import { useState } from 'react'
import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

// `align: 'right'` pushes a tab to the right end of the tab bar and renders it as a
// bordered BUTTON (with an optional Material-Symbols `icon`) rather than an underline
// tab — e.g. "Vortex | AI Assistant ……[⟳ Past Vortex Runs]". It still switches
// content like any other tab.
type Tab = { id: string; label: string; content: ReactNode; align?: 'left' | 'right'; icon?: string }

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
  const renderButtonTab = (t: Tab) => (
    <button
      key={t.id}
      onClick={() => setActive(t.id)}
      className={cn(
        'flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm transition-colors',
        active === t.id
          ? 'border-red-600 bg-red-600/10 font-semibold text-red-600 dark:border-red-500 dark:text-red-500'
          : 'border-border text-muted-foreground hover:bg-muted hover:text-foreground',
      )}
    >
      {t.icon && (
        <span aria-hidden className="material-symbols-outlined text-[18px] leading-none">
          {t.icon}
        </span>
      )}
      {t.label}
    </button>
  )
  const leftTabs = tabs.filter((t) => t.align !== 'right')
  const rightTabs = tabs.filter((t) => t.align === 'right')
  return (
    <div>
      <div className="mb-4 flex items-end justify-between gap-3 border-b border-border">
        <div className="flex gap-1">{leftTabs.map(renderTab)}</div>
        <div className="flex items-center gap-2 pb-1.5">
          {rightTabs.map(renderButtonTab)}
          {rightAccessory}
        </div>
      </div>
      <div>{tabs.find((t) => t.id === active)?.content}</div>
    </div>
  )
}

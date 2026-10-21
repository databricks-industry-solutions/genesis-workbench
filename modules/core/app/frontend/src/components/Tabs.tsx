import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

// `align: 'right'` pushes a tab to the right end of the tab bar and renders it as a
// bordered BUTTON (with an optional Material-Symbols `icon`) rather than an underline
// tab — e.g. "Vortex | AI Assistant ……[⟳ Past Vortex Runs]". It still switches
// content like any other tab.
//
// `overflow: true` keeps a (left-aligned) tab OUT of the inline bar and reachable only
// via a compact `»` menu at the end of the left tabs — use it to de-crowd a bar with
// many tabs. The active overflow tab is surfaced on the `»` trigger so the current
// selection is still visible.
type Tab = {
  id: string
  label: string
  content: ReactNode
  align?: 'left' | 'right'
  icon?: string
  overflow?: boolean
}

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
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  // Close the overflow menu on an outside click or Escape.
  useEffect(() => {
    if (!menuOpen) return
    const onDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMenuOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [menuOpen])

  const select = (id: string) => {
    setActive(id)
    setMenuOpen(false)
  }

  const renderTab = (t: Tab) => (
    <button
      key={t.id}
      onClick={() => select(t.id)}
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
      onClick={() => select(t.id)}
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
  const inlineTabs = leftTabs.filter((t) => !t.overflow)
  const overflowTabs = leftTabs.filter((t) => t.overflow)
  const activeOverflow = overflowTabs.find((t) => t.id === active)

  const renderOverflowMenu = () => (
    <div ref={menuRef} className="relative self-center">
      <button
        type="button"
        onClick={() => setMenuOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={menuOpen}
        title="More workflows"
        className={cn(
          'flex items-center gap-1 rounded-md border px-2 py-1.5 text-sm transition-colors',
          activeOverflow
            ? 'border-red-600 bg-red-600/10 font-semibold text-red-600 dark:border-red-500 dark:text-red-500'
            : 'border-border text-muted-foreground hover:bg-muted hover:text-foreground',
        )}
      >
        <span aria-hidden className="material-symbols-outlined text-[18px] leading-none">
          keyboard_double_arrow_right
        </span>
        {activeOverflow && <span className="max-w-[12rem] truncate">{activeOverflow.label}</span>}
      </button>
      {menuOpen && (
        <div
          role="menu"
          className="absolute left-0 top-full z-20 mt-1 min-w-[14rem] overflow-hidden rounded-md border border-border bg-card py-1 shadow-lg"
        >
          {overflowTabs.map((t) => (
            <button
              key={t.id}
              role="menuitem"
              onClick={() => select(t.id)}
              className={cn(
                'block w-full px-3 py-2 text-left text-sm transition-colors',
                active === t.id
                  ? 'font-semibold text-red-600 dark:text-red-500'
                  : 'text-foreground hover:bg-muted',
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )

  return (
    <div>
      <div className="mb-4 flex items-end justify-between gap-3 border-b border-border">
        <div className="flex gap-1">
          {inlineTabs.map(renderTab)}
          {overflowTabs.length > 0 && renderOverflowMenu()}
        </div>
        <div className="flex items-center gap-2 pb-1.5">
          {rightTabs.map(renderButtonTab)}
          {rightAccessory}
        </div>
      </div>
      <div>{tabs.find((t) => t.id === active)?.content}</div>
    </div>
  )
}

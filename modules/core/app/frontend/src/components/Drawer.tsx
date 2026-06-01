import { useEffect } from 'react'
import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

type Props = {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
  width?: string
}

export function Drawer({ open, onClose, title, children, width = 'max-w-3xl' }: Props) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  return (
    <div
      className={cn(
        'fixed inset-0 z-50 flex justify-end transition-opacity',
        open ? 'pointer-events-auto bg-black/50 opacity-100' : 'pointer-events-none opacity-0',
      )}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
      aria-hidden={!open}
    >
      <aside
        className={cn(
          'flex h-full w-full flex-col border-l border-border bg-card shadow-2xl transition-transform duration-200',
          width,
          open ? 'translate-x-0' : 'translate-x-full',
        )}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <div className="flex items-center justify-between border-b border-border px-5 py-3">
          <h2 className="text-sm font-semibold">{title}</h2>
          <button
            onClick={onClose}
            className="rounded-md px-2 py-1 text-muted-foreground hover:bg-accent hover:text-foreground"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div className="flex-1 overflow-auto px-5 py-4">{children}</div>
      </aside>
    </div>
  )
}

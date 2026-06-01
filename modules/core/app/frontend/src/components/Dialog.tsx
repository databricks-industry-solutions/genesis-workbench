import { useEffect, useRef } from 'react'
import type { ReactNode } from 'react'

type Props = {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
  width?: string
}

export function Dialog({ open, onClose, title, children, width = 'max-w-3xl' }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        ref={ref}
        className={`w-full ${width} max-h-[90vh] overflow-auto rounded-lg border border-border bg-card shadow-xl`}
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
        <div className="px-5 py-4">{children}</div>
      </div>
    </div>
  )
}

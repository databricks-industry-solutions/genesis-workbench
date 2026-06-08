// "Paste from Clipboard" affordance — a small dropdown listing the clipboard
// items of one kind (genes, sequences, …) for the current context. Picking one
// hands it to the parent (fill a field, add a perturbation target, …).
import { useEffect, useRef, useState } from 'react'

import { MaterialIcon } from '@/components/MaterialIcon'
import { useClipboard, type ClipItem, type ClipKind } from '@/stores/clipboard'

export function ClipboardPaste({
  kind,
  onPick,
  label = 'Paste from Clipboard',
}: {
  kind: ClipKind
  onPick: (item: ClipItem) => void
  label?: string
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const items = useClipboard((s) => s.items).filter((i) => i.kind === kind)

  // Close on outside click via a document listener — NOT a full-screen overlay.
  // A `fixed inset-0` backdrop would block all page interaction if it ever got
  // stuck open; this approach can't, since there's no blocking element.
  // We track the button separately so it's excluded from the outside-close
  // (the button's own onClick toggles).
  useEffect(() => {
    if (!open) return
    const onDown = (e: Event) => {
      const t = e.target as Node
      // Clicks on the button → let its onClick toggle.
      if (buttonRef.current && buttonRef.current.contains(t)) return
      // Anything outside the whole component closes the dropdown.
      if (ref.current && !ref.current.contains(t)) setOpen(false)
    }
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    // CAPTURE phase + pointerdown: fires before any ancestor that might
    // stopPropagation() a bubble-phase mousedown (which would otherwise prevent
    // the outside-click from ever reaching us — the symptom where clicking the
    // page didn't close the dropdown but Esc did).
    document.addEventListener('pointerdown', onDown, true)
    document.addEventListener('keydown', onEsc)
    return () => {
      document.removeEventListener('pointerdown', onDown, true)
      document.removeEventListener('keydown', onEsc)
    }
  }, [open])

  return (
    <div ref={ref} className="relative inline-block text-xs">
      <button
        ref={buttonRef}
        type="button"
        onClick={() => setOpen((o) => !o)}
        disabled={items.length === 0}
        title={
          items.length === 0
            ? 'Clipboard has no items of this type yet'
            : 'Paste an item collected on your Clipboard'
        }
        className="inline-flex items-center gap-1 rounded-md border border-primary/50 bg-primary/10 px-2.5 py-1 font-medium text-primary hover:bg-primary/20 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <MaterialIcon name="assignment" className="text-[15px] text-cyan-400" />
        {label}
        {items.length > 0 && ` (${items.length})`}
      </button>

      {open && items.length > 0 && (
        <div className="absolute left-0 z-50 mt-1 max-h-64 w-64 overflow-auto rounded-md border border-border bg-card p-1 shadow-lg">
          {items.map((it) => (
            <button
              key={`${it.kind}:${it.value}`}
              type="button"
              onClick={() => {
                onPick(it)
                setOpen(false)
              }}
              title={it.value}
              className="block w-full truncate rounded px-2 py-1 text-left hover:bg-accent"
            >
              <span className="font-medium">{it.label || it.value}</span>
              {it.source && <span className="ml-1 text-[10px] text-muted-foreground">· {it.source}</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

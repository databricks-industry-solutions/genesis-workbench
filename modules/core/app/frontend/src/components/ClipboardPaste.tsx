// "Paste from Clipboard" affordance — a small dropdown listing the clipboard
// items of one kind (genes, sequences, …) for the current context. Picking one
// hands it to the parent (fill a field, add a perturbation target, …).
import { useState } from 'react'

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
  const items = useClipboard((s) => s.items).filter((i) => i.kind === kind)

  return (
    <div className="relative inline-block text-xs">
      <button
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
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} />
          <div className="absolute right-0 z-40 mt-1 max-h-64 w-64 overflow-auto rounded-md border border-border bg-card p-1 shadow-lg">
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
        </>
      )}
    </div>
  )
}

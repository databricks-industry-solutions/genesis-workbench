// App-wide "Clipboard" companion: a slide-out drawer pinned to the right edge of
// every module page. Holds the session-persisted items of interest (genes,
// sequences, …) collected from any result, in per-kind sections, ready to paste
// into another module. No server round-trip — it's a pure client-side carrier.
import { useState } from 'react'

import { MaterialIcon } from '@/components/MaterialIcon'
import { useClipboard, clipKindLabel, type ClipKind } from '@/stores/clipboard'
import { cn } from '@/lib/utils'

const SECTION_ORDER: ClipKind[] = ['gene', 'sequence']

export function ClipboardDrawer() {
  const [open, setOpen] = useState(false)
  const items = useClipboard((s) => s.items)
  const remove = useClipboard((s) => s.remove)
  const clearKind = useClipboard((s) => s.clearKind)
  const clear = useClipboard((s) => s.clear)
  const count = items.length

  const kinds = SECTION_ORDER.filter((k) => items.some((i) => i.kind === k))

  return (
    <>
      {/* Center-top trapezoid tab — hangs from the top edge (wide top, narrowing
          down), \____/ shape. */}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        title="Clipboard — your collected items of interest"
        className="fixed left-1/2 top-0 z-40 flex -translate-x-1/2 items-center gap-1.5 bg-gradient-to-b from-teal-500/30 to-indigo-500/30 px-9 pb-2.5 pt-1.5 text-sm font-bold transition-colors hover:from-teal-500/45 hover:to-indigo-500/45"
        style={{ clipPath: 'polygon(0 0, 100% 0, 86% 100%, 14% 100%)' }}
      >
        <MaterialIcon name="assignment" className="text-lg text-cyan-400" />
        <span>Clipboard</span>
        {count > 0 && (
          <span className="rounded-full bg-primary px-1.5 py-0.5 text-[10px] font-bold text-primary-foreground">
            {count}
          </span>
        )}
      </button>

      {open && (
        <div className="fixed inset-0 z-40 bg-black/30" onClick={() => setOpen(false)} />
      )}

      <aside
        className={cn(
          'fixed right-0 top-0 z-50 flex h-full w-[360px] max-w-[85vw] flex-col border-l border-border bg-card shadow-xl transition-transform duration-200',
          open ? 'translate-x-0' : 'translate-x-full',
        )}
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div>
            <h2 className="flex items-center gap-1.5 text-sm font-bold">
              <MaterialIcon name="assignment" className="text-base text-cyan-400" /> Clipboard
            </h2>
            <p className="text-[10px] text-muted-foreground">
              Items of interest · kept for this session · paste across modules
            </p>
          </div>
          <div className="flex items-center gap-2">
            {count > 0 && (
              <button
                type="button"
                onClick={clear}
                className="text-[11px] text-muted-foreground hover:text-destructive"
              >
                Clear all
              </button>
            )}
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded-md px-2 py-1 text-muted-foreground hover:bg-accent hover:text-foreground"
              aria-label="Close"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {count === 0 ? (
            <p className="text-xs text-muted-foreground">
              Nothing yet. <strong>Copy to Clipboard</strong> from any result — genes in Single
              Cell (DE / Enrichment / Trajectory), sequences from Protein Search — and paste them
              into another module.
            </p>
          ) : (
            <div className="space-y-4">
              {kinds.map((kind) => {
                const group = items.filter((i) => i.kind === kind)
                return (
                  <div key={kind}>
                    <div className="mb-1.5 flex items-center justify-between">
                      <span className="text-xs font-bold uppercase tracking-wide text-foreground">
                        {clipKindLabel(kind)} ({group.length})
                      </span>
                      <button
                        type="button"
                        onClick={() => clearKind(kind)}
                        className="text-[10px] text-muted-foreground hover:text-destructive"
                      >
                        clear
                      </button>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {group.map((it) => (
                        <button
                          key={`${it.kind}:${it.value}`}
                          type="button"
                          onClick={() => remove(it.kind, it.value)}
                          title={`${it.value}${it.source ? ` · from ${it.source}` : ''} — click to remove`}
                          className="max-w-[200px] truncate rounded border border-primary/40 bg-primary/10 px-1.5 py-0.5 text-xs text-primary hover:bg-primary/20"
                        >
                          {it.label || it.value} ✕
                        </button>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </aside>
    </>
  )
}

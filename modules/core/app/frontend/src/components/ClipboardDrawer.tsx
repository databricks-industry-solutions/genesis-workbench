// App-wide "Clipboard" companion: a center-top tab that drops a panel DOWN from
// the top. Items of interest (genes, sequences, …) live in per-kind tabs; the
// tab relevant to the current module is auto-selected and marked. Session-
// persisted, pure client-side carrier — copy from any result, paste elsewhere.
import { useState } from 'react'
import { useLocation } from 'react-router-dom'

import { MaterialIcon } from '@/components/MaterialIcon'
import { useClipboard, clipKindLabel, type ClipKind } from '@/stores/clipboard'
import { cn } from '@/lib/utils'

const SECTION_ORDER: ClipKind[] = ['gene', 'sequence']

/** Which clipboard kind is most relevant on each module route. */
function contextKind(pathname: string): ClipKind | null {
  if (pathname.startsWith('/large-molecule')) return 'sequence'
  if (pathname.startsWith('/single-cell') || pathname.startsWith('/genomics')) return 'gene'
  return null
}

export function ClipboardDrawer() {
  const [open, setOpen] = useState(false)
  const [picked, setPicked] = useState<ClipKind | null>(null)
  const items = useClipboard((s) => s.items)
  const remove = useClipboard((s) => s.remove)
  const clearKind = useClipboard((s) => s.clearKind)
  const clear = useClipboard((s) => s.clear)
  const count = items.length
  const ctx = contextKind(useLocation().pathname)

  // Tabs = kinds that have items, plus the context kind (so it's selectable as
  // the "relevant here" tab even when empty).
  const kinds = SECTION_ORDER.filter((k) => items.some((i) => i.kind === k) || k === ctx)
  // Active tab: explicit pick → else the context kind → else first available.
  const activeKind =
    (picked && kinds.includes(picked) ? picked : null) ??
    (ctx && kinds.includes(ctx) ? ctx : (kinds[0] ?? null))
  const groupItems = activeKind ? items.filter((i) => i.kind === activeKind) : []

  return (
    <>
      {/* Center-top tab — short, translucent, rounded bottom corners. */}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        title="Clipboard — your collected items of interest"
        className="fixed left-1/2 top-0 z-40 flex -translate-x-1/2 items-center gap-1.5 rounded-b-2xl border border-t-0 border-border bg-card/85 px-4 py-1 text-xs font-semibold shadow-lg ring-1 ring-black/5 backdrop-blur-md transition-colors hover:bg-card"
      >
        <MaterialIcon name="assignment" className="text-base text-cyan-400" />
        <span>Clipboard</span>
        {count > 0 && (
          <span className="rounded-full bg-primary px-1.5 text-[10px] font-bold text-primary-foreground">
            {count}
          </span>
        )}
      </button>

      {open && (
        <div className="fixed inset-0 z-40 bg-black/30" onClick={() => setOpen(false)} />
      )}

      {/* Drops DOWN from the top, centered under the tab. */}
      <aside
        className={cn(
          'fixed left-1/2 top-0 z-50 flex max-h-[70vh] w-full max-w-3xl -translate-x-1/2 flex-col rounded-b-2xl border border-t-0 border-border bg-card shadow-xl transition-transform duration-200',
          open ? 'translate-y-0' : '-translate-y-[110%]',
        )}
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
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

        {count === 0 ? (
          <p className="p-4 text-xs text-muted-foreground">
            Nothing yet. <strong>Copy to Clipboard</strong> from any result — genes in Single Cell
            (DE / Enrichment / Trajectory), sequences from Protein Search — and paste them into
            another module.
          </p>
        ) : (
          <>
            {/* Per-kind tabs */}
            <div className="flex gap-1 border-b border-border px-3 pt-2">
              {kinds.map((k) => {
                const n = items.filter((i) => i.kind === k).length
                const isActive = k === activeKind
                return (
                  <button
                    key={k}
                    type="button"
                    onClick={() => setPicked(k)}
                    className={cn(
                      'flex items-center gap-1 rounded-t-md border border-b-0 px-3 py-1.5 text-xs',
                      isActive
                        ? 'border-border bg-background font-medium text-foreground'
                        : 'border-transparent text-muted-foreground hover:text-foreground',
                    )}
                  >
                    {clipKindLabel(k)} ({n})
                    {k === ctx && (
                      <span
                        title="Most relevant on this page"
                        className="h-1.5 w-1.5 rounded-full bg-cyan-400"
                      />
                    )}
                  </button>
                )
              })}
            </div>

            {/* Active tab's items */}
            <div className="flex-1 overflow-y-auto p-4">
              {groupItems.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  No {activeKind ? clipKindLabel(activeKind).toLowerCase() : 'items'} on the
                  clipboard yet.
                </p>
              ) : (
                <>
                  <div className="mb-2 flex justify-end">
                    <button
                      type="button"
                      onClick={() => activeKind && clearKind(activeKind)}
                      className="text-[10px] text-muted-foreground hover:text-destructive"
                    >
                      clear {activeKind ? clipKindLabel(activeKind).toLowerCase() : ''}
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {groupItems.map((it) => (
                      <button
                        key={`${it.kind}:${it.value}`}
                        type="button"
                        onClick={() => remove(it.kind, it.value)}
                        title={`${it.value}${it.source ? ` · from ${it.source}` : ''} — click to remove`}
                        className="max-w-[260px] truncate rounded border border-primary/40 bg-primary/10 px-1.5 py-0.5 text-xs text-primary hover:bg-primary/20"
                      >
                        {it.label || it.value} ✕
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          </>
        )}
      </aside>
    </>
  )
}

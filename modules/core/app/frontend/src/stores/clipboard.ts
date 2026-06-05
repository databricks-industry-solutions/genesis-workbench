// App-wide, session-persisted "Clipboard" of items of interest — the cross-module
// companion. Items carry a `kind` so the drawer can show sections and modules can
// paste the relevant kind by context (genes → Perturbation; sequences → Structure
// Prediction; …). Copy from any result, paste into another module.
import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'

export type ClipKind = 'gene' | 'sequence'
export type ClipItem = {
  kind: ClipKind
  value: string
  /** Short display name (e.g. accession / hit id). Falls back to value. */
  label?: string
  /** Where it came from, e.g. "Single Cell DE" / "Protein Search". */
  source?: string
}

const KIND_LABELS: Record<ClipKind, string> = { gene: 'Genes', sequence: 'Sequences' }
export const clipKindLabel = (k: ClipKind) => KIND_LABELS[k] ?? k

// Genes are normalized uppercase (so PARP1 == parp1); sequences kept verbatim.
const normValue = (kind: ClipKind, v: string) =>
  kind === 'gene' ? v.trim().toUpperCase() : v.trim()
const keyOf = (kind: ClipKind, v: string) => `${kind}::${normValue(kind, v)}`

type ClipboardState = {
  items: ClipItem[]
  add: (item: ClipItem) => void
  addMany: (items: ClipItem[]) => void
  remove: (kind: ClipKind, value: string) => void
  toggle: (item: ClipItem) => void
  has: (kind: ClipKind, value: string) => boolean
  clearKind: (kind: ClipKind) => void
  clear: () => void
}

const norm = (item: ClipItem): ClipItem => ({ ...item, value: normValue(item.kind, item.value) })

export const useClipboard = create<ClipboardState>()(
  persist(
    (set, get) => ({
      items: [],
      add: (item) =>
        set((s) => {
          const it = norm(item)
          if (!it.value) return s
          const k = keyOf(it.kind, it.value)
          if (s.items.some((x) => keyOf(x.kind, x.value) === k)) return s
          return { items: [...s.items, it] }
        }),
      addMany: (items) =>
        set((s) => {
          const seen = new Set(s.items.map((x) => keyOf(x.kind, x.value)))
          const toAdd: ClipItem[] = []
          for (const raw of items) {
            const it = norm(raw)
            const k = keyOf(it.kind, it.value)
            if (it.value && !seen.has(k)) {
              seen.add(k)
              toAdd.push(it)
            }
          }
          return toAdd.length ? { items: [...s.items, ...toAdd] } : s
        }),
      remove: (kind, value) =>
        set((s) => ({
          items: s.items.filter((x) => keyOf(x.kind, x.value) !== keyOf(kind, value)),
        })),
      toggle: (item) =>
        set((s) => {
          const it = norm(item)
          const k = keyOf(it.kind, it.value)
          return s.items.some((x) => keyOf(x.kind, x.value) === k)
            ? { items: s.items.filter((x) => keyOf(x.kind, x.value) !== k) }
            : { items: [...s.items, it] }
        }),
      has: (kind, value) =>
        get().items.some((x) => keyOf(x.kind, x.value) === keyOf(kind, value)),
      clearKind: (kind) => set((s) => ({ items: s.items.filter((x) => x.kind !== kind) })),
      clear: () => set({ items: [] }),
    }),
    { name: 'gwb-clipboard', storage: createJSONStorage(() => sessionStorage) },
  ),
)

// The cross-module "clipboard" of genes of interest. Session-persisted (survives
// reloads within the browser session via sessionStorage) and surfaced app-wide
// through the right-side ClipboardDrawer. Genes are marked in Single Cell
// (DE / Enrichment / Trajectory) and consumed downstream (Perturbation,
// Large Molecule target hand-off).
import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'

type GeneClipboardState = {
  genes: string[]
  add: (g: string | string[]) => void
  remove: (g: string) => void
  toggle: (g: string) => void
  clear: () => void
}

const norm = (g: string) => g.trim().toUpperCase()

export const useGeneClipboard = create<GeneClipboardState>()(
  persist(
    (set) => ({
      genes: [],
      add: (g) =>
        set((s) => {
          const incoming = (Array.isArray(g) ? g : [g]).map(norm).filter(Boolean)
          return { genes: Array.from(new Set([...s.genes, ...incoming])) }
        }),
      remove: (g) => set((s) => ({ genes: s.genes.filter((x) => x !== norm(g)) })),
      toggle: (g) =>
        set((s) => {
          const n = norm(g)
          return s.genes.includes(n)
            ? { genes: s.genes.filter((x) => x !== n) }
            : { genes: [...s.genes, n] }
        }),
      clear: () => set({ genes: [] }),
    }),
    { name: 'gwb-clipboard', storage: createJSONStorage(() => sessionStorage) },
  ),
)

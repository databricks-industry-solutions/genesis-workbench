// Session-scoped "study list" of genes a scientist marks while exploring (e.g.
// from Differential Expression) to carry into another analysis (e.g.
// Perturbation). Lives in memory for the browser session only — not persisted.
import { create } from 'zustand'

type GeneClipboardState = {
  genes: string[]
  add: (g: string | string[]) => void
  remove: (g: string) => void
  toggle: (g: string) => void
  clear: () => void
}

const norm = (g: string) => g.trim().toUpperCase()

export const useGeneClipboard = create<GeneClipboardState>((set) => ({
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
}))

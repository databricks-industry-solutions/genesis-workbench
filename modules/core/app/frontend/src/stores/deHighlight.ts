// Session-scoped "highlight gene set of interest" for the Differential
// Expression view. Lifted out of the DE component into a store so other tabs
// (e.g. Pathway Enrichment) can push a pathway's genes into the DE highlight,
// and so the choice survives tab switches.
import { create } from 'zustand'

import type { Highlight } from '@/components/GeneHighlightPicker'

type DeHighlightState = {
  highlight: Highlight | null
  setHighlight: (h: Highlight | null) => void
}

export const useDeHighlight = create<DeHighlightState>((set) => ({
  highlight: null,
  setHighlight: (h) => set({ highlight: h }),
}))

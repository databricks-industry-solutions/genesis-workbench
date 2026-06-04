// Tracks the Single Cell run currently open in the Analysis view, so the
// global ClipboardDrawer can offer "save marked genes onto this run" (the
// Large Molecule hand-off) only when there's a run in context.
import { create } from 'zustand'

type ScRunContextState = {
  runId: string | null
  runLabel: string | null
  set: (runId: string | null, runLabel?: string | null) => void
}

export const useScRunContext = create<ScRunContextState>((set) => ({
  runId: null,
  runLabel: null,
  set: (runId, runLabel = null) => set({ runId, runLabel }),
}))

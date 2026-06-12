// Vortex (ai_canvas) — Past Runs as its own page: browse past workflow runs,
// inspect their workflow / inputs / outputs, and re-run with edited inputs.
import { PastRunsTab } from '@/components/ai_canvas/RunHistory'

export function VortexRunsPage() {
  return (
    <div className="space-y-4 px-8 py-8">
      <p className="text-sm text-muted-foreground">
        Past Vortex workflow runs. Open a run to inspect its workflow, inputs, and outputs — or
        re-run it with edited inputs.
      </p>
      <PastRunsTab />
    </div>
  )
}

// Vortex (ai_canvas) — view the current canvas graph as JSON and copy it.
import { useState } from 'react'

import { Dialog } from '@/components/Dialog'
import type { CanvasGraph } from '@/types/api'

export function GraphJson({
  graph,
  disabled = false,
}: {
  graph: CanvasGraph
  disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const text = JSON.stringify(graph, null, 2)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard API can be blocked (e.g. no HTTPS focus); leave the JSON
      // visible for a manual select-and-copy.
    }
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        disabled={disabled}
        title="View graph JSON"
        aria-label="View graph JSON"
        className="rounded-md border border-border bg-card/95 p-1.5 text-muted-foreground shadow-md hover:bg-accent hover:text-foreground disabled:opacity-40"
      >
        {/* code / braces icon */}
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5a2 2 0 0 0 2 2h1" />
          <path d="M16 3h1a2 2 0 0 1 2 2v5a2 2 0 0 0 2 2 2 2 0 0 0-2 2v5a2 2 0 0 1-2 2h-1" />
        </svg>
      </button>

      <Dialog open={open} onClose={() => setOpen(false)} title="Graph JSON" width="max-w-2xl">
        <div className="space-y-3">
          <div className="flex justify-end">
            <button
              onClick={copy}
              className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-accent"
            >
              {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>
          <pre className="max-h-[60vh] overflow-auto rounded-md border border-border bg-muted/40 p-3 text-[11px] leading-relaxed">
            {text}
          </pre>
        </div>
      </Dialog>
    </>
  )
}

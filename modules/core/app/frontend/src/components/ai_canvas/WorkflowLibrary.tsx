// Vortex (ai_canvas) — Save current graph + browse/load/delete saved workflows.
import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '@/api/client'
import { Dialog } from '@/components/Dialog'
import type { CanvasGraph, CanvasWorkflowDetail } from '@/types/api'

export function WorkflowLibrary({
  graph,
  loadedId,
  loadedName,
  onLoad,
  onSaved,
  disabled,
}: {
  graph: CanvasGraph
  loadedId: string | null
  loadedName: string
  onLoad: (detail: CanvasWorkflowDetail) => void
  onSaved: (id: string, name: string) => void
  disabled: boolean
}) {
  const qc = useQueryClient()
  const [saveOpen, setSaveOpen] = useState(false)
  const [libOpen, setLibOpen] = useState(false)
  const [name, setName] = useState(loadedName)
  const [description, setDescription] = useState('')

  const list = useQuery({
    queryKey: ['ai_canvas', 'workflows'],
    queryFn: api.aiCanvasListWorkflows,
    enabled: libOpen,
  })

  const save = useMutation({
    mutationFn: () =>
      api.aiCanvasSaveWorkflow({
        workflow_id: loadedId ?? undefined,
        name: name.trim(),
        description: description.trim(),
        graph,
      }),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ['ai_canvas', 'workflows'] })
      onSaved(res.workflow_id, name.trim())
      setSaveOpen(false)
    },
  })

  const load = useMutation({
    mutationFn: (id: string) => api.aiCanvasGetWorkflow(id),
    onSuccess: (detail) => {
      onLoad(detail)
      setLibOpen(false)
    },
  })

  const del = useMutation({
    mutationFn: (id: string) => api.aiCanvasDeleteWorkflow(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['ai_canvas', 'workflows'] }),
  })

  return (
    <>
      <button
        onClick={() => {
          setName(loadedName)
          setSaveOpen(true)
        }}
        disabled={disabled}
        className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-accent disabled:opacity-40"
      >
        Save
      </button>
      <button
        onClick={() => setLibOpen(true)}
        className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-accent"
      >
        Load
      </button>

      {/* Save dialog */}
      <Dialog open={saveOpen} onClose={() => setSaveOpen(false)} title="Save workflow" width="max-w-md">
        <div className="space-y-3">
          <label className="block">
            <div className="mb-1 text-xs font-medium">Name</div>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded-md border border-border bg-background px-2 py-1 text-sm"
              placeholder="My folding + solubility workflow"
            />
          </label>
          <label className="block">
            <div className="mb-1 text-xs font-medium">Description</div>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
              className="w-full rounded-md border border-border bg-background px-2 py-1 text-sm"
            />
          </label>
          {loadedId != null && (
            <p className="text-[11px] text-muted-foreground">
              Updating the workflow you loaded. Change the name to save a copy.
            </p>
          )}
          {save.isError && (
            <p className="text-xs text-destructive">{(save.error as Error).message}</p>
          )}
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setSaveOpen(false)}
              className="rounded-md border border-border px-3 py-1 text-xs hover:bg-accent"
            >
              Cancel
            </button>
            <button
              onClick={() => save.mutate()}
              disabled={!name.trim() || save.isPending}
              className="rounded-md bg-primary px-3 py-1 text-xs font-medium text-primary-foreground hover:opacity-90 disabled:opacity-40"
            >
              {save.isPending ? 'Saving…' : 'Save'}
            </button>
          </div>
        </div>
      </Dialog>

      {/* Library dialog */}
      <Dialog open={libOpen} onClose={() => setLibOpen(false)} title="Load workflow" width="max-w-lg">
        {list.isLoading ? (
          <p className="text-sm text-muted-foreground">Loading…</p>
        ) : !list.data || list.data.workflows.length === 0 ? (
          <p className="text-sm text-muted-foreground">No saved workflows yet.</p>
        ) : (
          <ul className="divide-y divide-border">
            {list.data.workflows.map((w) => (
              <li key={w.workflow_id} className="flex items-center gap-3 py-2">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium">{w.name}</div>
                  {w.description && (
                    <div className="truncate text-xs text-muted-foreground">{w.description}</div>
                  )}
                  <div className="text-[10px] text-muted-foreground">Updated {w.updated_date}</div>
                </div>
                <button
                  onClick={() => load.mutate(w.workflow_id)}
                  disabled={load.isPending}
                  className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-accent disabled:opacity-40"
                >
                  Load
                </button>
                <button
                  onClick={() => del.mutate(w.workflow_id)}
                  disabled={del.isPending}
                  className="rounded-md border border-destructive/40 px-2.5 py-1 text-xs text-destructive hover:bg-destructive/10 disabled:opacity-40"
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
      </Dialog>
    </>
  )
}

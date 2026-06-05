// Cross-module "Pick a structure from a prior run" selector.
//
// Lets a downstream tab (e.g. Small Molecule docking / ligand design) load a
// PDB produced by a prior Large Molecule structure run instead of pasting it —
// the same "Pick a Run" UX as Genomics. Reuses the existing AlphaFold search +
// result endpoints (AlphaFold is the Large Molecule workflow that persists a
// foldable structure per run); no new backend.
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'

import { api } from '@/api/client'
import { Dialog } from '@/components/Dialog'
import { cn } from '@/lib/utils'
import type { AlphaFoldRun } from '@/types/api'

type SearchMode = 'experiment_name' | 'run_name'

export function StructurePicker({ onPick }: { onPick: (pdb: string) => void }) {
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState<SearchMode>('experiment_name')
  const [text, setText] = useState('alphafold')
  const [rows, setRows] = useState<AlphaFoldRun[]>([])

  const search = useMutation({
    mutationFn: () => api.alphafoldSearch(mode, text.trim()),
    onSuccess: (res) => setRows(res.runs),
  })
  const load = useMutation({
    mutationFn: (r: AlphaFoldRun) => api.alphafoldResult(r.run_id, r.run_name),
    onSuccess: (res) => {
      onPick(res.pdb)
      setOpen(false)
    },
  })

  return (
    <>
      <button
        type="button"
        onClick={() => {
          setOpen(true)
          if (rows.length === 0) search.mutate()
        }}
        className="rounded-md border border-border px-2.5 py-1 text-xs hover:bg-accent"
      >
        Pick a structure from a prior run
      </button>

      <Dialog open={open} onClose={() => setOpen(false)} title="Pick a predicted structure" width="max-w-2xl">
        <div className="mb-3 flex items-center gap-2">
          <div className="flex gap-1">
            {(['experiment_name', 'run_name'] as SearchMode[]).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={cn(
                  'rounded-full border px-2.5 py-1 text-xs',
                  m === mode ? 'border-primary bg-primary/10 text-primary' : 'border-border text-muted-foreground hover:bg-accent',
                )}
              >
                {m === 'experiment_name' ? 'Experiment' : 'Run name'}
              </button>
            ))}
          </div>
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="contains…"
            className="min-w-0 flex-1 rounded-md border border-border bg-background px-2 py-1 text-xs"
            onKeyDown={(e) => e.key === 'Enter' && search.mutate()}
          />
          <button
            type="button"
            onClick={() => search.mutate()}
            disabled={search.isPending}
            className="rounded-md border border-border px-3 py-1 text-xs hover:bg-accent disabled:opacity-40"
          >
            {search.isPending ? 'Searching…' : 'Search'}
          </button>
        </div>

        {search.isError && <p className="text-xs text-destructive">{String(search.error)}</p>}
        {load.isError && <p className="text-xs text-destructive">{String(load.error)}</p>}

        {rows.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            {search.isPending ? 'Searching…' : 'No AlphaFold runs found — run a structure prediction first, or adjust the search.'}
          </p>
        ) : (
          <div className="max-h-72 overflow-auto rounded-md border border-border">
            <table className="w-full text-xs">
              <thead className="bg-muted/50 uppercase text-muted-foreground">
                <tr>
                  <th className="px-3 py-1 text-left">Run</th>
                  <th className="px-3 py-1 text-left">Experiment</th>
                  <th className="px-3 py-1 text-left">Status</th>
                  <th className="px-3 py-1"></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const ready = r.status === 'fold_complete'
                  return (
                    <tr key={r.run_id} className="border-t border-border">
                      <td className="px-3 py-1">{r.run_name}</td>
                      <td className="px-3 py-1">{r.experiment_name}</td>
                      <td className="px-3 py-1">{r.status}</td>
                      <td className="px-3 py-1 text-right">
                        <button
                          type="button"
                          onClick={() => load.mutate(r)}
                          disabled={!ready || load.isPending}
                          title={ready ? 'Load this structure' : 'Available once status is fold_complete'}
                          className="rounded-md border border-primary/50 bg-primary/10 px-2.5 py-0.5 text-primary hover:bg-primary/20 disabled:opacity-40"
                        >
                          {load.isPending && load.variables?.run_id === r.run_id ? 'Loading…' : 'Use'}
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Dialog>
    </>
  )
}

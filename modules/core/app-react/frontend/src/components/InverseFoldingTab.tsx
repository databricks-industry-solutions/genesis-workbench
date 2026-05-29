import { useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { MolstarViewer } from '@/components/MolstarViewer'
import { WorkflowProgress } from '@/components/WorkflowProgress'

const DEFAULT_PDB = `ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N
ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 10.38           C
ATOM      3  C   MET A   1      26.913  26.639   3.531  1.00  9.62           C
ATOM      4  O   MET A   1      27.886  26.463   4.263  1.00  9.62           O
ATOM      5  N   GLN A   2      26.335  27.770   3.258  1.00  9.27           N
ATOM      6  CA  GLN A   2      26.850  29.021   3.898  1.00 10.07           C
ATOM      7  C   GLN A   2      26.100  29.253   5.202  1.00  9.68           C
ATOM      8  O   GLN A   2      24.865  29.024   5.330  1.00  9.38           O
ATOM      9  N   ILE A   3      26.849  29.569   6.250  1.00 10.00           N
ATOM     10  CA  ILE A   3      26.235  30.050   7.497  1.00 10.25           C
ATOM     11  C   ILE A   3      26.882  31.410   7.862  1.00 10.97           C
ATOM     12  O   ILE A   3      28.032  31.634   7.483  1.00 12.41           O
ATOM     13  N   PHE A   4      26.106  32.305   8.468  1.00  9.85           N
ATOM     14  CA  PHE A   4      26.574  33.660   8.776  1.00 10.62           C
ATOM     15  C   PHE A   4      26.644  34.482   7.486  1.00 10.33           C
ATOM     16  O   PHE A   4      25.724  34.471   6.669  1.00 10.89           O
ATOM     17  N   VAL A   5      27.741  35.183   7.310  1.00 10.10           N
ATOM     18  CA  VAL A   5      27.956  35.969   6.083  1.00 10.68           C
ATOM     19  C   VAL A   5      28.406  37.363   6.480  1.00 11.22           C
ATOM     20  O   VAL A   5      29.024  37.528   7.535  1.00 12.64           O
ATOM     21  N   LYS A   6      28.068  38.349   5.643  1.00 11.00           N
ATOM     22  CA  LYS A   6      28.399  39.737   5.953  1.00 12.16           C
ATOM     23  C   LYS A   6      27.170  40.466   6.518  1.00 12.50           C
ATOM     24  O   LYS A   6      26.022  40.004   6.313  1.00 12.23           O
ATOM     25  N   THR A   7      27.439  41.518   7.285  1.00 12.76           N
ATOM     26  CA  THR A   7      26.340  42.374   7.741  1.00 13.04           C
ATOM     27  C   THR A   7      26.934  43.770   7.965  1.00 13.42           C
ATOM     28  O   THR A   7      28.140  43.956   8.110  1.00 14.32           O
ATOM     29  N   LEU A   8      26.046  44.758   8.043  1.00 13.16           N
ATOM     30  CA  LEU A   8      26.451  46.148   8.258  1.00 14.51           C
ATOM     31  C   LEU A   8      25.695  46.713   9.457  1.00 14.39           C
ATOM     32  O   LEU A   8      24.462  46.672   9.483  1.00 15.33           O
END
`

export function InverseFoldingTab() {
  const [pdb, setPdb] = useState(DEFAULT_PDB)
  const [selectedIdx, setSelectedIdx] = useState(0)

  const design = useMutation({
    mutationFn: () => api.inverseFolding(pdb),
    onSuccess: () => setSelectedIdx(0),
  })

  const sequences = design.data?.sequences ?? []
  const selectedSeq = sequences[selectedIdx]

  // Auto-fold the selected sequence whenever the selection changes.
  const fold = useQuery({
    queryKey: ['inv-fold', 'esmfold', selectedSeq],
    queryFn: () => api.esmfold(selectedSeq ?? ''),
    enabled: Boolean(selectedSeq),
    staleTime: 5 * 60_000,
  })

  // Reset the dropdown when the design list shrinks (or first arrival).
  useEffect(() => {
    if (sequences.length > 0 && selectedIdx >= sequences.length) {
      setSelectedIdx(0)
    }
  }, [sequences.length, selectedIdx])

  const canDesign = pdb.trim().includes('ATOM') && !design.isPending

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Design Sequences for a Backbone</h3>
        <p className="text-xs text-muted-foreground">
          Given a protein backbone (PDB), generate new amino-acid sequences predicted to fold into
          that structure. Each design is then validated by ESMFold.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(320px,420px)_1fr]">
        {/* Left: form + design selector */}
        <div className="space-y-3">
          <label className="block text-xs">
            <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
              PDB content
            </span>
            <textarea
              rows={10}
              value={pdb}
              onChange={(e) => setPdb(e.target.value)}
              placeholder="Paste a PDB starting with ATOM records"
              className="w-full rounded-md border border-border bg-background p-3 font-mono text-[11px] leading-tight"
            />
          </label>
          <div className="flex gap-2">
            <button
              onClick={() => design.mutate()}
              disabled={!canDesign}
              className="flex-1 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
            >
              {design.isPending ? 'Designing…' : 'Design Sequences'}
            </button>
            <button
              onClick={() => {
                design.reset()
                setSelectedIdx(0)
              }}
              disabled={!design.data && !design.isError}
              className="rounded-md border border-border px-4 py-2 text-sm hover:bg-accent disabled:opacity-50"
            >
              Clear
            </button>
          </div>

          {sequences.length > 0 && (
            <div className="space-y-2 rounded-md border border-border bg-card p-3">
              <label className="block text-xs">
                <span className="mb-1 block uppercase tracking-wide text-muted-foreground">
                  Select design
                </span>
                <select
                  value={selectedIdx}
                  onChange={(e) => setSelectedIdx(parseInt(e.target.value))}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                >
                  {sequences.map((_, i) => (
                    <option key={i} value={i}>
                      Design {i + 1}
                    </option>
                  ))}
                </select>
              </label>
              <div className="text-[11px] text-muted-foreground">
                {sequences.length} designs returned by ProteinMPNN
              </div>
              <div>
                <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                  Sequence
                </div>
                <pre className="max-h-40 overflow-auto rounded-md border border-border bg-muted/30 p-2 font-mono text-[10px] leading-tight">
                  {selectedSeq}
                </pre>
              </div>
            </div>
          )}
        </div>

        {/* Right: progress + viewer */}
        <div className="space-y-3">
          <WorkflowProgress
            active={design.isPending}
            title="ProteinMPNN inverse-folding"
            stages={[{ label: 'Designing sequences for backbone', estSeconds: 8 }]}
          />

          {design.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
              {String(design.error)}
            </div>
          )}

          {sequences.length > 0 ? (
            <>
              <h4 className="text-sm font-medium">
                Validated structure (ESMFold) — Design {selectedIdx + 1}
              </h4>
              <WorkflowProgress
                active={fold.isLoading}
                title={`Folding Design ${selectedIdx + 1}`}
                stages={[{ label: 'Predicting structure with ESMFold', estSeconds: 12 }]}
              />
              {fold.error ? (
                <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
                  ESMFold failed: {String(fold.error)}
                </div>
              ) : fold.data ? (
                <MolstarViewer viewerHtml={fold.data.viewer_html} height={520} />
              ) : null}
            </>
          ) : (
            !design.isPending &&
            !design.error && (
              <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
                Paste a backbone PDB and run Design Sequences. ProteinMPNN returns candidate
                sequences for that backbone; each one is then auto-folded by ESMFold and shown
                here.
              </div>
            )
          )}
        </div>
      </div>
    </div>
  )
}

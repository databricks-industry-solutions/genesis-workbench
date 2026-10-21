// Decision-support readouts for the docking / ligand-binder result viewers. CPU-side numbers that
// help a scientist judge a pose/design at a glance: a compact metrics strip (shown under the viewer)
// and an RDKit ligand-druglikeness panel (one per run).
import type { LigandDruglikeness } from '@/types/api'

export type Metric = { label: string; value: string; title?: string }

export function MetricsStrip({ items }: { items: Metric[] }) {
  const shown = items.filter((i) => i.value !== '' && i.value != null)
  if (!shown.length) return null
  return (
    <div className="flex flex-wrap gap-2 text-xs">
      {shown.map((i) => (
        <div
          key={i.label}
          title={i.title}
          className="rounded-md border border-border bg-card px-2.5 py-1.5"
        >
          <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{i.label}</div>
          <div className="font-medium">{i.value}</div>
        </div>
      ))}
    </div>
  )
}

function Item({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  )
}

export function DruglikenessPanel({ dl }: { dl: LigandDruglikeness | null | undefined }) {
  if (!dl) return null
  return (
    <div className="rounded-md border border-border bg-card p-3 text-xs">
      <div className="mb-2 font-medium uppercase tracking-wide text-muted-foreground">
        Ligand druglikeness (RDKit)
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        <Item label="MW" value={`${dl.mw} Da`} />
        <Item label="logP" value={String(dl.logp)} />
        <Item label="QED" value={String(dl.qed)} />
        <Item label="TPSA" value={`${dl.tpsa} Å²`} />
        <Item label="H-bond donors" value={String(dl.hbd)} />
        <Item label="H-bond acceptors" value={String(dl.hba)} />
        <Item label="Rotatable bonds" value={String(dl.rotatable_bonds)} />
        <Item
          label="Lipinski Ro5"
          value={dl.lipinski_pass ? `pass (${dl.ro5_violations} viol.)` : `fail (${dl.ro5_violations} viol.)`}
        />
        <Item label="Veber" value={dl.veber_pass ? 'pass' : 'fail'} />
      </div>
    </div>
  )
}

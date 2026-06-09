// NVIDIA BioNeMo page. ESM2 (Fine Tune + Inference) today; Geneformer is a
// placeholder, mirroring the original Streamlit layout.
import { Tabs } from '@/components/Tabs'
import { BionemoFinetuneTab } from '@/components/bionemo/BionemoFinetuneTab'
import { BionemoInferenceTab } from '@/components/bionemo/BionemoInferenceTab'
import { KermtFinetuneTab } from '@/components/bionemo/KermtFinetuneTab'

function Esm2Tab() {
  return (
    <Tabs
      tabs={[
        { id: 'finetune', label: 'Fine Tune', content: <BionemoFinetuneTab /> },
        { id: 'inference', label: 'Inference', content: <BionemoInferenceTab /> },
      ]}
    />
  )
}

function GeneformerTab() {
  return (
    <div className="rounded-md border border-dashed border-border bg-muted/20 p-6 text-center text-sm text-muted-foreground">
      Geneformer support is coming soon.
    </div>
  )
}

export function BionemoPage() {
  return (
    <div className="space-y-6 px-8 py-8">
      <Tabs
        tabs={[
          { id: 'kermt', label: 'KERMT', content: <KermtFinetuneTab /> },
          { id: 'esm2', label: 'ESM2', content: <Esm2Tab /> },
          { id: 'geneformer', label: 'Geneformer', content: <GeneformerTab /> },
        ]}
      />
    </div>
  )
}

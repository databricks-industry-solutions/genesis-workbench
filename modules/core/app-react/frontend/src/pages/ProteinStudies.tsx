import { Tabs } from '@/components/Tabs'
import { DeployedModelsButton } from '@/components/DeployedModelsButton'
import { StructurePredictionTab } from '@/components/StructurePredictionTab'
import { SequenceSearchTab } from '@/components/SequenceSearchTab'
import { InverseFoldingTab } from '@/components/InverseFoldingTab'
import { ProteinDesignTab } from '@/components/ProteinDesignTab'

export function ProteinStudiesPage() {
  return (
    <div className="space-y-6 px-8 py-8">
      <header>
        <h1 className="text-2xl font-semibold">Protein Studies</h1>
      </header>
      <Tabs
        rightAccessory={<DeployedModelsButton module="protein_studies" />}
        tabs={[
          {
            id: 'sequence_search',
            label: 'Sequence Search',
            content: <SequenceSearchTab />,
          },
          {
            id: 'structure_prediction',
            label: 'Protein Structure Prediction',
            content: <StructurePredictionTab />,
          },
          {
            id: 'protein_design',
            label: 'Protein Design',
            content: <ProteinDesignTab />,
          },
          {
            id: 'inverse_folding',
            label: 'Inverse Folding',
            content: <InverseFoldingTab />,
          },
        ]}
      />
    </div>
  )
}

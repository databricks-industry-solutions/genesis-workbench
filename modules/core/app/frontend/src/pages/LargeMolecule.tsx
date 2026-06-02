import { Tabs } from '@/components/Tabs'
import { DeployedModelsButton } from '@/components/DeployedModelsButton'
import { EnzymeOptimizationTab } from '@/components/EnzymeOptimizationTab'
import { InverseFoldingTab } from '@/components/InverseFoldingTab'
import { ProteinBinderDesignTab } from '@/components/ProteinBinderDesignTab'
import { ProteinDesignTab } from '@/components/ProteinDesignTab'
import { SequenceSearchTab } from '@/components/SequenceSearchTab'
import { StructurePredictionTab } from '@/components/StructurePredictionTab'

export function LargeMoleculePage() {
  return (
    <div className="space-y-6 px-8 py-8">
      <Tabs
        rightAccessory={<DeployedModelsButton module="large_molecule" />}
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
          {
            id: 'binder_design',
            label: 'Protein Binder Design',
            content: <ProteinBinderDesignTab />,
          },
          {
            id: 'enzyme_optimization',
            label: 'Guided Enzyme Optimization',
            content: <EnzymeOptimizationTab />,
          },
        ]}
      />
    </div>
  )
}

import { Tabs } from '@/components/Tabs'
import { AdmetSafetyTab } from '@/components/AdmetSafetyTab'
import { DeployedModelsTab } from '@/components/DeployedModelsTab'
import { EnzymeOptimizationTab } from '@/components/EnzymeOptimizationTab'
import { LigandBinderDesignTab } from '@/components/LigandBinderDesignTab'
import { MolecularDockingTab } from '@/components/MolecularDockingTab'
import { MotifScaffoldingTab } from '@/components/MotifScaffoldingTab'
import { ProteinBinderDesignTab } from '@/components/ProteinBinderDesignTab'

export function SmallMoleculesPage() {
  return (
    <div className="space-y-6 px-8 py-8">
      <header>
        <h1 className="text-2xl font-semibold">Small Molecules</h1>
      </header>
      <Tabs
        tabs={[
          {
            id: 'docking',
            label: 'Molecular Docking',
            content: <MolecularDockingTab />,
          },
          {
            id: 'binder_design',
            label: 'Protein Binder Design',
            content: <ProteinBinderDesignTab />,
          },
          {
            id: 'ligand_binder',
            label: 'Ligand Binder Design',
            content: <LigandBinderDesignTab />,
          },
          {
            id: 'motif_scaffolding',
            label: 'Motif Scaffolding',
            content: <MotifScaffoldingTab />,
          },
          {
            id: 'enzyme_optimization',
            label: 'Guided Enzyme Optimization',
            content: <EnzymeOptimizationTab />,
          },
          {
            id: 'admet',
            label: 'ADMET & Safety',
            content: <AdmetSafetyTab />,
          },
          { id: 'models', label: 'Deployed Models', content: <DeployedModelsTab module="small_molecule" /> },
        ]}
      />
    </div>
  )
}

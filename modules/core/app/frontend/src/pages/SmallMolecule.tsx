import { Tabs } from '@/components/Tabs'
import { AdmetSafetyTab } from '@/components/AdmetSafetyTab'
import { DeployedModelsButton } from '@/components/DeployedModelsButton'
import { GenMolGenerateTab } from '@/components/GenMolGenerateTab'
import { LigandBinderDesignTab } from '@/components/LigandBinderDesignTab'
import { MolecularDockingTab } from '@/components/MolecularDockingTab'
import { MotifScaffoldingTab } from '@/components/MotifScaffoldingTab'

export function SmallMoleculePage() {
  return (
    <div className="space-y-6 px-8 py-8">
      <Tabs
        rightAccessory={<DeployedModelsButton module="small_molecule" />}
        tabs={[
          {
            id: 'design',
            label: 'Small Molecule Design',
            content: <GenMolGenerateTab />,
          },
          {
            id: 'docking',
            label: 'Molecular Docking',
            content: <MolecularDockingTab />,
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
            id: 'admet',
            label: 'ADMET & Safety',
            content: <AdmetSafetyTab />,
          },
        ]}
      />
    </div>
  )
}

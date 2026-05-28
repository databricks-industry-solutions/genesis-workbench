import { Tabs } from '@/components/Tabs'
import { DeployedModelsTab } from '@/components/DeployedModelsTab'
import { WorkflowComingSoon } from '@/components/WorkflowComingSoon'

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
            content: <WorkflowComingSoon name="Molecular Docking" />,
          },
          {
            id: 'binder_design',
            label: 'Protein Binder Design',
            content: <WorkflowComingSoon name="Protein Binder Design" />,
          },
          {
            id: 'ligand_binder',
            label: 'Ligand Binder Design',
            content: <WorkflowComingSoon name="Ligand Binder Design" />,
          },
          {
            id: 'motif_scaffolding',
            label: 'Motif Scaffolding',
            content: <WorkflowComingSoon name="Motif Scaffolding" />,
          },
          {
            id: 'enzyme_optimization',
            label: 'Guided Enzyme Optimization',
            content: <WorkflowComingSoon name="Guided Enzyme Optimization" />,
          },
          {
            id: 'admet',
            label: 'ADMET & Safety',
            content: <WorkflowComingSoon name="ADMET & Safety" />,
          },
          { id: 'models', label: 'Deployed Models', content: <DeployedModelsTab module="small_molecule" /> },
        ]}
      />
    </div>
  )
}

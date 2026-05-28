import { Tabs } from '@/components/Tabs'
import { DeployedModelsTab } from '@/components/DeployedModelsTab'
import { WorkflowComingSoon } from '@/components/WorkflowComingSoon'

export function DiseaseBiologyPage() {
  return (
    <div className="space-y-6 px-8 py-8">
      <header>
        <h1 className="text-2xl font-semibold">Disease Biology</h1>
      </header>
      <Tabs
        tabs={[
          {
            id: 'variant_calling',
            label: 'Variant Calling',
            content: <WorkflowComingSoon name="Variant Calling" />,
          },
          { id: 'gwas', label: 'GWAS', content: <WorkflowComingSoon name="GWAS Analysis" /> },
          {
            id: 'ingestion',
            label: 'VCF Ingestion',
            content: <WorkflowComingSoon name="VCF Ingestion" />,
          },
          {
            id: 'annotation',
            label: 'Variant Annotation',
            content: <WorkflowComingSoon name="Variant Annotation" />,
          },
          { id: 'models', label: 'Deployed Models', content: <DeployedModelsTab module="disease_biology" /> },
        ]}
      />
    </div>
  )
}

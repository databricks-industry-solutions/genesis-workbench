import { Tabs } from '@/components/Tabs'
import { DeployedModelsButton } from '@/components/DeployedModelsButton'
import { GwasTab } from '@/components/GwasTab'
import { VariantAnnotationTab } from '@/components/VariantAnnotationTab'
import { VariantCallingTab } from '@/components/VariantCallingTab'
import { VcfIngestionTab } from '@/components/VcfIngestionTab'

export function DiseaseBiologyPage() {
  return (
    <div className="space-y-6 px-8 py-8">
      <header>
        <h1 className="text-2xl font-semibold">Disease Biology</h1>
      </header>
      <Tabs
        rightAccessory={<DeployedModelsButton module="disease_biology" />}
        tabs={[
          {
            id: 'variant_calling',
            label: 'Variant Calling',
            content: <VariantCallingTab />,
          },
          {
            id: 'gwas',
            label: 'GWAS',
            content: <GwasTab />,
          },
          {
            id: 'ingestion',
            label: 'VCF Ingestion',
            content: <VcfIngestionTab />,
          },
          {
            id: 'annotation',
            label: 'Variant Annotation',
            content: <VariantAnnotationTab />,
          },
        ]}
      />
    </div>
  )
}

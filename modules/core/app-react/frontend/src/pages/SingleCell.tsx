import { AnalysisTab } from '@/components/AnalysisTab'
import { DeployedModelsButton } from '@/components/DeployedModelsButton'
import { RawProcessingTab } from '@/components/RawProcessingTab'
import { Tabs } from '@/components/Tabs'

export function SingleCellPage() {
  return (
    <div className="space-y-6 px-8 py-8">
      <header>
        <h1 className="text-2xl font-semibold">Single Cell Studies</h1>
      </header>
      <Tabs
        rightAccessory={<DeployedModelsButton module="single_cell" />}
        tabs={[
          {
            id: 'processing',
            label: 'Raw Single Cell Processing',
            content: <RawProcessingTab />,
          },
          {
            id: 'analysis',
            label: 'Analysis',
            content: <AnalysisTab />,
          },
        ]}
      />
    </div>
  )
}

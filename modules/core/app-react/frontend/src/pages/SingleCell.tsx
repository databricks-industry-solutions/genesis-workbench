import { AnalysisTab } from '@/components/AnalysisTab'
import { DeployedModelsTab } from '@/components/DeployedModelsTab'
import { RawProcessingTab } from '@/components/RawProcessingTab'
import { Tabs } from '@/components/Tabs'

// 3 top-level tabs:
//   1. Deployed Models — what's currently serving.
//   2. Raw Single Cell Processing — submit + search runs. View opens a popup
//      with the run's info card + QC/Raw Data.
//   3. Analysis — one shared run picker drives UMAP, markers, DE, pathway
//      enrichment, trajectory, cell similarity, and perturbation prediction.
export function SingleCellPage() {
  return (
    <div className="space-y-6 px-8 py-8">
      <header>
        <h1 className="text-2xl font-semibold">Single Cell Studies</h1>
      </header>
      <Tabs
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
          {
            id: 'models',
            label: 'Deployed Models',
            content: <DeployedModelsTab module="single_cell" />,
          },
        ]}
      />
    </div>
  )
}

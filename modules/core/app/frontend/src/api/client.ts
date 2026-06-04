import type {
  AdminDashboardResponse,
  AssistantQueryResponse,
  AvailableModelsResponse,
  BatchModelsResponse,
  BionemoDefaultsResponse,
  BionemoDispatchResponse,
  BionemoFinetuneRequest,
  BionemoFinetuneRunDetails,
  BionemoInferenceRequest,
  BionemoVariantsResponse,
  BionemoWeightsResponse,
  BootstrapResponse,
  DBSearchResponse,
  DeployedModelsResponse,
  GenomicsDefaultsResponse,
  DockingExampleResponse,
  DocsResponse,
  EnzymeDefaultsResponse,
  EnzymeOptimizationStartRequest,
  EnzymeOptimizationStartResponse,
  EnzymeSearchResponse,
  EnzymeSmokeTestResponse,
  EnzymeStatusResponse,
  EnzymeTopKResponse,
  GwasResultsResponse,
  GwasStartRequest,
  JobDispatchResponse,
  RunDetailsResponse,
  VariantAnnotationDashboardResponse,
  VariantAnnotationResultsResponse,
  VariantAnnotationStartRequest,
  VariantCallingPickerResponse,
  VariantCallingStartRequest,
  VcfIngestionPickerResponse,
  VcfIngestionStartRequest,
  EndpointStatusResponse,
  HealthResponse,
  MeResponse,
  MlflowTestResponse,
  ModuleBatchModelsResponse,
  ModuleName,
  AlphaFoldResultResponse,
  AlphaFoldSearchResponse,
  AlphaFoldStartResponse,
  AnnotateResponse,
  InverseFoldingResponse,
  ResolveGeneResponse,
  OrganismResponse,
  PerturbationResponse,
  ProfileResponse,
  ProteinDesignResponse,
  ColorPointsResponse,
  DEResponse,
  GenesetDbsResponse,
  GenesetTermsResponse,
  NarrativeResponse,
  PerturbationNarrativeRequest,
  DotplotResponse,
  EnrichmentResponse,
  RawDataResponse,
  RunInfoResponse,
  RunSummaryResponse,
  SavedAnnotationsResponse,
  SequenceSearchResponse,
  TrajectoryResponse,
  SimilarityResponse,
  SingleCellRunsResponse,
  StartProcessingRequest,
  StartProcessingResponse,
  StructurePredictionResponse,
  TeddyAnnotateResponse,
  ProfileSaveRequest,
  StartEndpointsStatusResponse,
  SystemSettingsResponse,
  WorkflowRunsResponse,
} from '@/types/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  // Read body as text once, then optionally parse as JSON. Avoids
  // 'body stream already read' when the initial .json() throws on
  // non-JSON / empty responses (e.g. auth redirects, plain-text errors).
  const text = await res.text()
  if (!res.ok) {
    let detail: unknown = text
    try {
      detail = JSON.parse(text)
    } catch {
      // fall through with the raw text
    }
    const summary = typeof detail === 'string' ? detail : JSON.stringify(detail)
    throw new Error(`HTTP ${res.status} ${path}: ${summary}`)
  }
  return (text ? JSON.parse(text) : ({} as T)) as T
}

export const api = {
  health: () => request<HealthResponse>('/api/health'),
  me: () => request<MeResponse>('/api/me'),
  bootstrap: () => request<BootstrapResponse>('/api/bootstrap'),

  getProfile: () => request<ProfileResponse>('/api/profile'),
  saveProfile: (body: ProfileSaveRequest) =>
    request<ProfileResponse>('/api/profile', { method: 'PUT', body: JSON.stringify(body) }),
  saveTheme: (theme: 'dark' | 'light') =>
    request<ProfileResponse>('/api/profile/theme', {
      method: 'PUT',
      body: JSON.stringify({ theme }),
    }),
  testMlflow: (mlflow_experiment_folder: string) =>
    request<MlflowTestResponse>('/api/mlflow/test', {
      method: 'POST',
      body: JSON.stringify({ mlflow_experiment_folder }),
    }),

  systemSettings: () => request<SystemSettingsResponse>('/api/settings/system'),
  endpointStatuses: () => request<EndpointStatusResponse>('/api/settings/endpoints'),
  batchModels: () => request<BatchModelsResponse>('/api/settings/batch-models'),
  startEndpointsStatus: () =>
    request<StartEndpointsStatusResponse>('/api/settings/start-endpoints/status'),
  startEndpointsTrigger: (num_hours: number) =>
    request<{ run_id: string }>('/api/settings/start-endpoints/trigger', {
      method: 'POST',
      body: JSON.stringify({ num_hours }),
    }),

  monitoringRuns: (days_back: number) =>
    request<WorkflowRunsResponse>(`/api/monitoring/runs?days_back=${days_back}`),
  adminDashboard: () => request<AdminDashboardResponse>('/api/monitoring/admin-dashboard'),

  assistantQuery: (query: string) =>
    request<AssistantQueryResponse>('/api/assistant/query', {
      method: 'POST',
      body: JSON.stringify({ query }),
    }),

  docs: () => request<DocsResponse>('/api/docs'),

  availableModels: (module: ModuleName) =>
    request<AvailableModelsResponse>(`/api/models/available?module=${module}`),
  deployedModelsByModule: (module: ModuleName) =>
    request<DeployedModelsResponse>(`/api/models/deployed?module=${module}`),
  batchModelsByModule: (module: ModuleName) =>
    request<ModuleBatchModelsResponse>(`/api/models/batch?module=${module}`),

  esmfold: (sequence: string) =>
    request<StructurePredictionResponse>('/api/large_molecule/esmfold', {
      method: 'POST',
      body: JSON.stringify({ sequence }),
    }),
  boltz: (sequence: string) =>
    request<StructurePredictionResponse>('/api/large_molecule/boltz', {
      method: 'POST',
      body: JSON.stringify({ sequence }),
    }),

  alphafoldStart: (body: { sequence: string; experiment_name: string; run_name: string }) =>
    request<AlphaFoldStartResponse>('/api/large_molecule/alphafold/start', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  alphafoldSearch: (by: 'experiment_name' | 'run_name', text: string) =>
    request<AlphaFoldSearchResponse>(
      `/api/large_molecule/alphafold/search?by=${by}&text=${encodeURIComponent(text)}`,
    ),
  alphafoldResult: (run_id: string, run_name: string) =>
    request<AlphaFoldResultResponse>(
      `/api/large_molecule/alphafold/result?run_id=${encodeURIComponent(run_id)}&run_name=${encodeURIComponent(run_name)}`,
    ),

  sequenceSearch: (sequence: string, top_k: number) =>
    request<SequenceSearchResponse>('/api/large_molecule/sequence_search', {
      method: 'POST',
      body: JSON.stringify({ sequence, top_k }),
    }),
  sequenceOrganism: (description: string) =>
    request<OrganismResponse>('/api/large_molecule/sequence_search/organism', {
      method: 'POST',
      body: JSON.stringify({ description }),
    }),

  resolveGene: (gene: string) =>
    request<ResolveGeneResponse>(
      `/api/large_molecule/resolve_gene?gene=${encodeURIComponent(gene)}`,
    ),

  inverseFolding: (pdb: string) =>
    request<InverseFoldingResponse>('/api/large_molecule/inverse_folding', {
      method: 'POST',
      body: JSON.stringify({ pdb }),
    }),

  proteinDesign: (body: {
    sequence: string
    experiment_name: string
    run_name: string
    n_rfdiffusion_hits: number
  }) =>
    request<ProteinDesignResponse>('/api/large_molecule/protein_design', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  singleCellRuns: () => request<SingleCellRunsResponse>('/api/single_cell/runs'),
  singleCellAnnotate: (body: {
    run_id: string
    cells_per_cluster: number
    k_neighbors: number
  }) =>
    request<AnnotateResponse>('/api/single_cell/annotate', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  singleCellAnnotateTeddy: (body: {
    run_id: string
    cells_per_cluster: number
    k_neighbors: number
    bias_correct: boolean
  }) =>
    request<TeddyAnnotateResponse>('/api/single_cell/annotate-teddy', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  singleCellSavedAnnotations: (run_id: string) =>
    request<SavedAnnotationsResponse>(
      `/api/single_cell/annotations?run_id=${encodeURIComponent(run_id)}`,
    ),
  singleCellPerturbationNarrative: (body: PerturbationNarrativeRequest) =>
    request<NarrativeResponse>('/api/single_cell/perturbation/narrative', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  singleCellGenesetDbs: () => request<GenesetDbsResponse>('/api/single_cell/geneset-dbs'),
  singleCellGenesetTerms: (db: string, q: string) =>
    request<GenesetTermsResponse>(
      `/api/single_cell/geneset-terms?db=${encodeURIComponent(db)}&q=${encodeURIComponent(q)}`,
    ),
  singleCellRunInfo: (run_id: string, top_genes_per_cluster = 50) =>
    request<RunInfoResponse>('/api/single_cell/run-info', {
      method: 'POST',
      body: JSON.stringify({ run_id, top_genes_per_cluster }),
    }),
  singleCellSimilarity: (body: {
    run_id: string
    cluster: string
    k_neighbors: number
    cells_per_cluster: number
  }) =>
    request<SimilarityResponse>('/api/single_cell/similarity', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  singleCellPerturbation: (body: {
    run_id: string
    cluster: string
    perturbation_type: 'knockout' | 'overexpress'
    genes_to_perturb: string[]
  }) =>
    request<PerturbationResponse>('/api/single_cell/perturbation', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  singleCellStart: (body: StartProcessingRequest) =>
    request<StartProcessingResponse>('/api/single_cell/start', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  singleCellRunSummary: (run_id: string, max_umap_points = 10_000) =>
    request<RunSummaryResponse>('/api/single_cell/run-summary', {
      method: 'POST',
      body: JSON.stringify({ run_id, max_umap_points }),
    }),
  singleCellColorPoints: (body: { run_id: string; color_column: string; max_points?: number }) =>
    request<ColorPointsResponse>('/api/single_cell/run-color-points', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  singleCellDotplot: (body: {
    run_id: string
    n_top_genes_per_cluster?: number
    selected_genes?: string[] | null
    scale_data?: boolean
  }) =>
    request<DotplotResponse>('/api/single_cell/run-dotplot', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  singleCellDE: (body: { run_id: string; cluster_a: string; cluster_b: string }) =>
    request<DEResponse>('/api/single_cell/run-de', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  singleCellEnrichment: (body: { run_id: string; cluster: string; dbs: string[] }) =>
    request<EnrichmentResponse>('/api/single_cell/run-enrichment', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  singleCellTrajectory: (body: { run_id: string; gene?: string | null }) =>
    request<TrajectoryResponse>('/api/single_cell/run-trajectory', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  singleCellRawData: (body: { run_id: string; columns: string[]; limit?: number }) =>
    request<RawDataResponse>('/api/single_cell/run-rawdata', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  diffdockExample: () =>
    request<DockingExampleResponse>('/api/small_molecule/diffdock/example'),

  enzymeOptDefaults: () =>
    request<EnzymeDefaultsResponse>('/api/large_molecule/enzyme_optimization/defaults'),
  enzymeOptStart: (body: EnzymeOptimizationStartRequest) =>
    request<EnzymeOptimizationStartResponse>(
      '/api/large_molecule/enzyme_optimization/start',
      { method: 'POST', body: JSON.stringify(body) },
    ),
  enzymeOptSearch: (by: 'run_name' | 'experiment_name', text: string) =>
    request<EnzymeSearchResponse>(
      `/api/large_molecule/enzyme_optimization/search?by=${by}&text=${encodeURIComponent(text)}`,
    ),
  enzymeOptStatus: (run_id: string) =>
    request<EnzymeStatusResponse>(
      `/api/large_molecule/enzyme_optimization/status?run_id=${encodeURIComponent(run_id)}`,
    ),
  enzymeOptTopK: (run_id: string) =>
    request<EnzymeTopKResponse>(
      `/api/large_molecule/enzyme_optimization/top_k?run_id=${encodeURIComponent(run_id)}`,
    ),
  enzymeOptSmokeTest: () =>
    request<EnzymeSmokeTestResponse>(
      '/api/large_molecule/enzyme_optimization/smoke_test',
      { method: 'POST' },
    ),

  // ─── Genomics ────────────────────────────────────────────────────

  variantCallingStart: (body: VariantCallingStartRequest) =>
    request<JobDispatchResponse>('/api/genomics/variant_calling/start', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  variantCallingSearch: (by: 'run_name' | 'experiment_name', text: string) =>
    request<DBSearchResponse>(
      `/api/genomics/variant_calling/search?by=${by}&text=${encodeURIComponent(text)}`,
    ),
  variantCallingSuccessful: () =>
    request<VariantCallingPickerResponse>('/api/genomics/variant_calling/successful'),

  gwasStart: (body: GwasStartRequest) =>
    request<JobDispatchResponse>('/api/genomics/gwas/start', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  gwasSearch: (by: 'run_name' | 'experiment_name', text: string) =>
    request<DBSearchResponse>(
      `/api/genomics/gwas/search?by=${by}&text=${encodeURIComponent(text)}`,
    ),
  gwasResults: (run_id: string) =>
    request<GwasResultsResponse>(
      `/api/genomics/gwas/results?run_id=${encodeURIComponent(run_id)}`,
    ),

  vcfIngestionStart: (body: VcfIngestionStartRequest) =>
    request<JobDispatchResponse>('/api/genomics/vcf_ingestion/start', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  vcfIngestionSearch: (by: 'run_name' | 'experiment_name', text: string) =>
    request<DBSearchResponse>(
      `/api/genomics/vcf_ingestion/search?by=${by}&text=${encodeURIComponent(text)}`,
    ),
  vcfIngestionSuccessful: () =>
    request<VcfIngestionPickerResponse>('/api/genomics/vcf_ingestion/successful'),

  variantAnnotationStart: (body: VariantAnnotationStartRequest) =>
    request<JobDispatchResponse>('/api/genomics/variant_annotation/start', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  variantAnnotationSearch: (by: 'run_name' | 'experiment_name', text: string) =>
    request<DBSearchResponse>(
      `/api/genomics/variant_annotation/search?by=${by}&text=${encodeURIComponent(text)}`,
    ),
  variantAnnotationResults: (run_id: string) =>
    request<VariantAnnotationResultsResponse>(
      `/api/genomics/variant_annotation/results?run_id=${encodeURIComponent(run_id)}`,
    ),
  variantAnnotationDashboard: (run_name?: string) =>
    request<VariantAnnotationDashboardResponse>(
      `/api/genomics/variant_annotation/dashboard${run_name ? `?run_name=${encodeURIComponent(run_name)}` : ''}`,
    ),

  diseaseBiologyRunDetails: (run_id: string) =>
    request<RunDetailsResponse>(
      `/api/genomics/run/details?run_id=${encodeURIComponent(run_id)}`,
    ),
  diseaseBiologyDefaults: () =>
    request<GenomicsDefaultsResponse>('/api/genomics/defaults'),

  // NVIDIA BioNeMo
  bionemoVariants: () => request<BionemoVariantsResponse>('/api/bionemo/variants'),
  bionemoDefaults: () => request<BionemoDefaultsResponse>('/api/bionemo/defaults'),
  bionemoWeights: () => request<BionemoWeightsResponse>('/api/bionemo/weights'),
  bionemoFinetune: (body: BionemoFinetuneRequest) =>
    request<BionemoDispatchResponse>('/api/bionemo/finetune', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  bionemoInference: (body: BionemoInferenceRequest) =>
    request<BionemoDispatchResponse>('/api/bionemo/inference', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  bionemoFinetuneSearch: (by: 'run_name' | 'experiment_name', text: string) =>
    request<DBSearchResponse>(
      `/api/bionemo/finetune/search?by=${by}&text=${encodeURIComponent(text)}`,
    ),
  bionemoFinetuneRunDetails: (run_id: string) =>
    request<BionemoFinetuneRunDetails>(
      `/api/bionemo/finetune/run-details?run_id=${encodeURIComponent(run_id)}`,
    ),
  bionemoInferenceSearch: (by: 'run_name' | 'experiment_name', text: string) =>
    request<DBSearchResponse>(
      `/api/bionemo/inference/search?by=${by}&text=${encodeURIComponent(text)}`,
    ),
  bionemoInferenceRunDetails: (run_id: string) =>
    request<BionemoFinetuneRunDetails>(
      `/api/bionemo/inference/run-details?run_id=${encodeURIComponent(run_id)}`,
    ),
}

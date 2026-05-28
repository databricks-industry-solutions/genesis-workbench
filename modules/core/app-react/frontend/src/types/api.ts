// Manual types — replaceable by openapi-typescript codegen against
// /openapi.json. Keep additions here in sync with backend Pydantic models.

export type HeaderIdentity = {
  email: string | null
  preferred_username: string | null
  user_id: string | null
}

export type SdkIdentity = {
  user_name: string | null
  display_name: string | null
  id: string | null
  active: boolean | null
}

export type MeResponse = {
  from_headers: HeaderIdentity
  from_workspace_client: SdkIdentity
}

export type HealthResponse = { status: string }

export type EnvInfo = {
  catalog: string
  schema_name: string
  warehouse_id: string
  llm_endpoint_name: string | null
  app_name: string | null
  app_service_principal_id: string | null
  admin_usage_dashboard_id: string | null
}

export type BootstrapUser = {
  email: string | null
  preferred_username: string | null
  display_name: string | null
  user_name: string | null
}

export type BootstrapResponse = {
  env: EnvInfo
  user: BootstrapUser
  user_settings: Record<string, string>
  deployed_modules: string[]
}

export type ProfileResponse = {
  email: string
  user_settings: Record<string, string>
}

export type ProfileSaveRequest = {
  user_display_name: string
  mlflow_experiment_folder: string
}

export type MlflowTestResponse = { ok: boolean; message: string }

export type SettingRow = { key: string; value: string; module: string }
export type SystemSettingsResponse = {
  catalog: string
  schema_name: string
  warehouse_id: string
  settings: SettingRow[]
  workflows: SettingRow[]
}

export type EndpointRow = {
  deployment: string
  endpoint: string
  model: string
  status: string
}
export type EndpointStatusResponse = { endpoints: EndpointRow[] }

export type BatchModelRow = {
  model_display_name: string
  model_category: string
  module: string
  cluster_type: string | null
  job_name: string | null
  job_id: string | null
}
export type BatchModelsResponse = { batch_models: BatchModelRow[] }

export type StartEndpointsStatusResponse = {
  active: boolean
  run_id?: number | null
  start_time_iso?: string | null
  duration_hours?: number | null
  remaining_minutes?: number | null
}

export type WorkflowRun = {
  job_id: number
  job_name: string
  run_id: number
  lifecycle_state: string | null
  result_state: string | null
  start_time_ms: number | null
  end_time_ms: number | null
  creator_user_name: string | null
  run_url: string
}
export type WorkflowRunsResponse = { runs: WorkflowRun[] }

export type AdminDashboardResponse = { embed_url: string | null }

export type AssistantQueryResponse = { answer: string }

export type DocEntry = { file: string; title: string; content: string }
export type DocsResponse = { docs: DocEntry[] }

export type AvailableModel = {
  model_id: number
  model_name: string
  model_display_name: string
  model_source_version: string | null
  model_uc_name: string
  model_uc_version: number
}
export type AvailableModelsResponse = { models: AvailableModel[] }

export type DeployedModel = {
  model_id: number
  deployment_id: number
  deployment_name: string
  deployment_description: string | null
  model_display_name: string
  model_source_version: string | null
  uc_name: string
  model_endpoint_name: string
}
export type DeployedModelsResponse = { models: DeployedModel[] }

export type ModuleBatchModel = {
  model_display_name: string
  model_description: string | null
  job_name: string
  cluster_type: string | null
}
export type ModuleBatchModelsResponse = { models: ModuleBatchModel[] }

export type ModuleName =
  | 'single_cell'
  | 'protein_studies'
  | 'small_molecule'
  | 'disease_biology'

export type StructurePredictionResponse = {
  pdb: string
  viewer_html: string
  model: string
}

export type AlphaFoldStartResponse = { job_run_id: string }

export type AlphaFoldRun = {
  run_id: string
  run_name: string
  experiment_name: string
  protein_sequence: string
  start_time_ms: number | null
  status: string
}

export type AlphaFoldSearchResponse = { runs: AlphaFoldRun[] }

export type AlphaFoldResultResponse = { pdb: string; viewer_html: string }

export type SequenceHit = {
  seq_id: string
  description: string
  seq_length: number
  identity_pct: number
  sw_score: number
  alignment_length: number
  vector_distance: number
  aligned_query: string
  aligned_comp: string
  aligned_target: string
}

export type SequenceSearchResponse = { hits: SequenceHit[] }

export type OrganismResponse = { organism: string }

export type InverseFoldingResponse = { sequences: string[] }

export type ProteinDesignResponse = {
  viewer_html: string
  experiment_id: string
  run_id: string
  n_designs: number
}

export type SingleCellRun = {
  run_id: string
  run_name: string
  experiment_name: string
  processing_mode: string
  start_time_ms: number | null
  status: string
  progress: string
  cells: number | null
}
export type SingleCellRunsResponse = { runs: SingleCellRun[] }

export type StartProcessingRequest = {
  mode: 'scanpy' | 'rapids-singlecell'
  data_path: string
  mlflow_experiment: string
  mlflow_run_name: string
  gene_name_column: string
  species: string
  min_genes: number
  min_cells: number
  pct_counts_mt: number
  n_genes_by_counts: number
  target_sum: number
  n_top_genes: number
  n_pcs: number
  cluster_resolution: number
  compute_pseudotime: boolean
}

export type StartProcessingResponse = {
  job_id: number
  job_run_id: number
  mlflow_run_id: string
  experiment_id: string
  run_url: string
}

export type RunSummaryKeyMetric = { label: string; value: string }
export type RunSummaryUmapPoint = { umap_0: number; umap_1: number; cluster: string }
export type RunSummaryResponse = {
  cells_total: number | null
  cells_subsample: number
  clusters_count: number
  markers_count: number
  has_umap: boolean
  has_pseudotime: boolean
  key_metrics: RunSummaryKeyMetric[]
  umap_points: RunSummaryUmapPoint[]
  cluster_col: string
  clusters: string[]
  expr_genes: string[]
  obs_categorical: string[]
  obs_numerical: string[]
  all_columns: string[]
  mlflow_run_url: string | null
}

export type ColorPointsResponse = {
  is_categorical: boolean
  umap_0: number[]
  umap_1: number[]
  values_str: string[] | null
  values_num: number[] | null
}

export type DotplotCell = {
  cluster: string
  gene: string
  expression: number
  size: number
}
export type DotplotResponse = {
  cells: DotplotCell[]
  color_label: string
  color_scale: string
  clusters: string[]
  genes: string[]
}

export type DEGene = {
  gene: string
  log2fc: number
  p_value: number
  p_adj: number
  neg_log10_p_adj: number
  mean_a: number
  mean_b: number
  significant: boolean
}
export type DEResponse = { genes: DEGene[]; n_significant: number }

export type EnrichmentTerm = {
  term: string
  overlap: string
  p_value: number
  p_adj: number
  neg_log10_p_adj: number
  genes: string
  gene_set: string
}
export type EnrichmentResponse = { terms: EnrichmentTerm[]; available_dbs: string[] }

export type TrajectoryUmapPoint = { umap_0: number; umap_1: number; pseudotime: number }
export type TrajectoryGenePoint = { pseudotime: number; expression: number }
export type TrajectoryResponse = {
  has_pseudotime: boolean
  umap_points: TrajectoryUmapPoint[]
  gene_points: TrajectoryGenePoint[]
  genes: string[]
}

export type RawDataResponse = {
  columns: string[]
  rows: Record<string, unknown>[]
  total_cells: number
}

export type ClusterAnnotation = {
  cluster: string
  predicted_cell_type: string
  confidence_pct: number
  top_predictions: string
}

export type UmapPoint = {
  umap_0: number
  umap_1: number
  cluster: string
  predicted_cell_type: string
}

export type AnnotateResponse = {
  annotations: ClusterAnnotation[]
  umap_points: UmapPoint[]
}

export type TeddyClusterAnnotation = {
  cluster: string
  n_cells: number
  predicted_cell_type: string
  cell_type_confidence_pct: number
  cell_type_top3: string
  predicted_disease: string
  disease_confidence_pct: number
  disease_top3: string
}

export type TeddyAnnotateResponse = {
  annotations: TeddyClusterAnnotation[]
  cluster_to_cell_type: Record<string, string>
  cluster_to_disease: Record<string, string>
}

export type SavedAnnotationsResponse = {
  scimilarity: AnnotateResponse | null
  teddy: TeddyAnnotateResponse | null
}

export type GeneEntry = { gene: string; mean_expr: number }

export type RunInfoResponse = {
  cluster_col: string
  clusters: string[]
  n_cells: number
  has_umap: boolean
  top_genes_by_cluster: Record<string, GeneEntry[]>
}

export type CategoryCount = { name: string; count: number }

export type SimilarityResponse = {
  total_neighbors: number
  cell_types: CategoryCount[]
  diseases: CategoryCount[]
  tissues: CategoryCount[]
  sources: CategoryCount[]
}

export type PerturbationGene = {
  gene_name: string
  original_expression: number | null
  predicted_expression: number | null
  delta: number | null
  abs_delta: number | null
}

export type PerturbationResponse = {
  results: PerturbationGene[]
  summary_total_genes: number
  summary_max_abs_delta: number
  summary_significant_count: number
}

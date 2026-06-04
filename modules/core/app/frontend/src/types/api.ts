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
  | 'large_molecule'
  | 'small_molecule'
  | 'genomics'

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
  run_url: string
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
  /** Fraction of the query that was aligned, 0-100. */
  query_coverage_pct: number
  /** Composite ranking score = identity_pct × (coverage / 100). */
  similarity_score: number
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
export type DEResponse = {
  genes: DEGene[]
  n_significant: number
  /** Human-readable notes about data quality — empty on a clean run.
   * Populated when NaN values were dropped, genes were placeholdered, or
   * mannwhitneyu fell back to p=1. The UI surfaces these as an amber
   * banner above the volcano plot so '0' values aren't misread as real
   * signal. */
  warnings?: string[]
}

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
  /** Per-batch embedding failures that were tolerated. Non-empty → some
   * cells were skipped; clusters still annotated using the remaining
   * neighbor data. UMAP sub-tab renders these in an amber banner. */
  warnings?: string[]
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

// ─── Small Molecule ──────────────────────────────────────────────────────

export type DockingExampleResponse = {
  smiles: string
  pdb: string
}

export type DockingPose = {
  rank: number
  confidence: number
  ligand_sdf: string
  viewer_html: string
  error: string | null
}

export type MolecularDockingResponse = {
  poses: DockingPose[]
  experiment_id: string
  run_id: string
  n_success: number
}

export type BinderDesign = {
  sample_id: string
  sequence: string
  rewards: number
  esmfold_validated: boolean
  viewer_html_binder_only: string | null
  viewer_html_with_target: string | null
}

export type BinderDesignResponse = {
  designs: BinderDesign[]
  target_pdb: string
  target_only_viewer_html: string
  experiment_id: string
  run_id: string
  warnings: string[]
}

export type LigandBinderDesign = {
  sample_id: string
  sequence: string
  rewards: number
  esmfold_validated: boolean
  dock_confidence: number | null
  viewer_html_ca_backbone: string | null
  viewer_html_esmfold: string | null
  viewer_html_ca_plus_dock: string | null
  viewer_html_esmfold_plus_dock: string | null
}

export type LigandBinderDesignResponse = {
  designs: LigandBinderDesign[]
  ligand_pdb: string
  experiment_id: string
  run_id: string
  warnings: string[]
}

export type MotifScaffold = {
  sample_id: string
  sequence: string
  mpnn_sequence: string | null
  rewards: number
  esmfold_validated: boolean
  viewer_html: string | null
}

export type MotifScaffoldingResponse = {
  scaffolds: MotifScaffold[]
  motif_pdb: string
  experiment_id: string
  run_id: string
  warnings: string[]
}

export type AdmetResponse = {
  smiles: string[]
  bbbp: (number | null)[] | null
  clintox: (number | null)[] | null
  /** Per-molecule dict keyed by ADMET task name. */
  admet: Record<string, number | null>[] | null
  experiment_id: string
  run_id: string
  warnings: string[]
}

// ─── Enzyme Optimization ──────────────────────────────────────────────────

export type EnzymeRefRow = {
  sequence: string
  half_life_hours: number
  cell_system: string
}

export type EnzymeOptimizationStartRequest = {
  motif_pdb: string
  motif_residues: number[]
  target_chain: string
  scaffold_length_min: number
  scaffold_length_max: number
  num_samples: number
  num_iterations: number
  weights: Record<string, number>
  substrate_smiles: string
  references: EnzymeRefRow[]
  half_life_margin: number
  resampling_temperature: number
  strategy: 'resample' | 'noop'
  run_proteinmpnn: boolean
  convergence_threshold: number | null
  convergence_window: number
  target_reward: number | null
  best_k_target: number | null
  best_k_threshold: number | null
  use_inprocess_ame: boolean
  mlflow_experiment: string
  mlflow_run_name: string
}

export type EnzymeOptimizationStartResponse = {
  job_id: number
  job_run_id: number
  mlflow_run_id: string
  experiment_id: string
  run_url: string
}

export type EnzymeRunRow = {
  run_id: string
  run_name: string
  experiment_name: string
  generation_mode: string
  iter_max_reward: number | null
  iterations_completed: number | null
  start_time_ms: number | null
  job_status: string
  progress: string
  /** Workspace UI link to the dispatched orchestrator-job run. */
  run_url: string
}

export type EnzymeSearchResponse = { runs: EnzymeRunRow[] }

export type EnzymeRewardHistoryPoint = { step: number; value: number }

export type EnzymeStatusResponse = {
  status: string
  job_status: string
  run_name: string
  experiment_id: string
  iter_max_reward_history: EnzymeRewardHistoryPoint[]
  iter_mean_reward_history: EnzymeRewardHistoryPoint[]
  current_metrics: Record<string, number>
  trajectory: Record<string, number | string | null>[]
}

export type EnzymeCandidate = {
  candidate_id: string
  pdb: string
  viewer_html: string
}

export type EnzymeTopKResponse = { candidates: EnzymeCandidate[] }

export type EnzymeSmokeTestResponse = {
  sequence: string
  solubility: number | null
  half_life: number | null
  thermostab: number | null
  immuno: number | null
}

export type EnzymeDefaultsResponse = {
  motif_pdb: string
  default_weights: Record<string, number>
  default_references: EnzymeRefRow[]
}

// ─── Genomics ──────────────────────────────────────────────────────

export type DBRunRow = {
  run_id: string
  run_name: string
  experiment_name: string
  status: string
  progress: string
  start_time_ms: number | null
  detail: string
  /** Workspace UI link to the dispatched job's run page; empty if unavailable. */
  run_url: string
}

export type DBSearchResponse = { runs: DBRunRow[] }

export type JobDispatchResponse = {
  job_run_id: number
  run_url: string
}

export type RunDetailsResponse = {
  run_name: string
  experiment_id: string
  status: string
  job_status: string
  job_run_id: string
  params: Record<string, string>
  tags: Record<string, string>
}

export type VariantCallingStartRequest = {
  fastq_r1: string
  fastq_r2: string
  reference_genome_path: string
  output_volume_path: string
  mlflow_experiment: string
  mlflow_run_name: string
}

export type VariantCallingPickerRow = {
  run_id: string
  run_name: string
  experiment_name: string
  output_vcf: string
  start_time_ms: number | null
}

export type VariantCallingPickerResponse = { runs: VariantCallingPickerRow[] }

export type GwasStartRequest = {
  vcf_path: string
  phenotype_path: string
  phenotype_column: string
  contigs: string
  hwe_cutoff: string
  pvalue_threshold: string
  mlflow_experiment: string
  mlflow_run_name: string
}

export type GwasHit = {
  contig: string
  position: number
  pvalue: number
  neg_log_pval: number | null
  reference_allele: string
  alternate_alleles: string
  effect: number | null
  phenotype: string | null
}

export type GwasResultsResponse = {
  total_variants: number
  significant_count: number
  min_pvalue: number | null
  top_hits: GwasHit[]
  manhattan_points: { x: number; y: number }[]
}

export type VcfIngestionStartRequest = {
  vcf_path: string
  output_table_name: string
  mlflow_experiment: string
  mlflow_run_name: string
}

export type VcfIngestionPickerRow = {
  run_id: string
  run_name: string
  experiment_name: string
  output_table: string
  start_time_ms: number | null
}

export type VcfIngestionPickerResponse = { runs: VcfIngestionPickerRow[] }

export type VariantAnnotationStartRequest = {
  variants_table: string
  gene_regions: string
  pathogenic_vcf_path: string
  gene_panel_mode: 'custom' | 'acmg'
  mlflow_experiment: string
  mlflow_run_name: string
}

export type AnnotationVariant = {
  gene: string
  chromosome: string
  position: number
  ref: string
  alt: string
  zygosity: string | null
  clinical_significance: string | null
  disease_name: string | null
  category: string | null
  condition: string | null
}

export type VariantAnnotationResultsResponse = {
  variants: AnnotationVariant[]
  total: number
}

export type VariantAnnotationDashboardResponse = {
  embed_url: string
  run_name: string | null
}

export type GenomicsDefaultsResponse = {
  variant_calling: {
    fastq_r1: string
    fastq_r2: string
    reference_genome_path: string
    output_volume_path: string
  }
  gwas: {
    vcf_path: string
    phenotype_path: string
    phenotype_column: string
    contigs: string
    hwe_cutoff: string
    pvalue_threshold: string
  }
  vcf_ingestion: {
    vcf_path: string
  }
}

// ─── NVIDIA BioNeMo ──────────────────────────────────────────────────────────

export type BionemoVariantsResponse = { esm2: string[] }

export type BionemoWeight = {
  ft_id: number
  ft_label: string
  variant: string
  model_type: string
  experiment_name: string | null
  run_id: string | null
  created_by: string | null
  created_datetime: string | null
}
export type BionemoWeightsResponse = { weights: BionemoWeight[] }

export type BionemoDispatchResponse = { job_run_id: number; run_url: string }

export type BionemoFinetuneRequest = {
  esm_variant: string
  train_data: string
  evaluation_data: string
  finetune_label: string
  experiment_name: string
  should_use_lora: boolean
  task_type: string
  num_steps: number
  micro_batch_size: number
  precision: string
  mlp_ft_dropout: number
  mlp_hidden_size: number
  mlp_target_size: number
  mlp_lr: number
  mlp_lr_multiplier: number
}

export type BionemoInferenceRequest = {
  esm_variant: string
  is_base_model: boolean
  finetune_run_id: number
  task_type: string
  data_location: string
  sequence_column_name: string
  result_location: string
}

export type BionemoDefaultsResponse = {
  train_data: string
  evaluation_data: string
  inference_data: string
  sequence_column: string
  result_location: string
}

export type BionemoFinetuneRunDetails = {
  run_name: string
  status: string
  job_status: string
  result_location: string
  job_run_id: string
  params: Record<string, string>
  metrics: Record<string, number>
}

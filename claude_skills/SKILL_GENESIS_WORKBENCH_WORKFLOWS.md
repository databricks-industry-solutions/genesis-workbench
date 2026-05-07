---
name: genesis-workbench-workflows
description: End-to-end user guide for all Genesis Workbench UI workflows — what each tab does, required inputs, expected outputs, and how to interpret results across protein studies, single cell, small molecule, and disease biology modules.
---

# Genesis Workbench Workflows Skill

Guide users through each workflow tab in the Genesis Workbench Streamlit application.

## Protein Studies

### Structure Prediction
**Tab:** Protein Studies → Protein Structure Prediction

Three models available via pills selector:

**ESMFold (real-time):**
- Input: Amino acid sequence (single chain)
- Output: 3D structure displayed in Mol* viewer
- Speed: Seconds
- Best for: Quick single-chain structure prediction

**AlphaFold2 (batch job):**
- Input: Amino acid sequence + MLflow experiment/run name
- Output: PDB structure stored in MLflow (view via search)
- Speed: Hours (MSA + template search + folding)
- Best for: High-accuracy prediction when time is not a constraint
- Workflow: Start Job → Search Past Runs → View Result

**Boltz (real-time):**
- Input: Amino acid sequence (auto-formatted as `protein_A:SEQUENCE`)
- Also supports multi-chain: `protein_A:SEQ1;rna_B:SEQ2;smiles_C:CCCC`
- Output: 3D structure in Mol* viewer
- Best for: Multi-chain complexes (protein-protein, protein-ligand, protein-RNA)

### Protein Design
**Tab:** Protein Studies → Protein Design

- Input: Sequence with region to replace marked by `[brackets]`, e.g., `CASRRSG[FTYPGF]FFEQYF`
- Also needs: MLflow experiment name and run name
- Pipeline: ESMFold → RFDiffusion inpainting → ProteinMPNN → ESMFold validation
- Output: Overlaid original + designed structures in Mol* viewer
- Progress: Step-by-step progress bar

### Inverse Folding
**Tab:** Protein Studies → Inverse Folding

- Input: PDB content (backbone structure pasted into text area, default provided)
- Output: 3 designed amino acid sequences (selectbox to browse)
- Auto-validation: ESMFold automatically folds the selected design and shows the predicted structure
- Use case: Protein engineering — design new sequences for an existing backbone

### Sequence Search
**Tab:** Protein Studies → Sequence Search

- Input: Protein sequence (paste or FASTA upload)
- Pipeline: ESM2 embedding → Vector Search (top 500) → Smith-Waterman alignment → Ranked results
- Output: Table with sequence identity %, alignment score, organism, vector distance
- Can view individual hit structures via ESMFold

## Single Cell

### Raw Single Cell Processing
**Tab:** Single Cell → Raw Single Cell Processing → Run New Analysis

- Input: Path to h5ad file in Unity Catalog Volumes
- Parameters: Gene name column (or species for Ensembl mapping), QC thresholds (min_genes, min_cells, pct_mt, n_genes_by_counts), normalization (target_sum, n_top_genes), clustering (n_pcs, resolution), optional pseudotime
- Modes: Scanpy (CPU) or Rapids-SingleCell (GPU-accelerated)
- Output: MLflow run with markers_flat.parquet, UMAP, marker genes, QC plots

**View Analysis Results** (sub-tab):
- Select a completed run → Load → Interactive UMAP (color by cluster/gene/metric)
- Marker Gene Dot Plot: Top genes per cluster ranked by Wilcoxon test
- Differential Expression: Pick 2 clusters → volcano plot + significant genes table
- Pathway Enrichment: Pick cluster → GO/KEGG/Reactome bar chart via Enrichr
- Trajectory Analysis: UMAP colored by pseudotime + gene expression along pseudotime (requires "Compute Pseudotime" enabled)
- QC Outputs: Link to full MLflow run with all plots
- Data Table: Browse raw data, download CSV

### Cell Type Annotation
**Tab:** Single Cell → Cell Type Annotation

- Input: Select a completed processing run
- Parameters: Cells per cluster (default 10), Neighbors k (default 20)
- Pipeline: Gene order → align + lognorm → batch embeddings → nearest neighbor search → majority vote
- Output: Table (Cluster → Predicted Cell Type → Confidence → Top 3 predictions) + UMAP colored by cell type
- Requires: SCimilarity endpoints deployed and active

### Cell Similarity Search
**Tab:** Single Cell → Cell Similarity

- Input: Select a completed processing run → Load → Pick a cluster
- Parameters: Neighbors k (default 100)
- Pipeline: Same as annotation but shows full neighbor analysis instead of just cell types
- Output: Bar charts (neighbor cell type distribution, disease distribution), study sources table, full results table
- Requires: SCimilarity endpoints deployed and active

### Perturbation Prediction
**Tab:** Single Cell → Perturbation Prediction

- Input: Select a completed processing run → Pick a cluster → Select genes to perturb (multiselect ranked by expression, or type custom genes) → Choose Knockout or Overexpress
- Pipeline: Mean cluster expression → scGPT perturbation endpoint → predicted expression changes
- Output: Bar chart (top 20 affected genes by delta), scatter plot (original vs predicted), summary metrics, full results table
- Requires: scGPT Perturbation endpoint deployed

## Small Molecule

### Binder Design
**Tab:** Small Molecules → Protein Binder Design

- Input: Target protein (sequence or PDB) + binder length range + num samples
- Optional validation: ESMFold structure validation
- Output: Design selector (reward scores), Mol* viewer

### Ligand Binder Design
**Tab:** Small Molecules → Ligand Binder Design

- Input: Target protein PDB + ligand (SMILES or PDB) + binder params
- Pipeline: SMILES → PDB conversion (rdkit) → Proteina-Complexa-Ligand → optional ESMFold + DiffDock validation
- Output: Multi-view display (backbone, full protein, protein+ligand)

### Motif Scaffolding
**Tab:** Small Molecules → Motif Scaffolding

- Input: Motif PDB + scaffold params
- Pipeline: Proteina-Complexa-AME → ProteinMPNN optimization → ESMFold validation
- Output: Design selector with Mol* viewer

### Guided Enzyme Optimization
**Tab:** Small Molecules → Guided Enzyme Optimization

A reward-weighted resampling loop on top of Motif Scaffolding's stack — instead of accepting whatever AME produces, it iterates and biases each round toward higher-reward parents. The form exposes a **Generation mode** toggle (Fast / Accurate, default Fast) that picks where the reward signal applies.

- **Generation mode (Fast vs Accurate):**
  - **Fast (~30 min, no GPU cost)** — AME runs as the deployed Model Serving endpoint; the loop scores K candidates after generation and resamples parents by reward for the next iteration. Reward signal applies *between* iterations only. Job: `run_enzyme_optimization_gwb` on a CPU cluster.
  - **Accurate (~30-60 min, ~$22 GPU cost)** — AME loads on an A10 GPU cluster (checkpoints pulled from the registered UC model `{catalog}.{schema}.proteina_complexa_ame`, version resolved from the `models` table) and uses Feynman-Kac steering during sampling: at intermediate denoising steps, partial structures are scored and trajectories are importance-sampled so losing branches get pruned early. Reward signal applies *during* diffusion. Job: `run_enzyme_optimization_gwb_inprocess_ame` on an A10 GPU cluster. Both jobs share the same notebook; the toggle drives the dispatcher to pick the right job by name.
  - Verified side-by-side on K=4, N=2, all-axes-weight-1.0: Fast `iter_mean=0.449`, Accurate `iter_mean=0.506` — first signal that FK-steering's importance-weighted resampling concentrates reward more uniformly across the batch.

- **Inputs:**
  - Motif PDB + chain id + motif residue numbers (CSV) — same as Motif Scaffolding
  - Scaffold length range, K (candidates per iteration), N (iterations)
  - Substrate SMILES (optional — gates the Boltz axis)
  - Reference enzymes (1+ rows in a `data_editor`): sequence + measured half-life in hours + cell system. Used to anchor the half-life axis.
  - Half-life margin (default 0.05) — how far above `min(reference PLTNUM)` a candidate must score for positive half-life reward
  - Per-axis weight sliders: `motif_rmsd`, `plddt`, `boltz`, `solubility`, `half_life`, `thermostab`, `immuno` (0–5; weight 0 disables that axis)
  - **Generation mode** radio (Fast / Accurate, default Fast)
  - Strategy radio: `resample` (default) or `noop` (verification)
  - Resampling temperature, ProteinMPNN checkbox, MLflow experiment + run name
- **Pipeline (per iteration):**
  1. AME → K scaffolds. **Fast**: SDK call to `gwb_*_proteina_complexa_ame_endpoint`. **Accurate**: in-process `Proteina.load_from_checkpoint(...)` + FK-steering search with `DevelopabilityCompositeReward` attached via `model.reward_model = reward`.
  2. (Optional) ProteinMPNN redesign with `fixed_positions={"A": motif_residues}` to preserve catalytic motif identity → ESMFold re-fold
  3. Score each candidate via the predictor registry (motif backbone RMSD, ESMFold pLDDT, Boltz substrate confidence, NetSolP solubility, PLTNUM anchor-relative half-life, DeepSTABp Tm, MHCflurry immuno burden)
  4. Compose a weighted reward (z-score+min-max within batch; half-life is pre-normalized via the anchor sigmoid)
  5. Log per-candidate metrics + PDBs to MLflow; resample parents for the next iteration
- **Endpoint warmup at job start (both paths):** the orchestrator pre-warms NetSolP, PLTNUM, DeepSTABp, MHCflurry (via `warmup_developability_endpoints`) and AME / ESMFold / ProteinMPNN (via `warmup_generation_endpoints`) with a 1-call dummy hit each. Without it, scale-to-zero cold starts of 5-20 min mid-loop bust the request timeout.
- **Outputs:**
  - Live `iter_max_reward` / `iter_mean_reward` line chart while the loop runs
  - Top-25 candidates table with all per-axis scores
  - PDB selector + Mol* viewer overlaying the input motif on the chosen designed scaffold
  - Top-K PDBs and a full reward trajectory CSV in MLflow
  - MLflow params include `generation_mode` (Fast/Accurate) + `use_inprocess_ame` (bool) + the FK-steering knobs when Accurate, so two runs can be diffed against each other
- **Honest caveats:**
  - Half-life reward is *anchor-relative*, not in hours. Top candidates are predicted to be at least as long-lived as your reference enzyme + margin. With no reference, the axis falls back to a neutral 0.5 contribution.
  - The Boltz axis only activates if a substrate SMILES is supplied.
  - First-iteration AME generation in **Fast** isn't bias-influenced; the reward signal kicks in from iteration 2. **Accurate** mode applies the reward inside iteration 1's AME generation via FK-steering.
  - **Accurate** depends on `proteina_complexa/proteina_complexa_v1` being deployed first — its UC model is the only source for the AME checkpoints. If not deployed, `_fetch_ame_checkpoints_from_uc` raises with a clear "deploy proteina_complexa first" message (no NGC fallback).
- **Smoke test:** an "Test predictors on T4 lysozyme" expander on the form runs a single sequence through all four developability endpoints — useful to sanity-check the deployment before kicking off a full loop.

### ADMET & Safety
**Tab:** Small Molecules → ADMET & Safety

- Input: SMILES strings (one per line)
- Pipeline: Chemprop BBBP + ClinTox + ADMET endpoints (independent, parallel scoring)
- Output: Risk cards (green/orange/red) per molecule, expandable property details

## Disease Biology

### VCF Ingestion
- Input: VCF file path in Volumes
- Output: Delta table with genomic variants

### Variant Annotation
- Input: Ingested VCF Delta table
- Pipeline: ClinVar annotation + gene filtering (e.g., BRCA)
- Output: Annotated variants with clinical significance

### GWAS Analysis
- Input: Genotype + phenotype data
- Output: Manhattan plot, QQ plot, significant associations

## Settings

### Endpoint Management
- View real-time endpoint status (Ready/Starting/Scaled to zero/Failed)
- Refresh button to re-check statuses
- Start All Endpoints: Keep endpoints alive for 1-12 hours (prevents scale-to-zero during demos)

## Instructions

1. When a user asks "how do I predict protein structure?" → guide them to Protein Studies → Structure Prediction, recommend ESMFold for quick results or AlphaFold2 for accuracy.
2. When a user asks about cell types → guide them to Single Cell → Cell Type Annotation (requires a completed processing run first).
3. When a user asks about drug-target interactions → guide them to Small Molecules → Ligand Binder Design or ADMET & Safety.
4. When a user asks about gene effects → guide them to Single Cell → Perturbation Prediction.
5. For demos, recommend starting all endpoints first (Settings → Start All Endpoints) to avoid scale-from-zero delays.
6. Always mention that batch workflows (AlphaFold2, Scanpy/Rapids processing) take time — start them first, then demo real-time workflows while waiting.

## When to Use This Skill

- User asks how to use a specific Genesis Workbench feature
- User is preparing a demo and needs a walkthrough
- User wants to know what inputs a workflow needs
- User is confused about which tab or model to use for their task
- User asks about interpreting results (volcano plots, UMAP, enrichment charts)

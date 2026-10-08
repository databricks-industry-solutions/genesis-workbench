---
name: gwb-batch-jobs
description: Dispatch a long-running Genesis Workbench batch workflow from a notebook via the Databricks Jobs API, then poll it and read results from MLflow, a Delta table, or a UC Volume. Use for life-sciences workflows that are too long or heavy for a synchronous endpoint call and run as a GWB job — guided enzyme optimization, GPU variant calling (Parabricks), VCF ingestion, GWAS, model (re)registration and reference (re)builds. Triggers on "run the enzyme optimization", "launch/kick off a batch job", "variant calling", "parabricks", "GWAS", "run it as a job and check the results", "long-running workflow".
---

# Dispatch Genesis Workbench batch workflows from a notebook

Some GWB capabilities are **jobs**, not endpoints: they run for minutes-to-hours, use serverless-GPU or
classic compute, and write results to MLflow + Delta/Volumes. The GWB app launches these from its UI
("form → job → MLflow run → search past runs → result dialog"); from a notebook you do the same with the
Jobs API. Use an endpoint for a synchronous call (see **gwb-serving-endpoints**); use a job when the work
is long/multi-step or needs GPU batch compute.

## Pattern
```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# 1) find the job by name (GWB job names are stable; the app also stores IDs in the settings table)
jobs = [j for j in w.jobs.list(expand_tasks=False) if j.settings.name == "run_enzyme_optimization_gwb"]
job_id = jobs[0].job_id

# 2) dispatch with the job's declared parameters (job_parameters = the name:default pairs on the job)
run = w.jobs.run_now(job_id, job_parameters={
    "catalog": "main", "schema": "genesis_workbench",
    "mlflow_experiment": "/Users/me/enzyme_opt", "mlflow_run_name": "my_run",
    "motif_pdb_path": "/Volumes/.../motif.pdb", "motif_residues_csv": "A:10,A:11,A:12",
    "substrate_smiles": "CC(=O)Oc1ccccc1C(=O)O", "num_samples": "4", "num_iterations": "2",
    "weights_json": '{"plddt":1.3,"solubility":1.0,"thermostab":1.0,"immuno":1.5}',
})
run_id = run.run_id

# 3) poll to terminal
import time
while True:
    st = w.jobs.get_run(run_id).state
    print(st.life_cycle_state, st.result_state)
    if st.life_cycle_state.value in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
        break
    time.sleep(60)
```

## Reading results
- **MLflow** (most GWB workflows log here): find the run by experiment + run name, then read metrics +
  artifacts.
  ```python
  import mlflow
  df = mlflow.search_runs(experiment_names=["/Users/me/enzyme_opt"],
                          filter_string="tags.mlflow.runName = 'my_run'")
  # artifacts (e.g. reward_trajectory.csv, topK_pdbs/) via mlflow.artifacts.download_artifacts(run_id=...)
  ```
- **Delta tables** — workflows that build a reference/table (e.g. a VS source) write a managed table you
  query with Spark/SQL.
- **UC Volume** — large per-run outputs (PDBs, CSVs, VCFs) land under the module's cache Volume.

## GWB jobs you can dispatch
- **`run_enzyme_optimization_gwb`** (Fast, serverless CPU) / **`run_enzyme_optimization_gwb_inprocess_ame`**
  (Accurate, serverless GPU) — reward-weighted motif/enzyme optimization; needs `motif_pdb_path` +
  `substrate_smiles`, scores on pLDDT / motif-RMSD / optional Boltz / developability (`weights_json`).
  Accurate adds Feynman-Kac steering (`fk_*` params); hours-long.
- **`parabricks_alignment` / `vcf_ingestion_glow` / `gwas_glow_analysis`** — genomics (GPU align, Glow
  VCF→Delta, GWAS). The GWAS/VCF/variant-annotation setup jobs are classic (`%sh`/Glow); the rest serverless.
- **`register_*` / `sequence_search_workflow`** — model (re)registration + reference/VS (re)builds; mostly
  idempotent (self-skip when already built).

Discover what's installed with `w.jobs.list()` (or see **gwb-discover-resources**).

## Notes
- Jobs are **not** reaped (only idle serving/VS endpoints are). A running job survives.
- A GPU task may fail with a transient `INTERNAL_ERROR` ("compute failed to start within 900s") — just
  re-run; it's a provisioning flake, not your payload.
- `job_parameters=` sets job-level parameters by name; use `notebook_params=` only for a single-notebook
  job that reads `dbutils.widgets`. Pass values as strings.

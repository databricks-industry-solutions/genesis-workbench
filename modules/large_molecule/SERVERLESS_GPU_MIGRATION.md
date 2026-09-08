# Large Molecule — Serverless GPU migration

## Why

On workspaces whose AWS account has **no GPU vCPU quota** (e.g. locked lab/workshop
environments), the classic single-node GPU **register jobs** fail at cluster launch with:

```
AWS_RESOURCE_QUOTA_EXCEEDED / VcpuLimitExceeded (current vCPU limit of 0)
```

**Serverless GPU** compute draws from a Databricks-managed pool rather than the account's
EC2 vCPU quota, so it sidesteps this block entirely. `esm2_embeddings` already used this
pattern; this change extends it to the other GPU register jobs.

## The conversion pattern (per submodule)

Mirrors `esm2_embeddings/esm2_embeddings_v1`:

1. **`resources/<register>.job.yml`** — on the register task: drop `job_cluster_key`, add
   `environment_key: gpu_env` + `compute: {hardware_accelerator: GPU_1xA10}` (plus
   `disable_auto_optimization: true`, `retry_on_timeout: false`). Delete the `job_clusters:`
   block. Add a job-level `environments: [{environment_key: gpu_env, spec: {client: "4"}}]`.
2. **`databricks.yml`** — delete the per-target (`prod_aws/prod_azure/prod_gcp`)
   `resources.jobs.<job>.job_clusters` cloud-attribute overrides (serverless needs none).
3. **Register notebook** — the serverless "AI runtime" env is **Python 3.12 / torch 2.7 /
   CUDA 12.6** with torch+CUDA and mlflow preinstalled, but **not** transformers/accelerate —
   `%pip install` those in-notebook. Do **not** `pip install torch` (use the runtime's) and do
   **not** `%pip install -U mlflow` (the runtime's mlflow 2.x is required; 3.x breaks UC logging).
   Valid accelerators: `GPU_1xA10`, `GPU_8xH100`.

Confirmed: `GPU_1xA10` provisions and runs on this workspace (us-west-2), bypassing the quota.

## Large model-weight handling (registration → serving)

Databricks Model Serving **requires** weights be **packaged into the model artifact**
(`artifacts={"model": "/Volumes/..."}`); it does not read `/Volumes` at serving runtime.
Two traps when bundling multi-GB weights, both fixed here for esmfold:

- **Do not log the raw HuggingFace cache dir.** Its symlink+blob structure makes MLflow upload
  the weights twice (~2× size, many files) → guaranteed 5-min upload timeout. Instead
  `model.save_pretrained(flat_dir)` (a single clean `safetensors`) and log that.
- **Save/download to fast local temp (`tempfile.mkdtemp()`), not a `/Volumes` FUSE path.**
  `save_pretrained`/`from_pretrained(cache_dir=...)` of ~2.8 GB through FUSE is pathologically
  slow (stretched runs to ~40 min). Serverless has no writable `/local_disk0` — use tempfile.

A standalone log of a single clean 2.7 GB artifact uploads in ~50 s.

## ✅ SOLVED: the `Timed out after 0:05:00` UC-registration cap

The 5-min registration timeout was **not** infrastructure throttling. MLflow's UC model
upload dispatches (in `mlflow/utils/_unity_catalog_utils.py::is_databricks_sdk_models_artifact_repository_enabled`)
to the **Databricks-SDK models artifact repo**, which wraps the whole upload in the SDK's
retry with a **5-min `retry_timeout_seconds`** → `TimeoutError('Timed out after 0:05:00')` on
any multi-GB model. (`MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT` does not help — it is `None` on
AWS. The `python_model.pkl` "timing out" was a red herring: the 5-min alarm just fired on
whichever small file happened to be in flight.)

**Fix — one env var, set before `log_model`:**

```python
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "false"
```

When that var is *defined*, the dispatch function returns it directly, routing the upload to
the **PresignedUrlArtifactRepository (direct S3, boto3 multipart, no 5-min cap)**. Combined
with the fp16/single-shard flat artifact (fast upload), esmfold registers cleanly. This is THE
fix for every large GWB model's registration.

## ✅ SOLVED: serving-container load failure = pip_requirements version mismatch

After registration succeeds, the GPU serving endpoint can still fail with
`DEPLOYMENT_FAILED — "Model server failed to load the model" / "Phase: mlflow_parse ... builtins.AttributeError"`.
Cause: the model is logged on the serverless runtime (**mlflow 2.22.0, cloudpickle 3.0.0**)
but `log_model(pip_requirements=[...])` pinned OLD versions (mlflow 2.15.1 / cloudpickle 2.2.1),
so the serving container can't parse/unpickle it. **Fix: pin `mlflow==2.22.0` + `cloudpickle==3.0.0`
in `pip_requirements`** to match the logging runtime. (Keeping `torch==2.3.1+cu121` in the
serving deps loads the safetensors fine and is safer for the container's CUDA.) Inspect the
load error via `databricks api get /api/2.0/serving-endpoints/<ep>/events` (CLI `build-logs`/`logs`
fail on a failed pending config).

## Per-model status

| Model | Serverless-GPU job | Notes |
|---|---|---|
| esm2_embeddings | ✅ (pre-existing) | **live** endpoint |
| esmfold | ✅ **live end-to-end** | `gwb_demo_esmfold_endpoint` READY on GPU_MEDIUM, folds a sequence → PDB. Full register+endpoint build ≈ 42 min. Applies all fixes above (serverless GPU + local-temp + fp16 single-shard + presigned-upload env var + aligned pip_requirements) |
| boltz | ✅ **live end-to-end** (**Boltz-2**, `boltz==2.2.1`) | `gwb_demo_boltz_endpoint` READY on GPU_MEDIUM; sequence → PDB + confidence. Weights auto-download to local temp + packaged in; `--no_kernels`. |
| protein_mpnn | ✅ **live end-to-end** (ported off `torch 1.11`) | `gwb_demo_proteinmpnn_endpoint` READY on GPU_SMALL; backbone PDB → designed sequences. Modern torch 2.3.1+cu121, `weights_only=False`. |
| rfdiffusion | ✅ **live end-to-end** (**RFD3 via rc-foundry**) | `gwb_demo_rfdiffusion_inpainting_endpoint` READY on GPU_MEDIUM; inpainting preserved (`{pdb,start_idx,end_idx}` → backbone PDB); `rfdiffusion_unconditional` dropped (unused). Inline pyfunc (`python=3.12`), checkpoint via `foundry install rfd3` packaged in, `--extra-index-url https://pypi.org/simple` for rc-foundry, mmCIF→backbone PDB via Biopython. |

## Two more serving-build fixes (found porting boltz + protein_mpnn)

1. **Never pip-install the model's own package via a `- /model/artifacts/<pkg>` conda line.** That absolute path does not exist during the serving image build (it runs with the model at `./model`) → the container build fails instantly with `ambiguous_other`. Instead ship the package with mlflow **`code_paths`** and list its deps as normal PyPI packages (or define the pyfunc **inline** in the notebook). `rc-foundry` isn't on the private pypi-proxy → add `--extra-index-url https://pypi.org/simple` in the serving env (verified the build reaches it).
2. **Serving `python` must match the logging runtime (3.12).** An **inline** pyfunc class is cloudpickled by value (3.12 bytecode); a serving container pinned to 3.11 segfaults the model server (`Worker … code 139!`) in a crash loop until the deploy times out. Pin `conda_env` `python=3.12` (packaged-class models like boltz are immune — pickled by reference). Bump `wait_for_job_run_completion` to `7200`.

## RFdiffusion3 adoption (proven PoC — follow-up wiring)

GWB only uses RFdiffusion's single-chain **motif inpainting** (`rfdiffusion_inpainting`:
`{pdb,start_idx,end_idx}` → backbone PDB for ProteinMPNN; `rfdiffusion_unconditional` is
unused). RFdiffusion3 (all-atom) covers this and installs/runs on serverless GPU:

- `pip install rc-foundry` (BSD-3); `foundry install rfd3 -d <UC Volume>` (`rfd3_latest.ckpt`, 2.7 GB, ~63 s).
- Inference: `rfd3 design out_dir=<o> inputs=<in.json>` with
  `{"name": {"input": "<pdb>", "contig": "A1-{s-1},{e-s+1},A{e+1}-{N}", "select_fixed_atoms": "A1-{s-1},A{e+1}-{N}"}}`.
  Verified: motif inpainting ran in ~16 s on A10G, output gzipped **mmCIF** → wrapper converts to
  backbone PDB to preserve the endpoint contract (no app/executor/node/UI changes needed).
- ✅ Wired end-to-end: register notebook (`01_register_rfdiffusion.py`) logs the inline
  `RFD3Inpainting` pyfunc, `02_import_model_gwb.py` imports + deploys, endpoint verified READY.

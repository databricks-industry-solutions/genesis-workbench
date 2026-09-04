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

## ⚠️ Known blocker on quota-locked lab workspaces: UC model-registry upload cap

Even after the fixes above, **full model registration** of a large model to the UC model
registry on this workshop workspace **times out at exactly `0:05:00`** — after the big
`safetensors` uploads, the tiny `python_model.pkl` future is starved past its 5-min window
during the concurrent multi-file upload (`MlflowException: TimeoutError('Timed out after
0:05:00')`). `MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT` does **not** override it.

This is **workspace-infrastructure behavior** (a standalone 2.7 GB upload succeeds; full
multi-file registration does not), not a code defect — it is expected to work on a standard
workspace with normal model-registry upload throughput. Follow-ups to try: force sequential
artifact upload / reduce concurrency, shrink the logged pyfunc, or register from a workspace
without the throttled upload path.

## Per-model status

| Model | Serverless-GPU job | Notes |
|---|---|---|
| esm2_embeddings | ✅ (pre-existing) | **live** endpoint |
| esmfold | ✅ | flat-artifact + local-temp handling done; serving registration blocked by the upload cap above on this workspace |
| boltz | ✅ + dep bump to **Boltz-2** (`boltz==2.2.1`, dropped `flash_attn==1.0.9`) | notebook still needs: remove the `%sh` miniconda/jackhmmer step (not serverless-compatible) and switch to boltz2 weights via `BOLTZ_CACHE` |
| protein_mpnn | pending | port off `torch==1.11+cu113` (no py3.12 wheel) → `proteinmpnn` (foundry supplies weights) |
| rfdiffusion | pending | **RFdiffusion3 adoption PoC proven** — see below |

## RFdiffusion3 adoption (proven PoC — follow-up wiring)

GWB only uses RFdiffusion's single-chain **motif inpainting** (`rfdiffusion_inpainting`:
`{pdb,start_idx,end_idx}` → backbone PDB for ProteinMPNN; `rfdiffusion_unconditional` is
unused). RFdiffusion3 (all-atom) covers this and installs/runs on serverless GPU:

- `pip install rc-foundry` (BSD-3); `foundry install rfd3 -d <UC Volume>` (`rfd3_latest.ckpt`, 2.7 GB, ~63 s).
- Inference: `rfd3 design out_dir=<o> inputs=<in.json>` with
  `{"name": {"input": "<pdb>", "contig": "A1-{s-1},{e-s+1},A{e+1}-{N}", "select_fixed_atoms": "A1-{s-1},A{e+1}-{N}"}}`.
  Verified: motif inpainting ran in ~16 s on A10G, output gzipped **mmCIF** → wrapper converts to
  backbone PDB to preserve the endpoint contract (no app/executor/node/UI changes needed).
- Wiring is pending because it lands in the same serving-registration path blocked above.

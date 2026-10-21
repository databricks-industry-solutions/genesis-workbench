---
name: gwb-serverless-gpu
description: Run GPU compute for a life-sciences / bio-ML workflow in a Databricks notebook or job on serverless GPU (the AI Runtime) — no classic cluster to provision and no EC2 GPU vCPU quota. Use whenever a Genesis Workbench user needs to load a model on an A10, run a GPU notebook, or wire a serverless-GPU job task — and for the HuggingFace / torch / disk gotchas that bite bio models. Triggers on "run on GPU", "serverless GPU", "A10", "load the model on GPU", "GPU without a cluster", "bypass GPU quota" in a Genesis Workbench workspace. Pairs with gwb-ray-batch-inference for multi-node scaling.
---

# Serverless GPU (AI Runtime) for Genesis Workbench workflows

Genesis Workbench runs its GPU model registration, embedding, and serving on **serverless GPU** (the
Databricks AI Runtime), not classic GPU clusters. That's why GWB installs even where the account's EC2
GPU vCPU quota is 0 — serverless GPU draws from a managed pool. Use this skill to get a GPU for your own
bio-ML work (fold a protein, embed sequences/cells, score molecules) the same way.

## Get a GPU in an interactive notebook
Attach the notebook to **Serverless GPU** compute (the compute selector → Serverless → GPU, A10), then:
```python
# torch + CUDA are PREINSTALLED on the AI Runtime — do NOT pip-install/pin torch (a reinstall
# risks a cuDNN/driver mismatch). Add only your extra deps, then restart Python.
%pip install -q transformers==4.41.2 pyarrow==15.0.2 hf_transfer==0.1.9
dbutils.library.restartPython()
```
```python
import torch
print(torch.cuda.get_device_name(0))   # e.g. NVIDIA A10
```

## Get a GPU in a job (DABs / one-off run)
A serverless-GPU task declares a serverless `environment` and a GPU accelerator — no `job_clusters`,
no `spark_version`, no `node_type_id`:
```yaml
tasks:
  - task_key: embed
    environment_key: gpu_env
    compute:
      hardware_accelerator: GPU_1xA10      # single A10; the only GWB GPU shape today
    notebook_task:
      notebook_path: ../notebooks/embed.py
environments:
  - environment_key: gpu_env
    spec:
      client: '4'      # '2' = Python 3.11, '4' = Python 3.12
```
One-off submit from a notebook (handy for ad-hoc GPU runs):
```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
run = w.jobs.submit(run_name="adhoc-gpu", tasks=[{
    "task_key": "embed",
    "notebook_task": {"notebook_path": "/Users/me/embed", "base_parameters": {}},
    "environment_key": "gpu_env",
    "compute": {"hardware_accelerator": "GPU_1xA10"},
}], environments=[{"environment_key": "gpu_env", "spec": {"client": "4"}}])
print(run.run_id)
```

## Gotchas that bite bio models (learned the hard way in GWB)
- **Don't pin `torch`.** The AI Runtime ships a matched torch/CUDA/cuDNN; pinning (e.g. `torch==2.3.1`)
  can reinstall an incompatible build. Install everything else; leave torch alone.
- **HuggingFace downloads:** the runtime sets `HF_HUB_ENABLE_HF_TRANSFER=1`, so `from_pretrained(...)`
  fails unless `hf_transfer` is installed — either `%pip install hf_transfer==0.1.9`, **or** set
  `os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"` **before** importing/using `huggingface_hub`.
- **Serving endpoints can't reach the HF LFS CDN** (jobs can). For a *serving* model, pre-stage weights
  to a UC Volume at register time and load from the local snapshot — don't `from_pretrained` a hub id
  at serving time.
- **Scratch disk:** use `/tmp` (or a UC Volume), **not** `/local_disk0` — it doesn't exist on serverless.
  `%sh` and Spark-JAR attach aren't available either; tasks needing those stay on a classic cluster
  (GWB keeps the genomics `%sh`/Glow setup jobs and the py3.10-locked SCimilarity tasks classic by design).
- **Python version:** `client '4'` (py3.12) is the default; drop to `client '2'` (py3.11) only when a
  dep lacks py3.12 wheels (e.g. GenMol's `pandas==2.1.0`). Whatever you register on becomes the serving
  env's Python.

## Serving workload sizes
When GWB deploys a model behind an endpoint it picks a serverless-GPU workload size (`GPU_SMALL`,
`GPU_MEDIUM`); you consume those via the endpoint (see **gwb-serving-endpoints**) and never manage the GPU.

## Scale past one GPU
A single A10 is simplest and most reliably provisioned. For a large batch (millions of sequences/cells),
fan out across many A10s with **Ray** — see **gwb-ray-batch-inference**.

---
name: gwb-ray-batch-inference
description: Scale a life-sciences batch-inference or embedding workflow across many GPUs in a Databricks notebook using Ray on serverless GPU (the AI Runtime). Use when a Genesis Workbench user must embed or score a large dataset — millions of protein sequences, single cells, or molecules — faster than one GPU allows, or build a Vector Search reference. Covers serverless_gpu.ray.ray_launch, ray.data map_batches with a stateful GPU actor, the UC Volume bridge, the stage→embed split, and CTAS to a managed Delta table. Mirrors how Genesis Workbench builds its own VS references (teddy, sequence_search). Triggers on "batch embed", "score millions", "run inference at scale", "multi-GPU", "Ray", "build a vector index over my data".
---

# Ray batch inference on serverless GPU (the GWB VS-reference pattern)

This is exactly how Genesis Workbench builds its own Vector Search references (teddy cells, sequence_search
proteins): fan a GPU forward pass across N A10 serverless-GPU workers with Ray, then promote the result into
a managed Delta table. Reach for it when one A10 (see **gwb-serverless-gpu**) is too slow for the dataset.

## The shape
```
Spark (orchestrator)                  Ray cluster (N × A10, serverless GPU)         Spark
read source → write parquet   ──►   read_parquet → map_batches(GPUActor) → write_parquet   ──►  CTAS → managed Delta
to a UC Volume (input stage)        (one model load per GPU, batched forward)       (output stage) + VS index
```

**Why the UC Volume bridge?** UC vends table storage credentials only to Spark on Databricks — the Ray
workers (non-Spark clients) can't read a managed table or write one directly. UC **Volumes** are
FUSE-accessible from any client, so Volumes are the hand-off at both ends: Spark stages the input columns
to a Volume as parquet; Ray reads/writes parquet on the Volume; Spark CTAS-promotes the output Volume
parquet into a managed Delta table.

## Minimal skeleton (adapt the actor to your model)
```python
%pip install -q transformers==4.41.2 pyarrow==15.0.2 hf_transfer==0.1.9 "ray[data]"
dbutils.library.restartPython()
```
```python
CATALOG, SCHEMA, VOL = "main", "genesis_workbench", "my_vol"
SRC_TABLE   = f"{CATALOG}.{SCHEMA}.my_inputs"            # has an id column + a payload column
OUT_TABLE   = f"{CATALOG}.{SCHEMA}.my_embeddings"
IN_STAGE    = f"/Volumes/{CATALOG}/{SCHEMA}/{VOL}/ray_input_stage"
OUT_STAGE   = f"/Volumes/{CATALOG}/{SCHEMA}/{VOL}/ray_output_stage"
NUM_WORKERS = 4          # serverless-GPU A10 Ray workers (scales ~linearly)
BATCH       = 32         # per-forward GPU batch

# Idempotency: skip if already built (re-runs are cheap no-ops)
if spark.catalog.tableExists(OUT_TABLE) and spark.table(OUT_TABLE).count() > 100:
    dbutils.notebook.exit("already built")

# 1) Spark stages just the columns Ray needs → Volume parquet (Spark has UC creds)
spark.table(SRC_TABLE).select("id", "sequence").write.mode("overwrite").parquet(IN_STAGE)
```
```python
# 2) Ray fans the forward across N A10s
from serverless_gpu.ray import ray_launch

@ray_launch(gpus=NUM_WORKERS, gpu_type="a10", remote=True)
def embed_with_ray():
    import numpy as np, ray, torch
    from transformers import AutoTokenizer, AutoModel

    class GPUActor:                       # ONE model load per GPU (stateful actor)
        def __init__(self):
            self.tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            self.model = AutoModel.from_pretrained(
                "facebook/esm2_t33_650M_UR50D", torch_dtype=torch.float16).cuda().eval()
        def __call__(self, batch):
            embs = []
            for s in range(0, len(batch["sequence"]), BATCH):
                seqs = list(batch["sequence"][s:s+BATCH])
                t = self.tok(seqs, return_tensors="pt", truncation=True, max_length=1024,
                             padding=True).to("cuda")
                with torch.no_grad():
                    out = self.model(**t)
                m = t["attention_mask"].unsqueeze(-1).float()          # mask-weighted mean pool
                e = (out.last_hidden_state*m).sum(1) / m.sum(1).clamp(min=1)
                embs.append(e.cpu().float().numpy())
            return {"id": np.asarray(batch["id"]),
                    "embedding": np.vstack(embs).tolist()}

    ds = ray.data.read_parquet(IN_STAGE)                      # Ray recurses dirs on its own
    (ds.map_batches(GPUActor, batch_size=BATCH, num_gpus=1,
                    concurrency=NUM_WORKERS, batch_format="numpy")   # one actor per GPU
       .write_parquet(OUT_STAGE))                             # each worker writes its own blocks

dbutils.fs.rm(OUT_STAGE, recurse=True)
embed_with_ray.distributed()                                 # blocks until the Ray run finishes
```
```python
# 3) Spark promotes the Volume parquet → managed Delta, then clean the stages
spark.sql(f"CREATE OR REPLACE TABLE {OUT_TABLE} USING DELTA AS SELECT * FROM parquet.`{OUT_STAGE}`")
dbutils.fs.rm(IN_STAGE, recurse=True); dbutils.fs.rm(OUT_STAGE, recurse=True)
spark.sql(f"ALTER TABLE {OUT_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")  # for a VS index
```
Then build a Vector Search index over `OUT_TABLE` — see **gwb-vector-search**.

## Scaling / robustness notes from GWB's own builds
- **`concurrency=NUM_WORKERS`, `num_gpus=1`** → exactly one actor (one model load) per A10. Set
  `@ray_launch(gpus=NUM_WORKERS)` to the same N. 4 is a good default; raise it to go faster.
- **Split STAGE from EMBED for a *remote* source** (e.g. a TileDB/S3/HF fetch per batch): doing the
  network fetch *inside* the GPU actor leaves the A10 idle on I/O. GWB's teddy job stages the fetched
  model inputs to a Volume in one notebook (03a, network-bound, resumable) and runs a pure GPU forward
  from the local parquet in another (03b, GPU-bound). If your payload is already in a Delta column
  (like the skeleton above), one notebook is fine.
- **Resumable staging:** write each run to a unique sub-dir (`.../stage/chunk_<ts>/`) and skip ids already
  staged, so an interrupted run never loses progress.
- **recursiveFileLookup:** if you write nested `chunk_<ts>/` sub-dirs, a later **Spark** read needs
  `spark.read.option("recursiveFileLookup","true").parquet(dir)` — Spark's default reader won't descend
  into non-`key=value` sub-dirs and errors `UNABLE_TO_INFER_SCHEMA`. Ray's `read_parquet` recurses already.
- **Transient `INTERNAL_ERROR` ("compute failed to start within 900s")** on a GPU task is a provisioning
  flake — just retry.
- Don't pin torch; handle `hf_transfer`; use `/tmp` not `/local_disk0` (see **gwb-serverless-gpu**).

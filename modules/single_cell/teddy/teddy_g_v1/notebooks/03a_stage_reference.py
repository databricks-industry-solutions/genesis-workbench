# Databricks notebook source
# MAGIC %md
# MAGIC ### TEDDY reference stage — fetch Census X, precompute TEDDY tokens → Volume (Phase 1 of 2)
# MAGIC
# MAGIC **Why this notebook exists.** The old single-notebook reembed had the Ray GPU workers
# MAGIC call `cellxgene_census.get_anndata(...)` **per batch** to pull the X matrix from the remote
# MAGIC Census (TileDB-SOMA → S3) *while holding an A10*. The network fetch dominated (~6k cells/min),
# MAGIC so the A10s sat idle on I/O and a full 2M-cell run took ~5.5h. This splits the job:
# MAGIC
# MAGIC   - **03a (this notebook) — STAGE:** open Census, build the stratified obs sample, and for
# MAGIC     each cell compute the **TEDDY model input** (top-k gene **token IDs** by expression) and
# MAGIC     write them + metadata to a Volume as parquet. This is the network-bound part, done ONCE.
# MAGIC   - **03b — EMBED:** Ray reads the LOCAL token parquet (no Census) and runs a pure GPU forward
# MAGIC     → embeddings. Now GPU-bound → scales ~linearly with `num_gpu_workers`, and re-embeds never
# MAGIC     re-hit Census.
# MAGIC
# MAGIC **Representation (option b).** We stage the precomputed top-k gene token IDs (the exact output of
# MAGIC the former `_build_batch` top-k + `token_array` mapping) rather than raw sparse X. Smaller, and
# MAGIC 03b becomes a pure forward. The top-k uses `torch.topk` here so embeddings are byte-identical to
# MAGIC the former inline path. (`gene_values` = linspace rank and the CLS/attn are deterministic from k,
# MAGIC so they are NOT stored — 03b reconstructs them.)
# MAGIC
# MAGIC **Resumable.** On re-run we read the token IDs already staged and only fetch the remaining
# MAGIC `soma_joinid`s (appending a new parquet sub-dir), so an interrupted stage never loses progress —
# MAGIC the lesson from cancelling a 70%-done run.

# COMMAND ----------

# MAGIC # cellxgene-census (tiledbsoma) ships extensions built against numpy<2 → pin numpy==1.26.4 for
# MAGIC # the Census read. torch is only used here for the (CPU/GPU) top-k, identical to 03b's math.
# MAGIC %pip install -q -r ../requirements.txt cellxgene-census==1.17.0 pyarrow==15.0.2 numpy==1.26.4 "ray[data]"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("cache_dir", "teddy", "Cache dir")
dbutils.widgets.text("teddy_model_size", "400M", "TEDDY-G variant")
dbutils.widgets.text("target_n_cells", "2000000", "Target reference cell count")
dbutils.widgets.text("per_stratum_cap", "30000", "Max cells per (tissue, disease) stratum")
dbutils.widgets.text("census_version", "2024-07-01", "CELLxGENE Census version (LTS tag or 'latest')")
dbutils.widgets.text("num_gpu_workers", "4", "Number of single-A10 serverless-GPU Ray workers (stage fetch parallelism)")
dbutils.widgets.text("obs_limit", "0", "Cap Census obs scan to first N soma_joinids (0 = full scan; >0 for fast tests)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
cache_dir = dbutils.widgets.get("cache_dir")
model_size = dbutils.widgets.get("teddy_model_size")
target_n_cells = int(dbutils.widgets.get("target_n_cells"))
per_stratum_cap = int(dbutils.widgets.get("per_stratum_cap"))
census_version = dbutils.widgets.get("census_version")
num_gpu_workers = int(dbutils.widgets.get("num_gpu_workers"))
obs_limit = int(dbutils.widgets.get("obs_limit"))

cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"

# Module-level constants (ALL_CAPS) — @ray_launch serializes referenced globals to the Ray workers.
SNAPSHOT_DIR = f"{cache_full_path}/snapshots/main"
MODEL_DIR = f"{SNAPSHOT_DIR}/teddy/models/teddy_g/{model_size}"
CENSUS_VERSION = census_version
MODEL_SIZE = model_size
NUM_GPU_WORKERS = num_gpu_workers
CENSUS_FETCH_BATCH = 10_000  # contiguous soma_joinids per get_anndata → sequential TileDB-SOMA reads

# Volume stages. OBS_SAMPLE holds the (resumable) remaining ids to fetch; TOKENS_STAGE_DIR is the
# durable precomputed-token output 03b reads. GENE_VOCAB maps the Census gene axis order → feature_id.
OBS_SAMPLE_DIR = f"{cache_full_path}/reembed_obs_sample"
TOKENS_STAGE_DIR = f"{cache_full_path}/reembed_tokens_stage"
GENE_VOCAB_PATH = f"{cache_full_path}/reembed_census_gene_vocab.json"

EXPECTED_DIM = {"70M": 512, "160M": 768, "400M": 1024}.get(model_size, 1024)

print(f"Cache: {cache_full_path}")
print(f"Model: TEDDY-G {model_size} at {MODEL_DIR}")
print(f"Target cells: {target_n_cells:,} (cap {per_stratum_cap:,}/stratum), {num_gpu_workers} A10 fetch workers")
print(f"Tokens stage → {TOKENS_STAGE_DIR}")

# COMMAND ----------

# DBTITLE 1,Orchestrator: open Census and build the stratified obs sample
import cellxgene_census
import numpy as np
import pandas as pd
import json, os

print(f"Opening CELLxGENE Census {census_version}…")
census = cellxgene_census.open_soma(census_version=census_version)
obs = census["census_data"]["homo_sapiens"].obs

# Healthy + disease primary cells (filtering disease!='normal' biased KNN toward disease cells).
_read_kwargs = dict(
    column_names=["soma_joinid", "cell_type", "disease", "tissue_general",
                  "tissue", "dataset_id", "assay", "is_primary_data"],
    value_filter="is_primary_data == True",
)
if obs_limit > 0:
    _read_kwargs["coords"] = (slice(0, obs_limit - 1),)
    print(f"obs_limit={obs_limit:,} — capping Census obs scan (test mode)")
obs_df = obs.read(**_read_kwargs).concat().to_pandas()
print(f"Census primary cells: {len(obs_df):,}")

_tissue = obs_df["tissue_general"].astype("object").where(obs_df["tissue_general"].notna(), "unknown")
_disease = obs_df["disease"].astype("object").where(obs_df["disease"].notna(), "unknown")
obs_df["__stratum"] = _tissue.astype(str) + " | " + _disease.astype(str)

rng = np.random.default_rng(seed=42)  # deterministic sample (same cells every run)
sampled_parts = []
for stratum, idx in obs_df.groupby("__stratum").groups.items():
    idx = np.array(idx)
    sampled_parts.append(idx if len(idx) <= per_stratum_cap
                         else rng.choice(idx, size=per_stratum_cap, replace=False))
sampled_idx = np.concatenate(sampled_parts)
rng.shuffle(sampled_idx)
if len(sampled_idx) > target_n_cells:
    sampled_idx = sampled_idx[:target_n_cells]
obs_sample = obs_df.iloc[sampled_idx].reset_index(drop=True)
print(f"Selected {len(obs_sample):,} cells across {obs_sample['__stratum'].nunique()} strata")
del census

# COMMAND ----------

# DBTITLE 1,Discover the Census gene vocab once → Volume JSON (gene-axis order get_anndata returns)
_c2 = cellxgene_census.open_soma(census_version=census_version)
_var = (_c2["census_data"]["homo_sapiens"].ms["RNA"]
        .var.read(column_names=["soma_joinid", "feature_id"]).concat().to_pandas())
census_gene_ids = _var.sort_values("soma_joinid")["feature_id"].astype(str).tolist()
del _c2
os.makedirs(cache_full_path, exist_ok=True)
with open(GENE_VOCAB_PATH, "w") as f:
    json.dump(census_gene_ids, f)
print(f"Census gene vocab: {len(census_gene_ids):,} → {GENE_VOCAB_PATH}")

# COMMAND ----------

# DBTITLE 1,Resume: subtract already-staged soma_joinids, stage only the remainder
# TOKENS_STAGE_DIR accumulates one parquet sub-dir per run (chunk=NNN), so prior progress is never
# overwritten. We read the ids already present and fetch only what's missing.
already_staged = set()
try:
    if any(True for _ in dbutils.fs.ls(TOKENS_STAGE_DIR)):
        # recursiveFileLookup: tokens live in nested chunk_<ts>/ sub-dirs (one per run), which
        # Spark's default partition discovery skips → UNABLE_TO_INFER_SCHEMA. Recurse to find them.
        staged_df = (spark.read.option("recursiveFileLookup", "true")
                     .parquet(TOKENS_STAGE_DIR).select("soma_joinid"))
        already_staged = {int(r["soma_joinid"]) for r in staged_df.distinct().collect()}
except Exception:
    already_staged = set()
print(f"Already staged: {len(already_staged):,} cells")

remaining = obs_sample[~obs_sample["soma_joinid"].astype("int64").isin(already_staged)]
if len(remaining) == 0:
    print(f"All {len(obs_sample):,} cells already tokenized in {TOKENS_STAGE_DIR} — stage complete.")
    dbutils.notebook.exit("stage_complete")

# Sort by soma_joinid so each Ray batch is a contiguous joinid range → sequential TileDB-SOMA S3
# reads (scattered point reads were the dominant cost). Write the remainder as the Ray input.
remaining_sorted = remaining.sort_values("soma_joinid").reset_index(drop=True)
stage_pdf = pd.DataFrame({
    "soma_joinid":    remaining_sorted["soma_joinid"].astype("int64").to_numpy(),
    "cell_type":      remaining_sorted["cell_type"].astype(str).fillna("").to_numpy(),
    "disease":        remaining_sorted["disease"].astype(str).fillna("").to_numpy(),
    "tissue":         remaining_sorted["tissue"].astype(str).fillna("").to_numpy(),
    "tissue_general": remaining_sorted["tissue_general"].astype(str).fillna("").to_numpy(),
    "dataset_id":     remaining_sorted["dataset_id"].astype(str).fillna("").to_numpy(),
})
dbutils.fs.rm(OBS_SAMPLE_DIR, recurse=True)
os.makedirs(OBS_SAMPLE_DIR, exist_ok=True)
stage_pdf.to_parquet(f"{OBS_SAMPLE_DIR}/obs_sample.parquet", index=False, row_group_size=CENSUS_FETCH_BATCH)
print(f"To fetch this run: {len(stage_pdf):,} cells → {OBS_SAMPLE_DIR}")

# Where this run's tokens land (unique sub-dir so appends never clobber prior runs).
import time as _time
RUN_TOKENS_DIR = f"{TOKENS_STAGE_DIR}/chunk_{int(_time.time())}"

# COMMAND ----------

# DBTITLE 1,Ray: N A10 workers fetch Census X in parallel → top-k TEDDY tokens → token parquet
from serverless_gpu.ray import ray_launch

@ray_launch(gpus=NUM_GPU_WORKERS, gpu_type="a10", remote=True)
def stage_tokens_with_ray():
    import inspect, json, sys
    import numpy as np
    import ray
    import torch
    import cellxgene_census

    print(f"Ray cluster resources: {ray.cluster_resources()}", flush=True)

    class TokenizeActor:
        """One per worker. Loads only the TEDDY tokenizer + gene->token map (NO model forward),
        opens its own Census handle, and for each batch fetches X and emits the top-k gene token
        IDs — exactly the former `_build_batch` top-k + token_array mapping."""

        def __init__(self):
            if SNAPSHOT_DIR not in sys.path:
                sys.path.insert(0, SNAPSHOT_DIR)
            from teddy.models.model_directory import get_architecture, model_dict
            from teddy.tokenizer.gene_tokenizer import GeneTokenizer

            arch = get_architecture(MODEL_DIR)
            config = model_dict[arch]["config_cls"].from_pretrained(MODEL_DIR)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = GeneTokenizer.from_pretrained(MODEL_DIR)
            unk_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
            gene_names = json.load(open(GENE_VOCAB_PATH))
            ids = tokenizer.convert_tokens_to_ids(list(gene_names))
            ids = [unk_id if i is None else i for i in ids]
            self.token_array = torch.tensor(ids, dtype=torch.long).to(self.device)
            # k = number of top genes per cell. Matches 03b: max_seq_len - 1 if add_cls else max_seq_len.
            self.add_cls = bool(getattr(config, "add_cls", False))
            self.max_seq_len = int(getattr(config, "max_position_embeddings", 2048))
            self.seq_tokens = self.max_seq_len - 1 if self.add_cls else self.max_seq_len
            self.census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)
            print(f"TokenizeActor on {self.device}, seq_tokens={self.seq_tokens}, "
                  f"vocab={len(gene_names):,}", flush=True)

        def __call__(self, batch):
            soma_ids = [int(x) for x in batch["soma_joinid"]]
            cols = ("cell_type", "disease", "tissue", "tissue_general", "dataset_id")
            meta = {int(batch["soma_joinid"][i]): {c: str(batch[c][i]) for c in cols}
                    for i in range(len(soma_ids))}

            adata = cellxgene_census.get_anndata(self.census, organism="Homo sapiens", obs_coords=soma_ids)
            n = adata.shape[0]
            if n == 0:
                return {"soma_joinid": np.array([], dtype=np.int64),
                        "gene_token_ids": [], **{c: np.array([], dtype=object) for c in cols}}

            X = adata.X
            X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            X_t = torch.tensor(X_dense, dtype=torch.float32, device=self.device)
            k = min(self.seq_tokens, X_t.shape[1])
            # IDENTICAL to the former _build_batch: top-k by expression, largest-first, then map
            # Census gene-axis index -> TEDDY token id. (rank values + CLS are reconstructed in 03b.)
            _vals, top_idx = torch.topk(X_t, k=k, largest=True, sorted=True)
            tok = self.token_array[top_idx].detach().cpu().numpy().astype(np.int64)  # (n, k)

            returned = adata.obs["soma_joinid"].astype("int64").to_numpy()
            def _n(v): return v if v else None
            return {
                "soma_joinid":    returned,
                "gene_token_ids": [row.tolist() for row in tok],
                "cell_type":      np.array([_n(meta[int(s)]["cell_type"]) for s in returned], dtype=object),
                "disease":        np.array([_n(meta[int(s)]["disease"]) for s in returned], dtype=object),
                "tissue":         np.array([_n(meta[int(s)]["tissue"]) for s in returned], dtype=object),
                "tissue_general": np.array([_n(meta[int(s)]["tissue_general"]) for s in returned], dtype=object),
                "dataset_id":     np.array([_n(meta[int(s)]["dataset_id"]) for s in returned], dtype=object),
            }

    ds = ray.data.read_parquet(OBS_SAMPLE_DIR)
    print(f"Dataset rows to tokenize: {ds.count()}", flush=True)
    result = ds.map_batches(
        TokenizeActor,
        batch_size=CENSUS_FETCH_BATCH,
        num_gpus=1,                   # one actor per A10 (GPU used only for the cheap top-k)
        concurrency=NUM_GPU_WORKERS,
        batch_format="numpy",
    )
    result.write_parquet(RUN_TOKENS_DIR)
    print(f"Ray tokenize complete → {RUN_TOKENS_DIR}", flush=True)

# COMMAND ----------

import time as _time
_t0 = _time.time()
stage_tokens_with_ray.distributed()
print(f"Stage (tokenize) complete in {(_time.time()-_t0)/60:.1f} min")

# COMMAND ----------

# DBTITLE 1,Verify staged token count + shape
# recursiveFileLookup: tokens live in nested chunk_<ts>/ sub-dirs; Spark won't recurse by default.
staged = spark.read.option("recursiveFileLookup", "true").parquet(TOKENS_STAGE_DIR)
n_tokens = staged.count()
from pyspark.sql import functions as F
dims = staged.select(F.min(F.size("gene_token_ids")).alias("min_k"),
                     F.max(F.size("gene_token_ids")).alias("max_k")).collect()[0].asDict()
print(f"Tokens staged: {n_tokens:,} cells (target {target_n_cells:,}); token-length {dims}")
dbutils.fs.rm(OBS_SAMPLE_DIR, recurse=True)  # transient per-run input; TOKENS_STAGE_DIR is durable
dbutils.notebook.exit(f"staged={n_tokens}")

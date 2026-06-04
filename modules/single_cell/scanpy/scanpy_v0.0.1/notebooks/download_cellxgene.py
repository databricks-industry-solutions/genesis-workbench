# Databricks notebook source
# MAGIC %md
# MAGIC ### Download CellxGene Reference Datasets
# MAGIC
# MAGIC Downloads a curated subset of single-cell h5ad files from the CZ CELLxGENE
# MAGIC Census into a Unity Catalog Volume. These serve as reference datasets for
# MAGIC protein studies and single-cell cross-referencing.
# MAGIC
# MAGIC **Strategy:** Download datasets with 6,000–8,000 cells (manageable size,
# MAGIC good diversity). Files are stored in `/Volumes/{catalog}/{schema}/raw_h5ad/`.

# COMMAND ----------

# MAGIC %pip install gget==0.28.6 cellxgene-census==1.17.0 numpy==1.26.4 pybiomart==0.2.0 scanpy==1.11.4
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("census_version", "2025-11-08", "Census Version")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
census_version = dbutils.widgets.get("census_version")

print(f"Catalog: {catalog}, Schema: {schema}, Census version: {census_version}")

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw_h5ad")
print(f"Volume ready: /Volumes/{catalog}/{schema}/raw_h5ad")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Fetch CellxGene metadata and identify datasets to download

# COMMAND ----------

import gget
import cellxgene_census
from pyspark.sql import functions as F
from pyspark.sql import types as T

gget.setup("cellxgene")

df = gget.cellxgene(
    meta_only=True,
    census_version=census_version,
    species="homo_sapiens",
    is_primary_data=True,
    suspension_type="cell",
    column_names=["dataset_id", "assay", "cell_type", "donor_id"],
)

sdf = spark.createDataFrame(df)
sdf_count = sdf.groupBy("dataset_id").count().sort("count", ascending=False)
sdf_count.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{schema}.cellxgene_cell_counts")

total_datasets = sdf_count.count()
print(f"Total CellxGene datasets found: {total_datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Download filtered subset (6k–8k cells per dataset)

# COMMAND ----------

sdf_count = spark.table(f"{catalog}.{schema}.cellxgene_cell_counts")
dataset_id_sdf = sdf_count.filter(
    (F.col("count") < 8_000) & (F.col("count") > 6_000)
)

n_datasets = dataset_id_sdf.count()
print(f"Datasets matching filter (6k-8k cells): {n_datasets}")

# COMMAND ----------

import gget
import cellxgene_census
gget.setup("cellxgene")

# Determine parallelism from cluster size
try:
    cores_per_worker = 4
    n_workers = int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers"))
    cores = max(cores_per_worker * n_workers, 4)
except Exception:
    cores = 12  # serverless fallback

if n_datasets > 0:
    dataset_id_sdf = dataset_id_sdf.coalesce(1).repartition(cores)

_catalog = catalog
_schema = schema
_census_version = census_version


@F.udf(returnType=T.BooleanType())
def download_czi_h5ad_to_volume(d_id):
    import cellxgene_census
    try:
        cellxgene_census.download_source_h5ad(
            dataset_id=d_id,
            to_path=f"/Volumes/{_catalog}/{_schema}/raw_h5ad/{d_id}.h5ad",
            census_version=_census_version,
            progress_bar=False,
        )
        return True
    except Exception:
        return False


dataset_id_sdf = dataset_id_sdf.withColumn(
    "download_successful",
    download_czi_h5ad_to_volume(F.col("dataset_id")),
)

dataset_id_sdf.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "True") \
    .saveAsTable(f"{catalog}.{schema}.cellxgene_datasets_downloaded")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verify downloads and catalog processed datasets

# COMMAND ----------

import h5py
import os

pdf = spark.table(f"{catalog}.{schema}.cellxgene_datasets_downloaded") \
    .orderBy("count", ascending=False).toPandas()
dataset_ids = list(pdf.dataset_id.values)

final_datasets = {}
for d_id in dataset_ids:
    file_path = f"/Volumes/{catalog}/{schema}/raw_h5ad/{d_id}.h5ad"
    if not os.path.exists(file_path):
        print(f"  Missing file for {d_id}")
        continue
    try:
        with h5py.File(file_path, "r") as f:
            if "raw" in f and "var" in f["raw"] and "X" in f["raw"]:
                gene_count = f["raw"]["var"]["feature_name"]["codes"].shape[0]
                entries = f["raw"]["X"]["indices"].shape[0]
                final_datasets[d_id] = {
                    "gene_count": gene_count,
                    "entries": entries,
                    "obs_in_raw": "obs" in f["raw"],
                }
            else:
                print(f"  Missing raw/var/X for {d_id}")
    except Exception as e:
        print(f"  Error reading {d_id}: {e}")

import pandas as pd

df_final = pd.DataFrame(final_datasets).T.reset_index().rename(columns={"index": "dataset_id"})
spark.createDataFrame(df_final).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "True") \
    .saveAsTable(f"{catalog}.{schema}.cellxgene_datasets_processed")

print(f"\nDownloaded: {len(dataset_ids)} datasets")
print(f"Processed successfully: {len(final_datasets)} datasets")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stage the curated end-to-end demo dataset (Ovarian HGSOC)
# MAGIC
# MAGIC The BRCA→PARP1 demo story needs a tumor scRNA-seq dataset with a clean
# MAGIC malignant-vs-normal contrast. We use the **MSK SPECTRUM HGSOC atlas**
# MAGIC (CELLxGENE dataset `44c93f2b-dd66-4d15-81ef-de9394c76290`, fallopian-tube
# MAGIC secretory epithelium — the cell of origin for high-grade serous ovarian
# MAGIC cancer), subsampled to ~15k cells so it processes quickly in the Single
# MAGIC Cell tab. We keep **all benign Ciliated cells** (the normal-epithelium
# MAGIC reference) and a proportional sample of the **malignant Cancer/Cycling
# MAGIC clusters**, set `X` to **raw integer counts**, and key genes by symbol
# MAGIC (`feature_name`). PARP1/BRCA1/BRCA2 are all present.
# MAGIC
# MAGIC Output: `/Volumes/{catalog}/{schema}/raw_h5ad/hgsoc_demo_15k.h5ad`
# MAGIC (process it via Single Cell → Run New Analysis with
# MAGIC `gene_name_column=feature_name`, `species=hsapiens`, `pct_counts_mt=20`,
# MAGIC `n_genes_by_counts=8000`). Idempotent: skips if the file already exists.

# COMMAND ----------

import os
import tempfile
import urllib.request

import numpy as np
import scipy.sparse as sp
import anndata as ad

DEMO_DATASET_ID = "44c93f2b-dd66-4d15-81ef-de9394c76290"  # MSK SPECTRUM HGSOC
DEMO_OUT = f"/Volumes/{catalog}/{schema}/raw_h5ad/hgsoc_demo_15k.h5ad"
DEMO_TARGET_CELLS = 15_000
DEMO_MALIGNANT_TARGET = 11_000
# Benign reference clusters kept in full; everything else is treated as malignant.
DEMO_NORMAL_CLUSTERS = ("Ciliated.cell.1", "Ciliated.cell.2")
DEMO_CLUSTER_COL = "cluster_label"
DEMO_SEED = 42

if os.path.exists(DEMO_OUT):
    print(f"Demo dataset already staged at {DEMO_OUT} — skipping.")
else:
    # 1) Resolve the current curated H5AD asset URL by dataset_id (version-independent).
    import json
    import urllib.request as _u

    versions_url = (
        f"https://api.cellxgene.cziscience.com/curation/v1/datasets/"
        f"{DEMO_DATASET_ID}/versions"
    )
    with _u.urlopen(versions_url, timeout=60) as resp:
        versions = json.loads(resp.read().decode())
    latest = versions[0]  # newest version first
    h5ad_url = next(a["url"] for a in latest["assets"] if a["filetype"] == "H5AD")
    print(f"Resolved HGSOC H5AD: {h5ad_url} ({latest.get('cell_count')} cells)")

    # 2) Download the full curated H5AD to a temp file (~1.9 GB).
    tmp_h5ad = os.path.join(tempfile.gettempdir(), f"{DEMO_DATASET_ID}.h5ad")
    if not os.path.exists(tmp_h5ad) or os.path.getsize(tmp_h5ad) < 1_000_000:
        print("Downloading curated H5AD …")
        urllib.request.urlretrieve(h5ad_url, tmp_h5ad)
    print(f"Downloaded {os.path.getsize(tmp_h5ad) / 1e6:.0f} MB -> {tmp_h5ad}")

    # 3) Stratified subsample (deterministic). Keep all benign reference cells,
    #    sample malignant clusters proportionally with a small floor per cluster.
    rng = np.random.RandomState(DEMO_SEED)
    a = ad.read_h5ad(tmp_h5ad, backed="r")
    if DEMO_CLUSTER_COL in a.obs.columns:
        cl = a.obs[DEMO_CLUSTER_COL].astype(str)
        keep_idx = []
        # all benign reference cells
        for c in DEMO_NORMAL_CLUSTERS:
            keep_idx.append(np.where(cl.values == c)[0])
        malig_counts = cl[~cl.isin(DEMO_NORMAL_CLUSTERS)].value_counts()
        total_malig = int(malig_counts.sum())
        for c, n in malig_counts.items():
            share = int(round(DEMO_MALIGNANT_TARGET * n / total_malig))
            take = max(min(int(n), share), min(int(n), 250))
            idx = np.where(cl.values == c)[0]
            keep_idx.append(np.sort(rng.choice(idx, size=min(take, len(idx)), replace=False)))
        keep = np.sort(np.concatenate(keep_idx))
    else:
        # Fallback: simple random subsample if the expected annotation is absent.
        print(f"WARNING: '{DEMO_CLUSTER_COL}' not in obs — random subsample.")
        keep = np.sort(rng.choice(a.n_obs, size=min(DEMO_TARGET_CELLS, a.n_obs), replace=False))
    print(f"Selected {len(keep)} cells of {a.n_obs}.")

    # 4) Build output: X = raw integer counts; var keyed by gene symbol (deduped).
    raw = a.raw if a.raw is not None else a
    raw_X = raw[keep].X
    raw_X = raw_X.tocsr() if sp.issparse(raw_X) else sp.csr_matrix(raw_X)
    var = raw.var.copy()
    fn = var["feature_name"].astype(str) if "feature_name" in var.columns else var.index.astype(str)
    keep_genes = ~fn.duplicated(keep="first")
    raw_X = raw_X[:, keep_genes.values]
    var = var.loc[keep_genes.values].copy()
    var.index = fn[keep_genes.values].values
    var["feature_name"] = var.index

    keep_obs = [
        c for c in (
            DEMO_CLUSTER_COL, "cell_type", "patient_id", "sample",
            "author_tumor_supersite", "disease", "Phase", "seurat_clusters",
            "nCount_RNA", "nFeature_RNA", "percent.mt",
        ) if c in a.obs.columns
    ]
    new_obs = a.obs[keep_obs].iloc[keep].reset_index(drop=True)
    out = ad.AnnData(X=raw_X, obs=new_obs, var=var)
    out.obs_names = [f"cell_{i}" for i in range(out.n_obs)]

    for g in ("PARP1", "BRCA1", "BRCA2"):
        print(f"  {g} present: {g in out.var_names}")
    out.write(DEMO_OUT)
    print(f"Staged demo dataset: {out.shape[0]} cells × {out.shape[1]} genes -> {DEMO_OUT}")

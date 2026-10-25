# Databricks notebook source
# MAGIC %md
# MAGIC ### Stage the Single-Cell demo dataset (Ovarian HGSOC)
# MAGIC
# MAGIC Stages the curated end-to-end demo h5ad the Single Cell tab prefills, into
# MAGIC `/Volumes/{catalog}/{schema}/raw_h5ad/hgsoc_demo_15k.h5ad`.
# MAGIC
# MAGIC **Serverless note:** the previous version first scanned the *entire* CELLxGENE
# MAGIC census metadata into `spark.createDataFrame(pandas_df)` to pick a 6k–8k-cell
# MAGIC subset. On serverless (Spark Connect) that ships the ~6 GB frame to the service
# MAGIC as a cached local relation, which exceeds the hard 3 GiB limit
# MAGIC (`LOCAL_RELATION_SIZE_LIMIT_EXCEEDED`) — so the job died before staging anything.
# MAGIC It also picked datasets non-deterministically. We now stage the curated HGSOC
# MAGIC demo directly (its id is fixed, it downloads via the curation API + anndata, no
# MAGIC Spark), which is what the UI default points at and works out of the box.

# COMMAND ----------

# MAGIC # anndata pinned to 0.10.x: 0.11+ imports `Format` from a newer typing_extensions
# MAGIC # than the serverless base ships, which breaks `import anndata`. This cell only
# MAGIC # needs numpy + scipy (via anndata) + anndata, so scanpy/gget aren't installed.
# MAGIC %pip install numpy==1.26.4 anndata==0.10.9
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
    # Read ONLY the selected cells' raw counts straight from the H5AD with h5py. anndata
    # 0.10.9's backed sparse reader (BaseCompressedSparseDataset.__getitem__) materializes
    # the FULL matrix before subsetting — it OOMs on a large atlas ("Execution ran out of
    # memory") — AND its index format is incompatible with the runtime's newer scipy (raises
    # "ValueError: setting an array element with a sequence" during index validation). So
    # there is no memory-safe or version-safe way to subset through anndata here. Slicing the
    # CSR rows by hand reads just the ~15k chosen rows and sidesteps both failures. (anndata
    # is still used for the small in-memory obs/var metadata, which don't hit the sparse reader.)
    import h5py

    parts_data, parts_idx, new_indptr = [], [], [0]
    with h5py.File(tmp_h5ad, "r") as _f:
        xgrp = _f["raw/X"] if "raw/X" in _f else _f["X"]
        enc = str(xgrp.attrs.get("encoding-type", "csr_matrix"))
        assert "csr" in enc, f"expected a CSR raw matrix, got encoding-type={enc!r}"
        shape = tuple(int(s) for s in (xgrp.attrs["shape"] if "shape" in xgrp.attrs else xgrp.attrs["h5sparse_shape"]))
        indptr = xgrp["indptr"][:]
        data_ds, indices_ds = xgrp["data"], xgrp["indices"]
        for r in keep:
            s, e = int(indptr[r]), int(indptr[r + 1])
            parts_data.append(data_ds[s:e])
            parts_idx.append(indices_ds[s:e])
            new_indptr.append(new_indptr[-1] + (e - s))
    raw_X = sp.csr_matrix(
        (
            np.concatenate(parts_data).astype(np.float32) if parts_data else np.zeros(0, np.float32),
            np.concatenate(parts_idx) if parts_idx else np.zeros(0, np.int64),
            np.asarray(new_indptr),
        ),
        shape=(len(keep), shape[1]),
    )
    var = (a.raw.var if a.raw is not None else a.var).copy()
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
    # anndata/h5py write HDF5 with many random-access/seek writes, which the /Volumes FUSE
    # mount rejects (OSError errno 5, "Driver write request failed" / "Input/output error").
    # Write to fast LOCAL disk first, then copy the finished file to the Volume — a single
    # sequential copy, which FUSE handles fine.
    import shutil

    local_out = os.path.join(tempfile.gettempdir(), "hgsoc_demo_15k.h5ad")
    out.write(local_out)
    shutil.copyfile(local_out, DEMO_OUT)
    print(f"Staged demo dataset: {out.shape[0]} cells × {out.shape[1]} genes -> {DEMO_OUT}")

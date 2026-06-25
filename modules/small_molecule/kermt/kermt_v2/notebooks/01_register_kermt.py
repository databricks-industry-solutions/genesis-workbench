# Databricks notebook source
# MAGIC %md
# MAGIC # KERMT — Stage / Register
# MAGIC One-time-per-deploy staging for the KERMT (Kinetic GROVER Multi-Task) ADMET model:
# MAGIC 1. Download the pretrained **GROVERbase** checkpoint into the `kermt` UC volume (skip-if-exists).
# MAGIC 2. Create the `kermt_weights` Delta table (records each fine-tuned model — mirrors `bionemo_weights`).
# MAGIC 3. Bundle a **TDC ClinTox** sample (train/val/test CSVs) for the default fine-tune.
# MAGIC
# MAGIC Runs on a classic single-node CPU cluster (SINGLE_USER + UC volumes) — no GPU needed for staging.

# COMMAND ----------

dbutils.widgets.text("catalog", "srijit_nair_ci_demo_catalog", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "kermt", "KERMT UC volume name")
dbutils.widgets.text("grover_base_url", "", "GROVERbase direct-download URL")
dbutils.widgets.text("tdc_group", "Tox", "TDC single_pred group (Tox/ADME)")
dbutils.widgets.text("tdc_dataset", "ClinTox", "TDC dataset name")
dbutils.widgets.text("target_name", "toxicity", "Target column name written to the sample CSVs")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
cache_dir = dbutils.widgets.get("cache_dir")
grover_base_url = dbutils.widgets.get("grover_base_url")
tdc_group = dbutils.widgets.get("tdc_group")
tdc_dataset = dbutils.widgets.get("tdc_dataset")
target_name = dbutils.widgets.get("target_name")

# COMMAND ----------

# MAGIC %pip install --no-deps PyTDC==1.1.15
# MAGIC %pip install fuzzywuzzy==0.18.0 tqdm==4.67.1 requests==2.32.3 pandas==1.5.3 huggingface_hub==0.25.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
cache_dir = dbutils.widgets.get("cache_dir")
grover_base_url = dbutils.widgets.get("grover_base_url")
tdc_group = dbutils.widgets.get("tdc_group")
tdc_dataset = dbutils.widgets.get("tdc_dataset")
target_name = dbutils.widgets.get("target_name")

vol_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"
pretrained_dir = f"{vol_root}/pretrained"
ft_data_dir = f"{vol_root}/ft_data"

# Volume is created by the bundle (resources/volumes.yml); ensure it + subdirs exist.
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{cache_dir}")
os.makedirs(pretrained_dir, exist_ok=True)
os.makedirs(ft_data_dir, exist_ok=True)
print("volume root:", vol_root)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Download GROVERbase (skip-if-exists)
# MAGIC OneDrive-hosted; default URL resolves the share link via the OneDrive shares API.

# COMMAND ----------

import requests

grover_path = f"{pretrained_dir}/kermt_contrastive_v2.0.pt"
if os.path.exists(grover_path) and os.path.getsize(grover_path) > 1_000_000:
    print(f"GROVERbase already present ({os.path.getsize(grover_path):,} bytes) — skipping download.")
else:
    assert grover_base_url, "grover_base_url is empty and no checkpoint is present"
    print(f"Downloading GROVERbase from {grover_base_url} ...")
    with requests.get(grover_base_url, stream=True, allow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        tmp = "/local_disk0/kermt_contrastive_v2.0.pt"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                if chunk:
                    f.write(chunk)
    size = os.path.getsize(tmp)
    assert size > 1_000_000, f"downloaded file too small ({size} bytes) — bad URL?"
    # write into the volume (FUSE POSIX on this SINGLE_USER cluster)
    import shutil
    shutil.copy(tmp, grover_path)
    print(f"GROVERbase staged: {grover_path} ({os.path.getsize(grover_path):,} bytes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Create the `kermt_weights` table
# MAGIC Records each fine-tuned KERMT model. Mirrors the structure of `bionemo_weights`.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.kermt_weights (
    ft_id BIGINT,
    ft_label STRING,
    model_type STRING,
    dataset_type STRING,
    task_names STRING,
    experiment_name STRING,
    run_id STRING,
    weights_volume_location STRING,
    created_by STRING,
    created_datetime TIMESTAMP,
    is_active BOOLEAN,
    deactivated_timestamp TIMESTAMP
)
""")
print(f"{catalog}.{schema}.kermt_weights ready")
display(spark.sql(f"DESCRIBE {catalog}.{schema}.kermt_weights"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Bundle the TDC sample (default: ClinTox)
# MAGIC Writes `train.csv` / `val.csv` / `test.csv` with columns `smiles`, `<target_name>`.
# MAGIC KERMT finetune expects a `smiles` column plus one column per prediction task.

# COMMAND ----------

import pandas as pd

def _exists(p):
    return os.path.exists(p) and os.path.getsize(p) > 0

train_csv = f"{ft_data_dir}/{tdc_dataset.lower()}_train.csv"
val_csv = f"{ft_data_dir}/{tdc_dataset.lower()}_val.csv"
test_csv = f"{ft_data_dir}/{tdc_dataset.lower()}_test.csv"

if all(_exists(p) for p in (train_csv, val_csv, test_csv)):
    print("TDC sample already bundled — skipping.")
else:
    if tdc_group == "Tox":
        from tdc.single_pred import Tox as _TDC
    else:
        from tdc.single_pred import ADME as _TDC
    data = _TDC(name=tdc_dataset, path="/local_disk0/tdc_data")
    split = data.get_split()  # dict: train/valid/test, columns Drug (SMILES), Y (label)

    def _to_csv(df, path):
        out = df[["Drug", "Y"]].rename(columns={"Drug": "smiles", "Y": target_name})
        out.to_csv(path, index=False)
        print(f"wrote {path}: {len(out)} rows")

    _to_csv(split["train"], train_csv)
    _to_csv(split["valid"], val_csv)
    _to_csv(split["test"], test_csv)

# COMMAND ----------

print("KERMT staging complete.")
print(f"  weights : {grover_path}")
print(f"  table   : {catalog}.{schema}.kermt_weights")
print(f"  ft_data : {ft_data_dir}/{tdc_dataset.lower()}_{{train,val,test}}.csv")

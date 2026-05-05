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

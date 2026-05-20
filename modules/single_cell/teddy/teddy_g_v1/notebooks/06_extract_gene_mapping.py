# Databricks notebook source
# MAGIC %md
# MAGIC ### Extract HGNC → ENSG gene mapping for TEDDY query path
# MAGIC
# MAGIC TEDDY's vocab is ENSG IDs (e.g., `ENSG00000000003`). User processing runs
# MAGIC typically store gene names as HGNC symbols (e.g., `TSPAN6`). Without a
# MAGIC translation, every gene becomes `<unk>` at query time → embeddings are
# MAGIC pure noise → KNN annotation returns garbage.
# MAGIC
# MAGIC This notebook is a one-shot extract from CELLxGENE Census's `var` table,
# MAGIC which carries both columns (`feature_name` = HGNC, `feature_id` = ENSG).
# MAGIC The result is written to a Volume as JSON so `teddy_tools.py` in the GWB
# MAGIC app can load it at startup.

# COMMAND ----------

# MAGIC %pip install -q "numpy<2" cellxgene-census==1.17.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "srijit_nair", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("cache_dir", "teddy", "Cache dir")
dbutils.widgets.text("census_version", "2024-07-01", "CELLxGENE Census version")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
cache_dir = dbutils.widgets.get("cache_dir")
census_version = dbutils.widgets.get("census_version")

out_path = f"/Volumes/{catalog}/{schema}/{cache_dir}/gene_mapping.json"
print(f"Will write mapping to: {out_path}")

# COMMAND ----------

import json
import cellxgene_census

print(f"Opening Census {census_version}…")
census = cellxgene_census.open_soma(census_version=census_version)

var_df = (
    census["census_data"]["homo_sapiens"].ms["RNA"].var
    .read(column_names=["feature_id", "feature_name"])
    .concat().to_pandas()
)
print(f"Census var rows: {len(var_df):,}")
print(var_df.head())

# COMMAND ----------

# Build mapping: feature_name (HGNC) -> feature_id (ENSG).
# Some HGNC names map to multiple ENSGs (paralogs, pseudogenes). Take the first
# ENSG for each name — same convention SCimilarity uses.
mapping = (
    var_df.dropna(subset=["feature_name", "feature_id"])
    .drop_duplicates(subset=["feature_name"], keep="first")
    .set_index("feature_name")["feature_id"]
    .to_dict()
)
print(f"Mapping size: {len(mapping):,} HGNC → ENSG entries")
# Spot check a few well-known genes
for sym in ["TSPAN6", "CD4", "CD8A", "EPCAM", "GAPDH", "ACTB"]:
    print(f"  {sym} -> {mapping.get(sym, '(missing)')}")

# COMMAND ----------

# Write to Volume as JSON
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(mapping, f)
print(f"Wrote {out_path} ({os.path.getsize(out_path):,} bytes)")

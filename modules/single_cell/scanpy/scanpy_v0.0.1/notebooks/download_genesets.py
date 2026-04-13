# Databricks notebook source
# MAGIC %md
# MAGIC # Download Gene Set (GMT) Files for Pathway Enrichment
# MAGIC
# MAGIC Downloads GMT gene set files from Enrichr's static library into the
# MAGIC `scanpy_reference` volume so that pathway enrichment can run locally
# MAGIC without external API calls.
# MAGIC
# MAGIC **Databases downloaded:**
# MAGIC - GO_Biological_Process_2023
# MAGIC - KEGG_2021_Human
# MAGIC - Reactome_2022
# MAGIC - GO_Molecular_Function_2023
# MAGIC - GO_Cellular_Component_2023

# COMMAND ----------

import os
import subprocess

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catalog Name")
dbutils.widgets.text("schema", "", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

if not CATALOG or not SCHEMA:
    raise ValueError("catalog and schema parameters must be provided")

print(f"Using catalog: {CATALOG}, schema: {SCHEMA}")

# COMMAND ----------

GENESET_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/scanpy_reference/genesets"
os.makedirs(GENESET_DIR, exist_ok=True)
print(f"Gene set directory: {GENESET_DIR}")

# COMMAND ----------

# Enrichr static gene set library URLs
# Format: tab-separated GMT (term \t description \t gene1 \t gene2 \t ...)
GMT_LIBRARIES = {
    "GO_Biological_Process_2023": "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2023",
    "KEGG_2021_Human": "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2021_Human",
    "Reactome_2022": "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=Reactome_2022",
    "GO_Molecular_Function_2023": "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023": "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Cellular_Component_2023",
}

results = []

for lib_name, url in GMT_LIBRARIES.items():
    output_path = f"{GENESET_DIR}/{lib_name}.gmt"

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        size = os.path.getsize(output_path)
        print(f"✓ {lib_name}: Already exists ({size:,} bytes)")
        results.append((lib_name, "exists", output_path))
    else:
        try:
            print(f"⏳ {lib_name}: Downloading...")
            subprocess.run(
                ["wget", "-q", "-O", output_path, url],
                check=True,
            )
            size = os.path.getsize(output_path)
            print(f"✓ {lib_name}: Downloaded ({size:,} bytes)")
            results.append((lib_name, "downloaded", output_path))
        except subprocess.CalledProcessError as e:
            print(f"✗ {lib_name}: Download failed - {e}")
            results.append((lib_name, "failed", str(e)))

# COMMAND ----------

# Summary
print("\n" + "=" * 80)
print("GENE SET DOWNLOAD SUMMARY")
print("=" * 80)

for lib_name, status, info in results:
    if status == "exists":
        print(f"✓ {lib_name}: Already available")
    elif status == "downloaded":
        print(f"✓ {lib_name}: Downloaded successfully")
    elif status == "failed":
        print(f"✗ {lib_name}: Failed - {info}")

print("=" * 80)

all_success = all(status in ["exists", "downloaded"] for _, status, _ in results)
if all_success:
    print(f"\n✓ All {len(results)} gene set files are ready in {GENESET_DIR}")
else:
    print("\n✗ Some gene set files failed to download. Check the errors above.")

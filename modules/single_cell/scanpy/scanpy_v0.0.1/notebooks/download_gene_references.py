# Databricks notebook source
# MAGIC %md
# MAGIC # Download Ensembl Gene Reference Tables
# MAGIC 
# MAGIC This notebook downloads gene reference tables from Ensembl for multiple species.
# MAGIC These tables map Ensembl gene IDs to external gene names.
# MAGIC 
# MAGIC **Species supported:**
# MAGIC - Human (Homo sapiens)
# MAGIC - Mouse (Mus musculus)  
# MAGIC - Rat (Rattus norvegicus)

# COMMAND ----------

# MAGIC %pip install pybiomart==0.2.0
# MAGIC %restart_python

# COMMAND ----------

from pybiomart import Dataset
import os
import time

# COMMAND ----------

# Get catalog and schema from job parameters
dbutils.widgets.text("catalog", "", "Catalog Name")
dbutils.widgets.text("schema", "", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

if not CATALOG or not SCHEMA:
    raise ValueError("catalog and schema parameters must be provided")

print(f"Using catalog: {CATALOG}, schema: {SCHEMA}")

# COMMAND ----------

def get_gene_table(species='hsapiens'):
    """
    Download gene reference table from Ensembl.
    
    Args:
        species: Species identifier (e.g., 'hsapiens', 'mmusculus', 'rnorvegicus')
    
    Returns:
        DataFrame with ensembl_gene_id and external_gene_name columns
    """
    print(f"Querying Ensembl for {species}...")
    dataset = Dataset(name=f'{species}_gene_ensembl',
                    host='http://www.ensembl.org')
    
    gene_table = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])
    
    # Rename BioMart's human-readable column names to code-friendly names
    gene_table.rename(columns={
        'Gene stable ID': 'ensembl_gene_id',
        'Gene name': 'external_gene_name'
    }, inplace=True)
    
    print(f"Retrieved {len(gene_table)} genes for {species}")
    return gene_table


def _download_with_retry(species, max_attempts=3):
    """BioMart (esp. the hsapiens dataset) is intermittently flaky. Retry with
    5s/15s/45s exponential backoff before propagating the last error."""
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return get_gene_table(species)
        except Exception as e:
            last_err = e
            if attempt < max_attempts:
                wait = 5 * (3 ** (attempt - 1))
                print(f"  attempt {attempt}/{max_attempts} failed: {e}; retrying in {wait}s")
                time.sleep(wait)
    raise last_err

# COMMAND ----------

# Define reference volume path (created by DBAB deployment)
REFERENCE_VOLUME = f'/Volumes/{CATALOG}/{SCHEMA}/scanpy_reference'

print(f"Scanpy reference volume: {REFERENCE_VOLUME}")

# COMMAND ----------

# Download reference tables for each species
species_list = [
    ('hsapiens', 'Human (Homo sapiens)'),
    ('mmusculus', 'Mouse (Mus musculus)'),
    ('rnorvegicus', 'Rat (Rattus norvegicus)')
]

results = []

for species_id, species_name in species_list:
    output_path = f'{REFERENCE_VOLUME}/ensembl_genes_{species_id}.csv'
    
    if os.path.exists(output_path):
        print(f"\n✓ {species_name}: Reference table already exists")
        results.append((species_id, "exists", output_path))
    else:
        try:
            print(f"\n⏳ {species_name}: Downloading from Ensembl...")
            gene_table = _download_with_retry(species_id)
            gene_table.to_csv(output_path, index=False)
            print(f"✓ {species_name}: Successfully downloaded and saved")
            results.append((species_id, "downloaded", output_path))
        except Exception as e:
            print(f"✗ {species_name}: Failed to download - {str(e)}")
            results.append((species_id, "failed", str(e)))

# COMMAND ----------

# Summary
print("\n" + "="*80)
print("DOWNLOAD SUMMARY")
print("="*80)

for species_id, status, info in results:
    species_name = dict(species_list)[species_id]
    if status == "exists":
        print(f"✓ {species_name}: Already available")
    elif status == "downloaded":
        print(f"✓ {species_name}: Downloaded successfully")
    elif status == "failed":
        print(f"✗ {species_name}: Failed - {info}")

print("="*80)

# Check if all successful
all_success = all(status in ["exists", "downloaded"] for _, status, _ in results)
if all_success:
    print("\n✓ All reference tables are ready!")
else:
    failed = [sid for sid, status, _ in results if status == "failed"]
    raise RuntimeError(
        f"Failed to download reference tables for: {failed}. "
        f"See per-species errors above. Re-run this job once BioMart is "
        f"reachable; species already on the volume will be skipped."
    )


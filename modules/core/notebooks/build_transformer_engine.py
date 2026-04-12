# Databricks notebook source
# MAGIC %md
# MAGIC ## Build Transformer Engine Wheel
# MAGIC
# MAGIC Builds the `transformer_engine_torch` wheel from source on a GPU node
# MAGIC and copies it to the UC libraries volume. This pre-built wheel is then
# MAGIC used by downstream workflows (ESM2 embeddings, sequence search) to avoid
# MAGIC rebuilding from source each time.
# MAGIC
# MAGIC **Must run on a GPU cluster** — the build requires CUDA headers.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

libraries_volume = f"/Volumes/{catalog}/{schema}/libraries"
print(f"Libraries volume: {libraries_volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check if wheel already exists

# COMMAND ----------

import os

existing_wheels = [f for f in os.listdir(libraries_volume) if f.startswith("transformer_engine_torch") and f.endswith(".whl")]

if existing_wheels:
    print(f"Transformer Engine wheel already exists: {existing_wheels}")
    print("Skipping build. Delete the wheel from the volume to force a rebuild.")
    dbutils.notebook.exit(f"SKIPPED: wheel already exists ({existing_wheels[0]})")

print("No existing wheel found. Building from source...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build transformer_engine from source

# COMMAND ----------

# MAGIC %pip install --no-build-isolation transformer_engine[pytorch]

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
libraries_volume = f"/Volumes/{catalog}/{schema}/libraries"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Locate the built wheel and copy to UC Volume

# COMMAND ----------

import glob
import shutil
import os

# Find the built wheel in pip's cache
home_dir = os.path.expanduser("~")
wheel_pattern = os.path.join(home_dir, ".cache/pip/wheels/**/transformer_engine_torch-*.whl")
wheels = glob.glob(wheel_pattern, recursive=True)

if not wheels:
    # Also check the site-packages for the installed version and rebuild wheel from it
    import subprocess
    result = subprocess.run(
        ["pip", "wheel", "--no-deps", "--no-build-isolation", "transformer_engine_torch"],
        capture_output=True, text=True, cwd="/tmp"
    )
    wheels = glob.glob("/tmp/transformer_engine_torch-*.whl")

if not wheels:
    raise RuntimeError("Could not find built transformer_engine_torch wheel. Build may have failed.")

# Use the most recently modified wheel
wheel_path = max(wheels, key=os.path.getmtime)
wheel_name = os.path.basename(wheel_path)

print(f"Found wheel: {wheel_path}")
print(f"Copying to: {libraries_volume}/{wheel_name}")

shutil.copy2(wheel_path, os.path.join(libraries_volume, wheel_name))

print(f"Successfully copied {wheel_name} to libraries volume")

# COMMAND ----------

# Verify the wheel is in the volume
for f in os.listdir(libraries_volume):
    if "transformer_engine" in f:
        full_path = os.path.join(libraries_volume, f)
        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        print(f"  {f} ({size_mb:.1f} MB)")

dbutils.notebook.exit(f"SUCCESS: {wheel_name}")

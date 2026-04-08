# Databricks notebook source

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("variant_annotation_job_id", "1234", "Variant Annotation Job ID")
dbutils.widgets.text("variant_annotation_dashboard_id", "1234", "Variant Annotation Dashboard ID")
dbutils.widgets.text("user_email", "a@b.com", "Email of the user running the deploy")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %pip install databricks-sdk>=0.50.0 databricks-sql-connector>=4.0.2 mlflow>=2.15

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")

print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
variant_annotation_job_id = dbutils.widgets.get("variant_annotation_job_id")
variant_annotation_dashboard_id = dbutils.widgets.get("variant_annotation_dashboard_id")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Variant Annotation Job ID: {variant_annotation_job_id}")
print(f"Dashboard ID: {variant_annotation_dashboard_id}")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

query = f"""
    INSERT INTO settings VALUES
    ('variant_annotation_job_id', '{variant_annotation_job_id}', 'disease_biology'),
    ('variant_annotation_dashboard_id', '{variant_annotation_dashboard_id}', 'disease_biology')
"""

spark.sql(query)

# COMMAND ----------

from genesis_workbench.workbench import set_app_permissions_for_job

set_app_permissions_for_job(job_id=variant_annotation_job_id, user_email=user_email)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Update dashboard catalog/schema references
# MAGIC The bundled dashboard JSON has hardcoded table references.
# MAGIC Replace them with this deployment's catalog and schema.

# COMMAND ----------

import json
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

try:
    dashboard = w.lakeview.get(dashboard_id=variant_annotation_dashboard_id)
    if dashboard.serialized_dashboard:
        dashboard_json = json.loads(dashboard.serialized_dashboard)

        serialized = json.dumps(dashboard_json)
        updated = serialized.replace("serverless_rg_catalog.brca_demo", f"{catalog}.{schema}")
        dashboard_json = json.loads(updated)

        w.lakeview.update(
            dashboard_id=variant_annotation_dashboard_id,
            display_name=dashboard.display_name,
            serialized_dashboard=json.dumps(dashboard_json),
            warehouse_id=dashboard.warehouse_id
        )
        print(f"Dashboard updated: table references now point to {catalog}.{schema}")
    else:
        print("Dashboard has no serialized content to update")
except Exception as e:
    print(f"Warning: Could not update dashboard references: {e}")
    print("You may need to manually update the dashboard table references.")

# COMMAND ----------

# Copy bundled demo data to the variant_annotation_data volume
import os, shutil

demo_data_dir = f"/Volumes/{catalog}/{schema}/variant_annotation_data/demo"
os.makedirs(demo_data_dir, exist_ok=True)

current_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
workspace_root = os.sep.join(current_path.split(os.sep)[:-1])
source_vcf = f"/Workspace{workspace_root}/../data/brca_pathogenic_corrected.vcf"

if os.path.exists(source_vcf):
    dest = os.path.join(demo_data_dir, "brca_pathogenic_corrected.vcf")
    shutil.copy2(source_vcf, dest)
    print(f"Copied demo pathogenic VCF to {dest}")
else:
    print(f"Demo VCF not found at {source_vcf}, skipping")

# COMMAND ----------

print("Disease Biology Variant Annotation module initialization complete")

# Databricks notebook source
# MAGIC %md
# MAGIC # RFD4-Proteina — Register jobs (Layer 2)
# MAGIC Runs once per deploy (after the orchestrator jobs exist):
# MAGIC 1. Persist `rfd4_finetune_job_id` / `rfd4_deploy_job_id` to the `settings` table so
# MAGIC    `grant_app_permissions` keeps the app SP's CAN_MANAGE_RUN grant in sync on redeploys.
# MAGIC 2. Grant the app SP CAN_MANAGE_RUN on both jobs now (so the app can dispatch immediately).
# MAGIC 3. Register RFD4-Proteina as a batch model in `batch_models` (the finetune job is the
# MAGIC    user-launchable workflow).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Parameters
# MAGIC Widgets for the catalog/schema, the two orchestrator job ids (passed in by the deploy), the SQL
# MAGIC warehouse, and the Databricks app name(s) (UI + MCP) whose service principals get the grants below.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("rfd4_finetune_job_id", "", "RFD4 finetune orchestrator job id")
dbutils.widgets.text("rfd4_deploy_job_id", "", "RFD4 deploy orchestrator job id")
dbutils.widgets.text("databricks_app_name", "genesis-workbench", "Databricks App Name")
dbutils.widgets.text("databricks_app_names", "genesis-workbench:mcp-genesis-workbench", "Databricks App Names (colon/comma-separated, UI + MCP)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Resolve the GWB library wheel
# MAGIC Find the `genesis_workbench` wheel staged in the UC `libraries` Volume so the next cell can install it.

# COMMAND ----------

gwb_library_path = None
for lib in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries"):
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")
print(gwb_library_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Install the GWB library + initialize
# MAGIC Install the resolved wheel (pinning mlflow/sdk/sql-connector to the GWB-lib versions) and restart.

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Re-read widgets after the restart, set the `DATABRICKS_APP_NAMES`/`DATABRICKS_APP_NAME` env vars
# MAGIC **before** importing the GWB helpers (they read these at import time), and `initialize()` the workbench.

# COMMAND ----------

import os
get_widget = dbutils.widgets.get
catalog, schema = get_widget("catalog"), get_widget("schema")
sql_warehouse_id, user_email = get_widget("sql_warehouse_id"), get_widget("user_email")
rfd4_finetune_job_id = get_widget("rfd4_finetune_job_id")
rfd4_deploy_job_id = get_widget("rfd4_deploy_job_id")
databricks_app_name = get_widget("databricks_app_name")
databricks_app_names = dbutils.widgets.get("databricks_app_names") or databricks_app_name

# set BEFORE importing helpers
os.environ["DATABRICKS_APP_NAMES"] = ",".join([n.strip() for n in databricks_app_names.replace(":", ",").split(",") if n.strip()])  # UI + MCP
os.environ["DATABRICKS_APP_NAME"] = databricks_app_name  # legacy single-app fallback

from genesis_workbench.workbench import initialize, set_app_permissions_for_job, set_app_permissions_for_endpoint
from genesis_workbench.models import register_batch_model, ModelCategory

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Stage the tutorial fine-tune dataset
# MAGIC Stage the tiny tutorial dataset so the DEFAULT `tutorial/finetune_test_dataset` run trains out of the
# MAGIC box (what the RFD4 fine-tune form uses when "training data" is left blank). It's one public tarball from
# MAGIC the same bucket as the checkpoints; the fine-tune job sets `DATA_PATH` to this Volume and the preset's
# MAGIC train source reads `${DATA_PATH}/test_dataset/{test_systems.parquet,cifs/}`. `PROJECT_ROOT` (for the
# MAGIC packaged val set) auto-resolves from the wheel's `.project-root`, and the preset drops the AF3 refolding
# MAGIC oracle, so nothing else is needed. Skip-if-exists → cheap on repeat deploys. Runs here (CPU) rather than
# MAGIC in `01_register` (GPU) so it adds no GPU time and doesn't churn the endpoint.

# COMMAND ----------

import os, tarfile, tempfile
import requests

RFD4_DATA_BASE_URL = "https://rfd4-proteina-public-e04a.cwobject.com"
dataset_dir = f"/Volumes/{catalog}/{schema}/rfd4_proteina/test_dataset"
dataset_marker = f"{dataset_dir}/test_systems.parquet"
if os.path.exists(dataset_marker):
    print(f"tutorial fine-tune dataset already staged at {dataset_dir}, skipping")
else:
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_url = f"{RFD4_DATA_BASE_URL}/test_dataset/test_dataset.tar.gz"
    print(f"staging tutorial fine-tune dataset <- {dataset_url}")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        with requests.get(dataset_url, stream=True, timeout=1800) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=1 << 20):
                tmp_file.write(chunk)
        tmp_path = tmp_file.name
    try:
        with tarfile.open(tmp_path, "r:gz") as tar:
            try:
                tar.extractall(dataset_dir, filter="data")  # py3.12: guards path traversal (official bucket)
            except TypeError:
                tar.extractall(dataset_dir)                  # py<3.12 has no `filter` kwarg
    finally:
        os.remove(tmp_path)
    assert os.path.exists(dataset_marker), f"staging did not produce {dataset_marker}; check tarball layout"
    print(f"staged tutorial fine-tune dataset -> {dataset_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Persist the orchestrator job ids to `settings`
# MAGIC Key `LIKE '%_job_id'` is what `grant_app_permissions` queries to keep the app SP's grants in sync.

# COMMAND ----------

for key, val in (("rfd4_finetune_job_id", rfd4_finetune_job_id),
                 ("rfd4_deploy_job_id", rfd4_deploy_job_id)):
    spark.sql(f"""
        MERGE INTO settings AS target
        USING (SELECT '{key}' AS key, '{val}' AS value, 'rfd4_proteina' AS module) AS source
        ON target.key = source.key AND target.module = source.module
        WHEN MATCHED THEN UPDATE SET target.value = source.value
        WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
    """)
    print(f"settings: {key} = {val}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Grant the app SP CAN_MANAGE_RUN on both jobs
# MAGIC So the app can dispatch the finetune + deploy jobs immediately.

# COMMAND ----------

set_app_permissions_for_job(job_id=rfd4_finetune_job_id, user_email=user_email)
set_app_permissions_for_job(job_id=rfd4_deploy_job_id, user_email=user_email)
print("app permissions granted on finetune + deploy jobs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Grant the app SP(s) CAN_QUERY on the live RFD4-Proteina serving endpoint
# MAGIC `01_register_rfd4_proteina` deploys the endpoint via the EXPRESS `create_and_wait` path — unlike the
# MAGIC standard `deploy_model` path, that does NOT call `set_app_permissions_for_endpoint`, so without this the
# MAGIC app's models/settings page shows `PermissionDenied ... 'View' on Endpoint gwb_*_rfd4_proteina_endpoint`.
# MAGIC This task `depends_on register_rfd4_proteina_task`, so the endpoint already exists here, and
# MAGIC `DATABRICKS_APP_NAMES` was set above so the grant covers both the UI + MCP app SPs.

# COMMAND ----------

try:
    dev_user_prefix = dbutils.secrets.get("dbx_genesis_workbench", "dev_user_prefix")
except Exception:
    dev_user_prefix = ""
rfd4_endpoint = f"gwb_{dev_user_prefix}_rfd4_proteina_endpoint" if dev_user_prefix and dev_user_prefix.strip() else "gwb_rfd4_proteina_endpoint"
try:
    set_app_permissions_for_endpoint(rfd4_endpoint)
    print(f"granted app SP(s) CAN_QUERY on {rfd4_endpoint}")
except Exception as error:
    print(f"WARNING: could not grant app perms on {rfd4_endpoint}: {error}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Register RFD4-Proteina as a batch model
# MAGIC The finetune job is the user-launchable workflow surfaced in the GWB UI.

# COMMAND ----------

register_batch_model(
    model_name="rfd4_proteina",
    model_display_name="RFD4-Proteina",
    model_description="Flow-matching foundation model for biomolecular design (de novo, motif/binder/antibody/enzyme, partial diffusion, reward-guided search); fine-tune + serve.",
    model_category=str(ModelCategory.LARGE_MOLECULE),
    module="rfd4_proteina",
    job_id=rfd4_finetune_job_id,
    job_name="rfd4_finetune_job",
    cluster_type="GPU",
    added_by=user_email,
)
print("registered RFD4-Proteina in batch_models")

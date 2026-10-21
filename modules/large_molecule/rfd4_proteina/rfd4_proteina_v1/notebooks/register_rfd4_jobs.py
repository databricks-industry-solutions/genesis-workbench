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

gwb_library_path = None
for lib in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries"):
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")
print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
g = dbutils.widgets.get
catalog, schema = g("catalog"), g("schema")
sql_warehouse_id, user_email = g("sql_warehouse_id"), g("user_email")
rfd4_finetune_job_id = g("rfd4_finetune_job_id")
rfd4_deploy_job_id = g("rfd4_deploy_job_id")
databricks_app_name = g("databricks_app_name")
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

# 0. Stage the tiny tutorial fine-tune dataset so the DEFAULT `tutorial/finetune_test_dataset` run
# trains out of the box (this is what the RFD4 fine-tune form uses when "training data" is left blank).
# It's one public tarball from the same bucket as the checkpoints; the fine-tune job sets DATA_PATH to
# this volume and the preset's train source reads ${DATA_PATH}/test_dataset/{test_systems.parquet,cifs/}.
# PROJECT_ROOT (for the packaged val set) auto-resolves from the wheel's .project-root, and the preset
# drops the AF3 refolding oracle, so nothing else is needed. Skip-if-exists -> cheap on repeat deploys.
# Runs here (CPU) rather than in 01_register (GPU) so it adds no GPU time and doesn't churn the endpoint.
import os, tarfile, tempfile
import requests

RFD4_DATA_BASE_URL = "https://rfd4-proteina-public-e04a.cwobject.com"
_ds_dir = f"/Volumes/{catalog}/{schema}/rfd4_proteina/test_dataset"
_ds_marker = f"{_ds_dir}/test_systems.parquet"
if os.path.exists(_ds_marker):
    print(f"tutorial fine-tune dataset already staged at {_ds_dir}, skipping")
else:
    os.makedirs(_ds_dir, exist_ok=True)
    _url = f"{RFD4_DATA_BASE_URL}/test_dataset/test_dataset.tar.gz"
    print(f"staging tutorial fine-tune dataset <- {_url}")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as _tmp:
        with requests.get(_url, stream=True, timeout=1800) as _r:
            _r.raise_for_status()
            for _chunk in _r.iter_content(chunk_size=1 << 20):
                _tmp.write(_chunk)
        _tmp_path = _tmp.name
    try:
        with tarfile.open(_tmp_path, "r:gz") as _tf:
            try:
                _tf.extractall(_ds_dir, filter="data")  # py3.12: guards path traversal (official bucket)
            except TypeError:
                _tf.extractall(_ds_dir)                  # py<3.12 has no `filter` kwarg
    finally:
        os.remove(_tmp_path)
    assert os.path.exists(_ds_marker), f"staging did not produce {_ds_marker}; check tarball layout"
    print(f"staged tutorial fine-tune dataset -> {_ds_dir}")

# COMMAND ----------

# 1. Persist job ids to settings (key LIKE '%_job_id' is what grant_app_permissions queries).
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

# 2. Grant the app SP CAN_MANAGE_RUN on both jobs.
set_app_permissions_for_job(job_id=rfd4_finetune_job_id, user_email=user_email)
set_app_permissions_for_job(job_id=rfd4_deploy_job_id, user_email=user_email)
print("app permissions granted on finetune + deploy jobs")

# COMMAND ----------

# 3. Grant the app SP(s) CAN_QUERY on the live RFD4-Proteina serving endpoint.
# nb01 (01_register_rfd4_proteina) deploys the endpoint via the EXPRESS create_and_wait path — unlike the
# standard deploy_model path, that does NOT call set_app_permissions_for_endpoint, so without this the app's
# models/settings page shows "PermissionDenied: User does not have permission 'View' on Endpoint
# gwb_*_rfd4_proteina_endpoint". This task depends_on register_rfd4_proteina_task, so the endpoint already
# exists here, and DATABRICKS_APP_NAMES was set above so the grant covers both the UI + MCP app SPs.
try:
    _pfx = dbutils.secrets.get("dbx_genesis_workbench", "dev_user_prefix")
except Exception:
    _pfx = ""
_rfd4_endpoint = f"gwb_{_pfx}_rfd4_proteina_endpoint" if _pfx and _pfx.strip() else "gwb_rfd4_proteina_endpoint"
try:
    set_app_permissions_for_endpoint(_rfd4_endpoint)
    print(f"granted app SP(s) CAN_QUERY on {_rfd4_endpoint}")
except Exception as e:
    print(f"WARNING: could not grant app perms on {_rfd4_endpoint}: {e}")

# COMMAND ----------

# 3. Register RFD4-Proteina as a batch model (the finetune job is the user-launchable workflow).
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

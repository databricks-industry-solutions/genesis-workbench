# Databricks notebook source
# MAGIC %md
# MAGIC ### Grant App Permissions
# MAGIC
# MAGIC Grants every Genesis Workbench app service principal:
# MAGIC - `CAN_QUERY` on all serving endpoints
# MAGIC - `CAN_MANAGE_RUN` on all registered jobs (from the settings table)
# MAGIC - `READ VOLUME` and `WRITE VOLUME` on the schema
# MAGIC - `EXECUTE` on all registered models in Unity Catalog
# MAGIC
# MAGIC Reads `databricks_app_names` (comma-separated, e.g. "genesis-workbench,gwb-react")
# MAGIC and falls back to `databricks_app_name` (legacy single value) when the multi-app
# MAGIC widget is empty. Iterates every app so multi-app installs (Streamlit + React)
# MAGIC don't end up with one SP missing CAN_QUERY on serving endpoints.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("databricks_app_name", "genesis-workbench", "Databricks App Name (legacy single)")
dbutils.widgets.text("databricks_app_names", "", "Databricks App Names (comma-separated, multi-app)")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
databricks_app_name = dbutils.widgets.get("databricks_app_name")
databricks_app_names = dbutils.widgets.get("databricks_app_names") or databricks_app_name
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")

print(f"GWB library: {gwb_library_path}")

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
databricks_app_name = dbutils.widgets.get("databricks_app_name")
databricks_app_names = dbutils.widgets.get("databricks_app_names") or databricks_app_name
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# Colon separator avoids clashing with the comma that `databricks bundle --var`
# uses to split multiple var=value pairs. The library still expects
# comma-separated in DATABRICKS_APP_NAMES, so normalise here.
app_name_list = [n.strip() for n in databricks_app_names.replace(":", ",").split(",") if n.strip()]
databricks_app_names_csv = ",".join(app_name_list)
print(f"Apps to grant: {app_name_list}")

# COMMAND ----------

import os
os.environ["DATABRICKS_APP_NAMES"] = databricks_app_names_csv
os.environ["DATABRICKS_APP_NAME"] = databricks_app_name  # legacy fallback

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServingEndpointAccessControlRequest, ServingEndpointPermissionLevel
from databricks.sdk.service.jobs import JobAccessControlRequest, JobPermissionLevel

import random
import time

w = WorkspaceClient()

# Collect operations that couldn't be patched even after retries — we raise at
# the end so the parent job FAILs loudly instead of returning SUCCESS with
# silent gaps (which is what masked the missing-CAN_QUERY rows previously).
failures: list[str] = []


def _retry(op_name: str, fn, max_attempts: int = 5, base_delay: float = 1.0):
    """Run fn() with exponential backoff. Returns fn()'s result, or appends to
    `failures` and re-raises if every attempt errors. Delays: 1s, 2s, 4s, 8s
    + up to 250ms jitter. Tuned for Databricks REST 429s and transient 5xx,
    which is what was eating per-endpoint grants in the previous deploys."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts:
                msg = f"{op_name} failed after {max_attempts} attempts: {e}"
                failures.append(msg)
                print(f"  ❌ {msg}")
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            print(f"  ⏳ {op_name} attempt {attempt}/{max_attempts} hit '{type(e).__name__}: {str(e)[:80]}', retrying in {delay:.1f}s")
            time.sleep(delay)


app_sps: list[tuple[str, str]] = []
for n in app_name_list:
    try:
        app_sps.append((n, _retry(f"apps.get({n})", lambda n=n: w.apps.get(name=n).service_principal_client_id)))
    except Exception as e:
        print(f"⚠️  Skipping app '{n}': {e}")

print(f"Resolved SPs: {[(n, sp) for n, sp in app_sps]}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Grant CAN_QUERY on all serving endpoints

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

endpoint_names_df = spark.sql(
    "SELECT DISTINCT model_endpoint_name FROM model_deployments WHERE is_active = true AND model_endpoint_name IS NOT NULL"
).collect()

print(f"Found {len(endpoint_names_df)} deployed endpoints in model_deployments")

for row in endpoint_names_df:
    ep_name = row["model_endpoint_name"]
    try:
        ep = _retry(f"serving_endpoints.get({ep_name})", lambda: w.serving_endpoints.get(ep_name))
        perms = _retry(
            f"serving_endpoints.get_permissions({ep_name})",
            lambda: w.serving_endpoints.get_permissions(serving_endpoint_id=ep.id),
        )
        existing_query_sps = {
            acl.user_name
            for acl in (perms.access_control_list or [])
            for p in (acl.all_permissions or [])
            if p.permission_level and "CAN_QUERY" in str(p.permission_level)
        }
        missing = [(name, sp) for name, sp in app_sps if sp not in existing_query_sps]
        if not missing:
            print(f"  All apps already have CAN_QUERY on {ep_name}, skipping")
            continue
        # update_permissions is additive (PATCH semantics) — won't clobber existing ACL.
        _retry(
            f"serving_endpoints.update_permissions({ep_name})",
            lambda: w.serving_endpoints.update_permissions(
                serving_endpoint_id=ep.id,
                access_control_list=[
                    ServingEndpointAccessControlRequest(
                        user_name=sp,
                        permission_level=ServingEndpointPermissionLevel.CAN_QUERY,
                    )
                    for _, sp in missing
                ],
            ),
        )
        for name, _ in missing:
            print(f"  Granted CAN_QUERY on {ep_name} to {name}")
    except Exception:
        # _retry already logged + appended to `failures`; continue so we
        # surface every offender rather than stopping at the first one.
        continue

# COMMAND ----------

# MAGIC %md
# MAGIC #### Grant CAN_MANAGE_RUN on all registered jobs

# COMMAND ----------

job_ids_df = spark.sql("SELECT value FROM settings WHERE key LIKE '%_job_id'").collect()

print(f"Found {len(job_ids_df)} registered jobs")

for row in job_ids_df:
    job_id = row["value"]
    try:
        perms = _retry(
            f"jobs.get_permissions({job_id})",
            lambda: w.jobs.get_permissions(job_id=job_id),
        )
        existing_run_sps = {
            acl.user_name
            for acl in (perms.access_control_list or [])
            for p in (acl.all_permissions or [])
            if p.permission_level and "CAN_MANAGE_RUN" in str(p.permission_level)
        }
        missing = [(name, sp) for name, sp in app_sps if sp not in existing_run_sps]
        if not missing:
            print(f"  All apps already have CAN_MANAGE_RUN on job {job_id}, skipping")
            continue
        _retry(
            f"jobs.update_permissions({job_id})",
            lambda: w.jobs.update_permissions(
                job_id=job_id,
                access_control_list=[
                    JobAccessControlRequest(
                        user_name=sp,
                        permission_level=JobPermissionLevel.CAN_MANAGE_RUN,
                    )
                    for _, sp in missing
                ],
            ),
        )
        for name, _ in missing:
            print(f"  Granted CAN_MANAGE_RUN on job {job_id} to {name}")
    except Exception:
        continue

# COMMAND ----------

# MAGIC %md
# MAGIC #### Grant READ VOLUME and WRITE VOLUME on all volumes in the schema

# COMMAND ----------

for app_name, app_sp_id in app_sps:
    try:
        _retry(
            f"GRANT READ VOLUME for {app_name}",
            lambda: spark.sql(f"GRANT READ VOLUME ON SCHEMA {catalog}.{schema} TO `{app_sp_id}`"),
        )
        print(f"Granted READ VOLUME on schema {catalog}.{schema} to {app_name}")
    except Exception:
        pass

    # WRITE VOLUME is required so the app can upload per-run inputs (e.g. the
    # motif PDB the Guided Enzyme Optimization form persists to UC before
    # dispatching the orchestrator job — otherwise the form errors with
    # "Failed to dispatch job: [Errno 13] Permission denied: '/Volumes'").
    try:
        _retry(
            f"GRANT WRITE VOLUME for {app_name}",
            lambda: spark.sql(f"GRANT WRITE VOLUME ON SCHEMA {catalog}.{schema} TO `{app_sp_id}`"),
        )
        print(f"Granted WRITE VOLUME on schema {catalog}.{schema} to {app_name}")
    except Exception:
        pass

# COMMAND ----------

# MAGIC %md
# MAGIC #### Grant EXECUTE on all registered models in Unity Catalog

# COMMAND ----------

model_names_df = spark.sql(
    f"SELECT DISTINCT deploy_model_uc_name FROM model_deployments WHERE is_active = true AND deploy_model_uc_name IS NOT NULL"
).collect()

print(f"Found {len(model_names_df)} registered models in model_deployments")

for row in model_names_df:
    model_uc_name = row["deploy_model_uc_name"]
    for app_name, app_sp_id in app_sps:
        try:
            _retry(
                f"GRANT EXECUTE on {model_uc_name} to {app_name}",
                lambda: spark.sql(f"GRANT EXECUTE ON FUNCTION {model_uc_name} TO `{app_sp_id}`"),
            )
            print(f"  Granted EXECUTE on {model_uc_name} to {app_name}")
        except Exception as e:
            # 'already has'/'inherited' are not real errors — pop them off the
            # `failures` list so the job doesn't fail on benign no-ops.
            if "already has" in str(e).lower() or "inherited" in str(e).lower():
                if failures and model_uc_name in failures[-1]:
                    failures.pop()
                print(f"  {app_name} already has EXECUTE on {model_uc_name}, skipping")

# COMMAND ----------

if failures:
    summary = "\n".join(f"  - {f}" for f in failures)
    raise RuntimeError(
        f"grant_app_permissions completed with {len(failures)} unrecoverable failure(s):\n{summary}"
    )

print("App permissions grant complete — all operations succeeded.")

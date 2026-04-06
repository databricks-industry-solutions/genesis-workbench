# Databricks notebook source
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")
dbutils.widgets.text("workload_type", "GPU_SMALL", "Workload Type for endpoints")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if(lib.name.startswith("genesis_workbench")):
        gwb_library_path = lib.path.replace("dbfs:","")

print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
workload_type = dbutils.widgets.get("workload_type")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)


# COMMAND ----------

from genesis_workbench.models import (ModelCategory, 
                                      import_model_from_uc,
                                      deploy_model,
                                      get_latest_model_version)

from genesis_workbench.workbench import wait_for_job_run_completion

# COMMAND ----------

model_name = "diffdock_v1"
model_uc_name = f"{catalog}.{schema}.{model_name}"
model_version = get_latest_model_version(model_uc_name)

gwb_model_id = import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.SMALL_MOLECULES,
                    model_uc_name=model_uc_name,
                    model_uc_version=model_version,
                    model_name="DiffDock",
                    model_display_name="DiffDock Molecular Docking",
                    model_source_version="v1.0 (commit 0f9c419)",
                    model_description_url="https://github.com/gcorso/DiffDock")

# COMMAND ----------

run_id = deploy_model(user_email=user_email,
                gwb_model_id=gwb_model_id,
                deployment_name=f"DiffDock Molecular Docking",
                deployment_description="DiffDock molecular docking — predicts 3D binding poses for protein-ligand complexes using diffusion generative modeling with confidence ranking",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type=workload_type,
                workload_size="Small")

# COMMAND ----------

result = wait_for_job_run_completion(run_id, timeout=3600)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Warm up the endpoint
# MAGIC DiffDock uses lazy model loading — ESM2 + score + confidence models are loaded on
# MAGIC the first predict() call. This warm-up sends an initial request so the models are
# MAGIC pre-loaded before users hit the endpoint.

# COMMAND ----------

import requests
import time
import json
import mlflow

db_host = spark.conf.get("spark.databricks.workspaceUrl")
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
headers = {"Authorization": f"Bearer {db_token}", "Content-Type": "application/json"}

# Get the endpoint name from the model_deployments table
from genesis_workbench.workbench import execute_select_query
endpoint_df = execute_select_query(f"""
    SELECT model_endpoint_name, deploy_model_uc_name, deploy_model_uc_version
    FROM {catalog}.{schema}.model_deployments
    WHERE model_id = {gwb_model_id} AND is_active = true
    ORDER BY deployment_id DESC LIMIT 1
""")

if len(endpoint_df) == 0:
    print("No active deployment found. Skipping warm-up.")
    dbutils.notebook.exit("No deployment to warm up")

endpoint_name = endpoint_df.iloc[0]["model_endpoint_name"]
uc_name = endpoint_df.iloc[0]["deploy_model_uc_name"]
uc_version = endpoint_df.iloc[0]["deploy_model_uc_version"]
print(f"Endpoint: {endpoint_name}")
print(f"Model: {uc_name}/{uc_version}")

# Load the input_example from the registered model
mlflow.set_registry_uri("databricks-uc")
model_info = mlflow.models.get_model_info(f"models:/{uc_name}/{uc_version}")
artifact_info = model_info.saved_input_example_info
if artifact_info:
    artifact_path = artifact_info.get("artifact_path")
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"{model_info.model_uri}/{artifact_path}"
    )
    with open(local_path, 'r') as f:
        warmup_payload = json.load(f)
    print(f"Loaded warm-up payload from model input_example")
else:
    print("No input_example found in model. Skipping warm-up.")
    dbutils.notebook.exit("No input_example for warm-up")

# COMMAND ----------

# Wait for endpoint to be READY
def check_endpoint_ready(host, name, token):
    url = f"https://{host}/api/2.0/serving-endpoints/{name}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    if resp.status_code == 200:
        return resp.json().get("state", {}).get("ready", "Unknown")
    return "Unknown"

print(f"Waiting for endpoint '{endpoint_name}' to be READY...")
max_wait_mins = 30
for i in range(max_wait_mins):
    status = check_endpoint_ready(db_host, endpoint_name, db_token)
    if status == "READY":
        print(f"Endpoint is READY after {i} minute(s)")
        break
    print(f"  Status: {status} — checking again in 60s ({i+1}/{max_wait_mins})")
    time.sleep(60)
else:
    print(f"WARNING: Endpoint not ready after {max_wait_mins} minutes. Attempting warm-up anyway.")

# COMMAND ----------

# Send warm-up request with retries (first call triggers lazy model loading and will likely timeout)
invoke_url = f"https://{db_host}/serving-endpoints/{endpoint_name}/invocations"
max_attempts = 5
request_timeout = 600  # 10 minutes per attempt
delay_between_attempts = 180  # 3 minutes

warmup_success = False
for attempt in range(1, max_attempts + 1):
    print(f"\nWarm-up attempt {attempt}/{max_attempts} (timeout: {request_timeout}s)...")
    try:
        resp = requests.post(
            invoke_url, headers=headers,
            data=json.dumps(warmup_payload),
            timeout=request_timeout
        )
        if resp.status_code == 200:
            print(f"Warm-up succeeded on attempt {attempt}!")
            warmup_success = True
            break
        else:
            print(f"  Response status: {resp.status_code}")
            print(f"  Response: {resp.text[:500]}")
    except requests.exceptions.Timeout:
        print(f"  Request timed out (expected on first call — models are loading)")
    except Exception as e:
        print(f"  Error: {e}")

    if attempt < max_attempts:
        print(f"  Retrying in {delay_between_attempts}s...")
        time.sleep(delay_between_attempts)

if warmup_success:
    print("\nDiffDock endpoint is warm and ready for inference.")
else:
    raise RuntimeError(f"DiffDock endpoint warm-up failed after {max_attempts} attempts. "
                       "The endpoint may need manual investigation.")

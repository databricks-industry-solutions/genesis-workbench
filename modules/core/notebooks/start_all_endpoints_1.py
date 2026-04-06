# Databricks notebook source
# MAGIC %md
# MAGIC ### Start All Endpoints
# MAGIC Starts all deployed model serving endpoints and keeps them alive by sending periodic pings with sample data.
# MAGIC Input examples are read directly from the MLflow model registry.

# COMMAND ----------

dbutils.widgets.text("catalog", "srijit_nair_ci_demo_catalog", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("sql_warehouse_id", "045df48d4afed522", "SQL Warehouse Id")
dbutils.widgets.text("num_hours", "4", "Number of hours to keep alive")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
num_hours = int(dbutils.widgets.get("num_hours"))

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

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
num_hours = int(dbutils.widgets.get("num_hours"))

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor

db_host = spark.conf.get("spark.databricks.workspaceUrl")
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

def start_endpoint(databricks_instance, endpoint_name, token):
    url = f"https://{databricks_instance}/api/2.0/serving-endpoints/{endpoint_name}/config:start"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        print(f"Successfully started endpoint: {endpoint_name}")
    else:
        print(f"Failed to start endpoint: {endpoint_name}, Status Code: {response.status_code}, Response: {response.text}")

def check_endpoint_status(databricks_instance, endpoint_name, token):
    url = f"https://{databricks_instance}/api/2.0/serving-endpoints/{endpoint_name}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        status = response.json().get("state", {}).get("ready", "Unknown")
        print(f"Endpoint: {endpoint_name}, Status: {status}")
        return status
    else:
        print(f"Failed to check status of endpoint: {endpoint_name}, Status Code: {response.status_code}")
        return "Unknown"

def wait_until_endpoints_ready(databricks_instance, endpoints, token, check_interval=60):
    all_ready = False
    while not all_ready:
        all_ready = True
        for endpoint in endpoints:
            status = check_endpoint_status(databricks_instance, endpoint, token)
            if status != "READY":
                all_ready = False
                print(f"Endpoint {endpoint} is not ready yet. Checking again in {check_interval} seconds...")
                break
        if not all_ready:
            time.sleep(check_interval)
    print("All endpoints are ready!")

def send_request(databricks_instance, endpoint_name, payload, token):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    url = f"https://{databricks_instance}/serving-endpoints/{endpoint_name}/invocations"
    try:
        json_payload = json.dumps(payload)
        response = requests.post(url, headers=headers, data=json_payload)
        if response.status_code == 200:
            print(f"Endpoint {endpoint_name} responded successfully.")
        else:
            print(f"Endpoint {endpoint_name} returned status {response.status_code}. Probably still waking up.")
    except Exception as e:
        print(f"Error sending request to endpoint {endpoint_name}: {e}")

def send_pings_parallel(databricks_instance, endpoint_payloads, token):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(send_request, databricks_instance, ep_name, payload, token)
            for ep_name, payload in endpoint_payloads.items()
        ]
        for future in futures:
            future.result()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load active endpoints and retrieve input_example from MLflow model registry

# COMMAND ----------

import mlflow
from genesis_workbench.workbench import execute_select_query
import os

mlflow.set_registry_uri("databricks-uc")

core_catalog = os.environ["CORE_CATALOG_NAME"]
core_schema = os.environ["CORE_SCHEMA_NAME"]

query = f"""
    SELECT model_endpoint_name, deploy_model_uc_name, deploy_model_uc_version
    FROM {core_catalog}.{core_schema}.model_deployments
    WHERE is_active = true
"""
endpoints_df = execute_select_query(query)

endpoint_payloads = {}
endpoint_names = []

for _, row in endpoints_df.iterrows():
    ep_name = row['model_endpoint_name']
    model_uc_name = row['deploy_model_uc_name']
    model_uc_version = row['deploy_model_uc_version']

    try:
        model_uri = f"models:/{model_uc_name}/{model_uc_version}"
        model_info = mlflow.models.get_model_info(model_uri)

        if model_info.saved_input_example_info:
            artifact_path = model_info.saved_input_example_info.get("artifact_path")
            input_example_type = model_info.saved_input_example_info.get("type")

            # Try multiple strategies to download the input_example artifact
            local_path = None
            download_errors = []

            # Strategy 1: Use the models:/ URI directly (works for most UC models)
            try:
                local_path = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{model_uc_name}/{model_uc_version}/{artifact_path}")
            except Exception as e1:
                download_errors.append(f"models:/ URI: {e1}")

            # Strategy 2: Use run_id with just the filename
            if local_path is None and model_info.run_id:
                try:
                    local_path = mlflow.artifacts.download_artifacts(run_id=model_info.run_id, artifact_path=f"model/{artifact_path}")
                except Exception as e2:
                    download_errors.append(f"run_id+model/: {e2}")

            # Strategy 3: Use run_id with just the filename at root
            if local_path is None and model_info.run_id:
                try:
                    local_path = mlflow.artifacts.download_artifacts(run_id=model_info.run_id, artifact_path=artifact_path)
                except Exception as e3:
                    download_errors.append(f"run_id+root: {e3}")

            if local_path is None:
                print(f"WARNING: All download strategies failed for {ep_name}: {download_errors}. Skipping.")
                continue
            with open(local_path, 'r') as f:
                raw_example = json.load(f)

            # Wrap in the correct serving format
            if input_example_type == "dataframe" or ("columns" in raw_example and "data" in raw_example):
                payload = {"dataframe_split": raw_example}
            elif isinstance(raw_example, list):
                payload = {"inputs": raw_example}
            else:
                payload = {"inputs": [raw_example]}

            endpoint_payloads[ep_name] = payload
            endpoint_names.append(ep_name)
            print(f"Loaded input_example for endpoint: {ep_name} (type: {input_example_type})")
        else:
            print(f"WARNING: No input_example found for {ep_name} (model: {model_uc_name}/{model_uc_version}). Skipping.")
    except Exception as e:
        print(f"WARNING: Could not load input_example for {ep_name}: {e}. Skipping.")

print(f"\nTotal endpoints to keep alive: {len(endpoint_names)}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Start endpoints, wait for readiness, and keep alive

# COMMAND ----------

if len(endpoint_names) == 0:
    print("No endpoints with valid input_example found. Nothing to do.")
    dbutils.notebook.exit("No endpoints to start")

num_mins_to_ping = 15

# Start all endpoints
print("Starting all endpoints...")
for ep_name in endpoint_names:
    start_endpoint(db_host, ep_name, db_token)

# Wait for all to be ready
print("\nWaiting for endpoints to become ready...")
wait_until_endpoints_ready(db_host, endpoint_names, db_token, check_interval=60)

# Initial ping
print("\nSending initial ping to all endpoints...")
send_pings_parallel(db_host, endpoint_payloads, db_token)

# Keep alive loop
total_pings = num_hours * 60 // num_mins_to_ping
print(f"\nKeeping endpoints alive for {num_hours} hours ({total_pings} pings every {num_mins_to_ping} minutes)")

for i in range(total_pings):
    print(f"\n--- Ping {i+1}/{total_pings} ---")
    time.sleep(num_mins_to_ping * 60)
    send_pings_parallel(db_host, endpoint_payloads, db_token)

print("\nKeep-alive period complete.")

# Databricks notebook source
!nvidia-smi

# COMMAND ----------

#dbutils.widgets.removeAll()

# COMMAND ----------

dbutils.widgets.text("core_catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("core_schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")
dbutils.widgets.text("is_base_model", "false", "Use Base Model?")
dbutils.widgets.text("esm_variant", "650M", "ESM Variant")
dbutils.widgets.text("task_type", "regression", "Task type: Regression or Classification")
dbutils.widgets.text("finetune_run_id", "3", "Finetune Run Id")
dbutils.widgets.text("data_location", "", "Training data location")
dbutils.widgets.text("sequence_column_name", "sequence", "Column name containing the sequence")
dbutils.widgets.text("result_location", "", "Result Location in UC Volume")
dbutils.widgets.text("user_email", "a@b.com", "User Email")
dbutils.widgets.text("experiment_name", "gwb_bionemo_esm2_inference", "MLflow experiment name")
dbutils.widgets.text("run_name", "esm2_inference", "MLflow run name")
dbutils.widgets.text("mlflow_run_id", "", "Pre-created MLflow run id (from the app dispatcher)")

catalog = dbutils.widgets.get("core_catalog")
schema = dbutils.widgets.get("core_schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Required Libraries

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
import shutil
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### Work Directory

# COMMAND ----------

work_dir = "/workdir"
catalog = dbutils.widgets.get("core_catalog")
schema = dbutils.widgets.get("core_schema")
is_base_model = True if dbutils.widgets.get("is_base_model") == "true" else False
esm_variant = dbutils.widgets.get("esm_variant")
task_type = dbutils.widgets.get("task_type")
finetune_run_id = dbutils.widgets.get("finetune_run_id")
data_location = dbutils.widgets.get("data_location")
sequence_column_name = dbutils.widgets.get("sequence_column_name")
result_location = dbutils.widgets.get("result_location")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
experiment_name = dbutils.widgets.get("experiment_name")
run_name = dbutils.widgets.get("run_name")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id") or None

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow run (status + search)

# COMMAND ----------

# Resume the app dispatcher's pre-created run (status already "submitted"), or
# create one if launched standalone — then advance job_status as inference runs.
# Mirrors the Fine Tune notebook so the Inference tab's Search Past Runs works.
import mlflow
from genesis_workbench.workbench import initialize
from genesis_workbench.models import set_mlflow_experiment

_db_host = spark.conf.get("spark.databricks.workspaceUrl")
_db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
os.environ["DATABRICKS_HOST"] = _db_host
os.environ["DATABRICKS_TOKEN"] = _db_token
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=_db_token)

experiment = set_mlflow_experiment(experiment_tag=experiment_name, user_email=user_email)
if mlflow_run_id:
    mlflow.start_run(run_id=mlflow_run_id)
else:
    mlflow.start_run(run_name=run_name or "esm2_inference", experiment_id=experiment.experiment_id)
mlflow.set_tag("origin", "genesis_workbench")
mlflow.set_tag("feature", "bionemo_esm_inference")
mlflow.set_tag("created_by", user_email)
mlflow.set_tag("result_location", result_location)
mlflow.log_param("esm_variant", esm_variant)
mlflow.log_param("is_base_model", str(is_base_model))
mlflow.set_tag("job_status", "running")

# COMMAND ----------

if os.path.exists(work_dir):
    shutil.rmtree(work_dir)

data_dir = f"{work_dir}/data"
results_dir = f"{work_dir}/results"
ft_weights_directory = f"{work_dir}/ft_weights"

os.makedirs(work_dir)
os.makedirs(data_dir)
os.makedirs(results_dir)
os.makedirs(ft_weights_directory)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download Pre-trained Model Checkpoints

# COMMAND ----------

from genesis_workbench.bionemo import BionemoModelType, GWBBionemoFTInfo

def get_ft_run_details(run_id:int) -> GWBBionemoFTInfo:
    query = f"SELECT * FROM {catalog}.{schema}.bionemo_weights WHERE ft_id = {run_id}"
    df = spark.sql(query)
    
    if df.count() == 0:
        raise Exception(f"Finetune run {run_id} not found")
    run_info = df.toPandas().apply(lambda row: GWBBionemoFTInfo(**row), axis=1).tolist()[0]
    return run_info


# COMMAND ----------

from bionemo.core.data.load import load

model_weights_location = ""
if is_base_model:    
    model_weights_location = load(f"esm2/{esm_variant.lower()}:2.0")
    print(model_weights_location)
else:
    run_info = get_ft_run_details(int(finetune_run_id))
    weights_volume_location = run_info.weights_volume_location
    shutil.rmtree(ft_weights_directory)
    shutil.copytree(weights_volume_location, ft_weights_directory)
    model_weights_location = ft_weights_directory
    print(f"Copied weights from {weights_volume_location} to {ft_weights_directory}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Inference

# COMMAND ----------

import os
import io
from databricks.sdk import WorkspaceClient

db_host = spark.conf.get("spark.databricks.workspaceUrl")
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Copy data locally

# COMMAND ----------

# w = WorkspaceClient(host=db_host, token=databricks_token)

# def download_data(remote_path, local_file_name):
#   file_content = ""
#   sequences = []
#   for f in w.files.list_directory_contents(remote_path):
#     print(f"Downloading {f.path}")
#     with w.files.download(f.path).contents as remote_file:
#       file_content += remote_file.read().decode("utf-8")

#   for seq in file_content.split("\n"):
#     seq = seq.strip()
#     if len(seq)>0:
#       sequences.append((seq.split(",")[0], seq.split(",")[1]))

#   # Create a DataFrame
#   df = pd.DataFrame(sequences, columns=["sequences", "labels"])

#   # Save the DataFrame to a CSV file
#   print(f"Writing to {local_file_name}")
#   df.to_csv(local_file_name, index=False)


# COMMAND ----------

import pandas as pd

workdir_data_file = f"{data_dir}/data.csv"

infer_data = pd.read_csv(data_location)

infer_data[[sequence_column_name]].rename(columns={sequence_column_name: "sequences"}).to_csv(workdir_data_file , index=False)



# COMMAND ----------

is_inference_success = False

# COMMAND ----------

! infer_esm2 --checkpoint-path {model_weights_location} \
             --config-class ESM2FineTuneSeqConfig \
             --data-path {workdir_data_file} \
             --results-path {results_dir} \
             --micro-batch-size 3 \
             --num-gpus 1 \
             --precision "bf16-mixed" \
             --include-embeddings \
             --include-input-ids

# COMMAND ----------

!ls -al {results_dir}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Copy the results back into volume
# MAGIC

# COMMAND ----------

import torch


results = torch.load(f"{results_dir}/predictions__rank_0.pt")

for key, val in results.items():
    if val is not None:
        print(f"{key}\t{val.shape}")

# COMMAND ----------

results_df = pd.read_csv(workdir_data_file)
if "classification_output" in results and results["classification_output"] is not None:
    results_df["predictions"] = [r.argmax().item() for r in results["classification_output"].tolist()]
elif "regression_output" in results and results["regression_output"] is not None:
    results_df["predictions"] = [r[0] for r in results["regression_output"].tolist()]
else:
    available_keys = [k for k, v in results.items() if v is not None]
    raise KeyError(f"No regression_output or classification_output found in results. Available keys: {available_keys}")
results_df

# COMMAND ----------

# Save the DataFrame to a CSV file
results_file_name = f"{result_location}/results.csv"
os.makedirs(result_location, exist_ok=True)
print(f"Writing to {results_file_name}")
results_df.to_csv(results_file_name, index=False)

# COMMAND ----------

# Mark the run complete so the Inference tab's Search Past Runs enables "View results".
mlflow.log_metric("num_sequences", int(len(results_df)))
mlflow.set_tag("results_file", results_file_name)
mlflow.set_tag("job_status", "complete")
mlflow.end_run()

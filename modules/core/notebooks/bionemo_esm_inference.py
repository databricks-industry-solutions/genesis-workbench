# Databricks notebook source
dbutils.widgets.text("core_catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("core_schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("is_base_model", "true", "Use Base Model?")
dbutils.widgets.text("esm_variant", "650M", "ESM Variant")
dbutils.widgets.text("task_type", "regression", "Task type: Regression or Classification")
dbutils.widgets.text("finetune_run_id", "1", "Finetune Run Id")
dbutils.widgets.text("data_location", "/Volumes/genesis_workbench/dev_srijit_nair_dbx_genesis_workbench_core/esm_finetune", "Training data location")
dbutils.widgets.text("result_table", "/Volumes/genesis_workbench/dev_srijit_nair_dbx_genesis_workbench_core/esm_finetune", "Result table")
dbutils.widgets.text("user_email", "a@b.com", "User Email")

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
result_table = dbutils.widgets.get("result_table")
user_email = dbutils.widgets.get("user_email")

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

db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().extraContext().apply('api_url')
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Copy data locally

# COMMAND ----------

w = WorkspaceClient(host=db_host, token=databricks_token)

def download_data(remote_path, local_file_name):
  file_content = ""
  sequences = []
  for f in w.files.list_directory_contents(remote_path):
    print(f"Downloading {f.path}")
    with w.files.download(f.path).contents as remote_file:
      file_content += remote_file.read().decode("utf-8")

  for seq in file_content.split("\n"):
    seq = seq.strip()
    if len(seq)>0:
      sequences.append((seq.split(",")[0], seq.split(",")[1]))

  # Create a DataFrame
  df = pd.DataFrame(sequences, columns=["sequences", "labels"])

  # Save the DataFrame to a CSV file
  print(f"Writing to {local_file_name}")
  df.to_csv(local_file_name, index=False)


# COMMAND ----------

workdir_data_file = f"{data_dir}/data.csv"

download_data(data_location, workdir_data_file)


# COMMAND ----------

is_inference_success = False

# COMMAND ----------

model_weights_location

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

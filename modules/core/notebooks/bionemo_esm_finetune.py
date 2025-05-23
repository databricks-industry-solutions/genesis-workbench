# Databricks notebook source
# MAGIC %md
# MAGIC ### Import Required Libraries

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.53.0
# MAGIC %restart_python

# COMMAND ----------

import os
import shutil
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### Work Directory

# COMMAND ----------

cleanup: bool = True
work_dir = "/workdir"

dbutils.widgets.text("esm_variant", "650M", "ESM Variant")
dbutils.widgets.text("train_data_location", "/Volumes/genesis_workbench/dev_srijit_nair_dbx_genesis_workbench_core/esm_finetune", "Training data location")
dbutils.widgets.text("validation_data_location", "/Volumes/genesis_workbench/dev_srijit_nair_dbx_genesis_workbench_core/esm_finetune", "Validation data location")
dbutils.widgets.text("should_use_lora", "false", "Should use LORA")
dbutils.widgets.text("finetune_label", "esm_650m_ft_xyz", "A label using which these finetune weights are saved")
dbutils.widgets.text("core_catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("core_schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")

dbutils.widgets.text("task_type", "regression", "Task type")
dbutils.widgets.text("mlp_ft_dropout", "0.25" , "Dropout")
dbutils.widgets.text("mlp_hidden_size", "256", "Hidden size")
dbutils.widgets.text("mlp_target_size", "1" , "Target size")
dbutils.widgets.text("experiment_name", "sequence_level_regression" , "Experiment name")
dbutils.widgets.text("num_steps", "50", "Num steps")
dbutils.widgets.text("lr", "5e-3","Learning rate")
dbutils.widgets.text("lr_multiplier", "1e2" , "Learning rate multiplier")
#dbutils.widgets.text("scale_lr_layer", "regression_head" ,"Layers to scale Learning Rate")
dbutils.widgets.text("micro_batch_size", "2" , "Micro batch size")
dbutils.widgets.text("precision", "bf16-mixed", "Precision")

catalog = dbutils.widgets.get("core_catalog")
schema = dbutils.widgets.get("core_schema")
esm_variant = dbutils.widgets.get("esm_variant")
train_data_volume_location = dbutils.widgets.get("train_data_location")
validation_data_volume_location = dbutils.widgets.get("validation_data_location")
should_use_lora = True if dbutils.widgets.get("should_use_lora")=="true" else False
finetune_label = dbutils.widgets.get("finetune_label")
task_type = dbutils.widgets.get("task_type")
mlp_ft_dropout = float(dbutils.widgets.get("mlp_ft_dropout"))
mlp_hidden_size = int(dbutils.widgets.get("mlp_hidden_size"))
mlp_target_size = int(dbutils.widgets.get("mlp_target_size"))
experiment_name = dbutils.widgets.get("experiment_name")
num_steps = int(dbutils.widgets.get("num_steps"))
lr = float(dbutils.widgets.get("lr"))
lr_multiplier = float(dbutils.widgets.get("lr_multiplier"))
scale_lr_layer = "regression_head" if task_type=="regression" else "classification_head" #dbutils.widgets.get("scale-lr-layer")
micro_batch_size = int(dbutils.widgets.get("micro_batch_size"))
precision = dbutils.widgets.get("precision")
                                  

# COMMAND ----------

if os.path.exists(work_dir):
    shutil.rmtree(work_dir)

os.makedirs(work_dir)
os.makedirs(work_dir + "/data/train")
os.makedirs(work_dir + "/data/val")
os.makedirs(work_dir + "/ft_weights")
ft_weights_directory = f"{work_dir}/ft_weights"
ft_weights_volume_location = f"/Volumes/{catalog}/{schema}/model_weights/esm2/{esm_variant}/{finetune_label}"
dbutils.fs.rm(ft_weights_volume_location, True)
dbutils.fs.mkdirs(ft_weights_volume_location)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Download Pre-trained Model Checkpoints

# COMMAND ----------

from bionemo.core.data.load import load

pretrain_checkpoint_path = load(f"esm2/{esm_variant.lower()}:2.0")
print(pretrain_checkpoint_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-tuning

# COMMAND ----------

import os
import io
from databricks.sdk import WorkspaceClient

db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().extraContext().apply('api_url')
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Copy training and validation data locally

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
  data_path = os.path.join(work_dir, local_file_name)
  print(f"Writing to {data_path}")
  df.to_csv(data_path, index=False)


# COMMAND ----------

workdir_train_data_file = f"{work_dir}/data/train/train.csv"
workdir_val_data_file = f"{work_dir}/data/val/val.csv"

download_data(train_data_volume_location, workdir_train_data_file)
download_data(validation_data_volume_location, workdir_val_data_file)

# COMMAND ----------

! finetune_esm2 \
    --restore-from-checkpoint-path {pretrain_checkpoint_path} \
    --train-data-path {workdir_train_data_file} \
    --valid-data-path {workdir_val_data_file} \
    --config-class ESM2FineTuneSeqConfig \
    --dataset-class InMemorySingleValueDataset \
    --task-type {task_type} \
    --mlp-ft-dropout {mlp_ft_dropout} \
    --mlp-hidden-size {mlp_hidden_size} \
    --mlp-target-size  {mlp_target_size} \
    --experiment-name {experiment_name} \
    --num-steps {num_steps} \
    --num-gpus 1 \
    --val-check-interval 10 \
    --log-every-n-steps 10 \
    --encoder-frozen \
    --lr {lr} \
    --lr-multiplier {lr_multiplier} \
    --scale-lr-layer {scale_lr_layer} \
    --result-dir {ft_weights_directory}  \
    --micro-batch-size {micro_batch_size} \
    --precision {precision}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Copy the weights back into volume


# COMMAND ----------

ft_weights_volume_location = f"/Volumes/{catalog}/{schema}/model_weights/esm2/{esm_variant}/{finetune_label}"

checkpoint_path = f"{ft_weights_directory}/{experiment_name}/dev/checkpoints"

for file in os.listdir(checkpoint_path):  
  if os.path.isdir(f"{checkpoint_path}/{file}") and file.endswith("-last"):
    last_checkpoint_path = f"file://{checkpoint_path}/{file}"
    print(f"Last checkpoint path is {last_checkpoint_path}")
    print(f"Copying last checkpointed weights to {ft_weights_volume_location}")
    dbutils.fs.cp(last_checkpoint_path, ft_weights_volume_location, True) 
    print("Done")
    

# COMMAND ----------


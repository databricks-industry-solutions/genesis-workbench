# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 2: Train a New Chemprop Toxicity Classifier on ClinTox
# MAGIC %md
# MAGIC %md
# MAGIC This notebook trains a new Chemprop model (Directed Message Passing Neural Network)
# MAGIC %md
# MAGIC for toxicity classification on the ClinTox dataset.
# MAGIC %md
# MAGIC %md
# MAGIC Sometimes existing models are inadequate — they may be nonexistent for a particular
# MAGIC %md
# MAGIC molecular property or may not be applicable to the chemical space of interest.
# MAGIC %md
# MAGIC This example shows how to train a custom binary classification model.
# MAGIC %md
# MAGIC %md
# MAGIC ClinTox contains 1,491 drugs labeled for clinical trial toxicity (Wu et al.).
# MAGIC %md
# MAGIC We train a binary classification D-MPNN model to predict toxicity from SMILES.

# COMMAND ----------

dbutils.widgets.text("catalog", "srijit_nair_ci_demo_catalog", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("model_name", "chemprop_clintox", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "045df48d4afed522", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "chemprop_clintox", "Cache dir")
dbutils.widgets.text("workload_type", "GPU_SMALL", "Workload Type for endpoints")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installing dependencies

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.2
# MAGIC %pip install chemprop>=2.0.0 rdkit torch>=2.0 lightning

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
model_name = dbutils.widgets.get("model_name")
experiment_name = dbutils.widgets.get("experiment_name")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
cache_dir = dbutils.widgets.get("cache_dir")
workload_type = dbutils.widgets.get("workload_type")

print(f"Cache dir: {cache_dir}")
cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"
print(f"Cache full path: {cache_full_path}")

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{cache_dir}")

# COMMAND ----------

# Initialize Genesis Workbench
from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train a Chemprop toxicity classifier on the ClinTox dataset

# COMMAND ----------

import os
import pandas as pd
import numpy as np
import torch
import mlflow

from chemprop import data as chemprop_data
from chemprop import models as chemprop_models
from chemprop import nn as chemprop_nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

import lightning as L

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load ClinTox dataset

# COMMAND ----------

clintox_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
clintox_df = pd.read_csv(clintox_url)
clintox_df = clintox_df[["smiles", "CT_TOX"]].dropna()

print(f"ClinTox dataset size: {clintox_df.shape[0]} molecules")
clintox_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prepare train/val/test splits and train the model

# COMMAND ----------

from sklearn.model_selection import train_test_split
from rdkit import Chem

smiles_list = clintox_df["smiles"].tolist()
targets_list = clintox_df["CT_TOX"].tolist()

# 80/10/10 split
smiles_train, smiles_temp, y_train, y_temp = train_test_split(
    smiles_list, targets_list, test_size=0.2, random_state=42, stratify=targets_list
)
smiles_val, smiles_test, y_val, y_test = train_test_split(
    smiles_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

train_datapoints = [MoleculeDatapoint(mol, [y]) for smi, y in zip(smiles_train, y_train) if (mol := Chem.MolFromSmiles(smi)) is not None]
val_datapoints = [MoleculeDatapoint(mol, [y]) for smi, y in zip(smiles_val, y_val) if (mol := Chem.MolFromSmiles(smi)) is not None]

train_dset = MoleculeDataset(train_datapoints)
val_dset = MoleculeDataset(val_datapoints)

train_loader = build_dataloader(train_dset, shuffle=True)
val_loader = build_dataloader(val_dset, shuffle=False)

# COMMAND ----------

# Build the Chemprop MPNN with binary classification FFN
mp = chemprop_nn.BondMessagePassing()
agg = chemprop_nn.MeanAggregation()
ffn = chemprop_nn.BinaryClassificationFFN()

mpnn_model = chemprop_models.MPNN(message_passing=mp, agg=agg, predictor=ffn)

# COMMAND ----------

# Train using PyTorch Lightning
trainer = L.Trainer(
    max_epochs=30,
    accelerator="auto",
    enable_progress_bar=True,
    logger=False,
)
trainer.fit(mpnn_model, train_loader, val_loader)

# COMMAND ----------

# Save model checkpoint to cache volume
checkpoint_dir = os.path.join(cache_full_path, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "chemprop_clintox.ckpt")
trainer.save_checkpoint(checkpoint_path)
print(f"Checkpoint saved to: {checkpoint_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wrap model in MLflow PyFunc for serving

# COMMAND ----------

class ChempropToxicityModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for a trained Chemprop MPNN binary classification model.
    Accepts SMILES strings and returns toxicity probability predictions.
    """

    def load_context(self, context):
        import torch
        from chemprop import nn as chemprop_nn
        from chemprop import models as chemprop_models

        checkpoint_path = context.artifacts["checkpoint"]

        mp = chemprop_nn.BondMessagePassing()
        agg = chemprop_nn.MeanAggregation()
        ffn = chemprop_nn.BinaryClassificationFFN()
        self.model = chemprop_models.MPNN.load_from_checkpoint(
            checkpoint_path,
            message_passing=mp,
            agg=agg,
            predictor=ffn,
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def predict(self, context, model_input, params=None):
        import torch
        import numpy as np
        from rdkit import Chem
        from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

        if hasattr(model_input, "tolist"):
            smiles_list = model_input.tolist()
        elif hasattr(model_input, "iloc"):
            smiles_list = model_input.iloc[:, 0].tolist()
        else:
            smiles_list = list(model_input)

        mols = [(smi, Chem.MolFromSmiles(smi)) for smi in smiles_list]
        valid_indices = [i for i, (_, mol) in enumerate(mols) if mol is not None]
        datapoints = [MoleculeDatapoint(mols[i][1]) for i in valid_indices]
        dset = MoleculeDataset(datapoints)
        loader = build_dataloader(dset, shuffle=False)

        device = next(self.model.parameters()).device
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch.bmg.to(device)
                V_d = batch.V_d.to(device) if batch.V_d is not None else None
                X_d = batch.X_d.to(device) if batch.X_d is not None else None
                batch_preds = self.model(batch.bmg, V_d, X_d)
                preds.append(batch_preds.cpu().numpy())

        valid_preds = np.concatenate(preds, axis=0).flatten() if preds else np.array([])
        predictions = [None] * len(smiles_list)
        for idx, pred in zip(valid_indices, valid_preds):
            predictions[idx] = float(pred)
        return predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the model locally before registration

# COMMAND ----------

chemprop_pyfunc = ChempropToxicityModel()

# COMMAND ----------

test_smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",       # Aspirin
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",       # Pyrene
]

# COMMAND ----------

from mlflow.types.schema import ColSpec, Schema

signature = mlflow.models.signature.ModelSignature(
    inputs=Schema([ColSpec(type="string")]),
    outputs=Schema([ColSpec(type="double")]),
    params=None,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register model in Unity Catalog

# COMMAND ----------

from genesis_workbench.models import (ModelCategory,
                                      import_model_from_uc,
                                      get_latest_model_version,
                                      deploy_model,
                                      set_mlflow_experiment)

from genesis_workbench.workbench import wait_for_job_run_completion

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

experiment = set_mlflow_experiment(experiment_tag=experiment_name,
                                   user_email=user_email,
                                   host=None,
                                   token=None,
                                   shared=True)

with mlflow.start_run(run_name=f"{model_name}", experiment_id=experiment.experiment_id):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="chemprop_clintox",
        python_model=chemprop_pyfunc,
        artifacts={
            "checkpoint": checkpoint_path,
        },
        pip_requirements=[
            "mlflow>=2.15",
            "chemprop>=2.0.0",
            "rdkit",
            "torch>=2.0",
            "lightning",
            "numpy",
            "pandas",
        ],
        input_example=test_smiles,
        signature=signature,
        registered_model_name=f"{catalog}.{schema}.{model_name}",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import model into Genesis Workbench and deploy serving endpoint

# COMMAND ----------

model_uc_name = f"{catalog}.{schema}.{model_name}"
model_version = get_latest_model_version(model_uc_name)

gwb_model_id = import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.SMALL_MOLECULES,
                    model_uc_name=model_uc_name,
                    model_uc_version=model_version,
                    model_name="Chemprop ClinTox",
                    model_display_name="Chemprop ClinTox Toxicity Classifier",
                    model_source_version="v2.0",
                    model_description_url="https://github.com/chemprop/chemprop")

# COMMAND ----------

run_id = deploy_model(user_email=user_email,
                gwb_model_id=gwb_model_id,
                deployment_name=f"Chemprop ClinTox",
                deployment_description="Chemprop D-MPNN toxicity classifier trained on ClinTox",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type=workload_type,
                workload_size="Small")

# COMMAND ----------

result = wait_for_job_run_completion(run_id, timeout=3600)

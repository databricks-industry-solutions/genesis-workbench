# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 4: Multi-task ADMET Regression with Chemprop
# MAGIC %md
# MAGIC %md
# MAGIC For a compound to be a good drug candidate it must possess several desirable
# MAGIC %md
# MAGIC ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties.
# MAGIC %md
# MAGIC %md
# MAGIC Multi-task training is advantageous because the predicted properties are highly
# MAGIC %md
# MAGIC correlated and joint data analysis allows knowledge from one task to improve another.
# MAGIC %md
# MAGIC Chemprop automatically handles missing entries by masking them in the loss function,
# MAGIC %md
# MAGIC so partial data can be utilized.
# MAGIC %md
# MAGIC %md
# MAGIC This notebook trains a multi-task regression D-MPNN model on 10 ADMET properties
# MAGIC %md
# MAGIC from the DrugBank/TDC datasets, registers it in Unity Catalog, and deploys it.

# COMMAND ----------

dbutils.widgets.text("catalog", "srijit_nair_ci_demo_catalog", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("model_name", "chemprop_admet", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "045df48d4afed522", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "chemprop_admet", "Cache dir")
dbutils.widgets.text("workload_type", "GPU_SMALL", "Workload Type for endpoints")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installing dependencies

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3
# MAGIC %pip install chemprop==2.2.3 rdkit==2025.3.6 torch==2.7.1 torchvision==0.22.1 lightning==2.6.1 PyTDC==1.1.15 scikit-learn==1.8.0
# MAGIC %pip install mlflow[databricks]==2.22.0

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
# MAGIC ### Load ADMET datasets from the Therapeutics Data Commons (TDC)
# MAGIC %md
# MAGIC %md
# MAGIC We load multiple ADMET properties and merge them into a single multi-task DataFrame.
# MAGIC %md
# MAGIC Missing values are handled automatically by Chemprop during training.

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

from tdc.single_pred import ADME, Tox

admet_datasets = {
    "Caco2_Wang": ADME(name="Caco2_Wang"),
    "Lipophilicity_AstraZeneca": ADME(name="Lipophilicity_AstraZeneca"),
    "Solubility_AqSolDB": ADME(name="Solubility_AqSolDB"),
    "HydrationFreeEnergy_FreeSolv": ADME(name="HydrationFreeEnergy_FreeSolv"),
    "PPBR_AZ": ADME(name="PPBR_AZ"),
    "VDss_Lombardo": ADME(name="VDss_Lombardo"),
    "Half_Life_Obach": ADME(name="Half_Life_Obach"),
    "Clearance_Hepatocyte_AZ": ADME(name="Clearance_Hepatocyte_AZ"),
    "LD50_Zhu": Tox(name="LD50_Zhu"),
    "hERG": Tox(name="hERG"),
}

# Build a merged multi-task DataFrame keyed by SMILES
merged_df = None
target_columns = []

for prop_name, dataset in admet_datasets.items():
    df = dataset.get_data()
    df = df[["Drug", "Y"]].rename(columns={"Drug": "smiles", "Y": prop_name})
    target_columns.append(prop_name)

    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on="smiles", how="outer")

print(f"Multi-task dataset: {merged_df.shape[0]} unique molecules, {len(target_columns)} ADMET targets")
print(f"Targets: {target_columns}")
merged_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prepare multi-task datapoints, split, and train

# COMMAND ----------

from sklearn.model_selection import train_test_split
from rdkit import Chem

smiles_all = merged_df["smiles"].tolist()
targets_all = merged_df[target_columns].values  # shape: (n_molecules, n_tasks), may contain NaN

# 80/10/10 split
indices = list(range(len(smiles_all)))
idx_train, idx_temp = train_test_split(indices, test_size=0.2, random_state=42)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)

def make_datapoints(idx_list):
    datapoints = []
    for i in idx_list:
        smi = smiles_all[i]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        y = targets_all[i].tolist()
        # Replace NaN with None so Chemprop masks them in the loss
        y = [val if not np.isnan(val) else None for val in y]
        datapoints.append(MoleculeDatapoint(mol, y))
    return datapoints

train_datapoints = make_datapoints(idx_train)
val_datapoints = make_datapoints(idx_val)

train_dset = MoleculeDataset(train_datapoints)
val_dset = MoleculeDataset(val_datapoints)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Normalize targets and build dataloaders

# COMMAND ----------

output_scaler = train_dset.normalize_targets()
val_dset.normalize_targets(output_scaler)

train_loader = build_dataloader(train_dset, shuffle=True)
val_loader = build_dataloader(val_dset, shuffle=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build the multi-task regression model

# COMMAND ----------

output_transform = chemprop_nn.transforms.UnscaleTransform.from_standard_scaler(output_scaler)

mp = chemprop_nn.BondMessagePassing()
agg = chemprop_nn.MeanAggregation()
ffn = chemprop_nn.RegressionFFN(n_tasks=len(target_columns), output_transform=output_transform)

mpnn_model = chemprop_models.MPNN(message_passing=mp, agg=agg, predictor=ffn)

print(f"Model parameters: {sum(p.numel() for p in mpnn_model.parameters()):,}")

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
checkpoint_path = os.path.join(checkpoint_dir, "chemprop_admet_multitask.ckpt")
trainer.save_checkpoint(checkpoint_path)
print(f"Checkpoint saved to: {checkpoint_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wrap model in MLflow PyFunc for serving

# COMMAND ----------

class ChempropADMETModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for a multi-task Chemprop MPNN regression model.
    Accepts SMILES strings and returns predictions for multiple ADMET properties.
    """

    def load_context(self, context):
        import json
        import pickle
        import torch
        from chemprop import nn as chemprop_nn
        from chemprop import models as chemprop_models

        with open(context.artifacts["target_columns"], "r") as f:
            self.target_columns = json.load(f)

        with open(context.artifacts["output_scaler"], "rb") as f:
            output_scaler = pickle.load(f)

        output_transform = chemprop_nn.transforms.UnscaleTransform.from_standard_scaler(output_scaler)

        mp = chemprop_nn.BondMessagePassing()
        agg = chemprop_nn.MeanAggregation()
        ffn = chemprop_nn.RegressionFFN(
            n_tasks=len(self.target_columns),
            output_transform=output_transform,
        )
        self.model = chemprop_models.MPNN.load_from_checkpoint(
            context.artifacts["checkpoint"],
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
        import pandas as pd
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

        valid_preds = np.concatenate(preds, axis=0) if preds else np.empty((0, len(self.target_columns)))
        result = pd.DataFrame(
            np.full((len(smiles_list), len(self.target_columns)), np.nan),
            columns=self.target_columns,
        )
        for row_idx, pred_row in zip(valid_indices, valid_preds):
            result.iloc[row_idx] = pred_row
        return result

# COMMAND ----------

class ChempropADMETModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for a multi-task Chemprop MPNN regression model.
    Accepts SMILES strings and returns predictions for multiple ADMET properties.
    """

    def load_context(self, context):
        import json
        import pickle
        import torch
        from chemprop import nn as chemprop_nn
        from chemprop import models as chemprop_models

        with open(context.artifacts["target_columns"], "r") as f:
            self.target_columns = json.load(f)

        with open(context.artifacts["output_scaler"], "rb") as f:
            output_scaler = pickle.load(f)

        output_transform = chemprop_nn.transforms.UnscaleTransform.from_standard_scaler(output_scaler)

        mp = chemprop_nn.BondMessagePassing()
        agg = chemprop_nn.MeanAggregation()
        ffn = chemprop_nn.RegressionFFN(
            n_tasks=len(self.target_columns),
            output_transform=output_transform,
        )
        self.model = chemprop_models.MPNN.load_from_checkpoint(
            context.artifacts["checkpoint"],
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
        import pandas as pd
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

        valid_preds = np.concatenate(preds, axis=0) if preds else np.empty((0, len(self.target_columns)))
        result = pd.DataFrame(
            np.full((len(smiles_list), len(self.target_columns)), np.nan),
            columns=self.target_columns,
        )
        for row_idx, pred_row in zip(valid_indices, valid_preds):
            result.iloc[row_idx] = pred_row
        return result

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the model locally before registration

# COMMAND ----------

chemprop_pyfunc = ChempropADMETModel()

# COMMAND ----------

test_smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",       # Aspirin
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",       # Pyrene
]

# COMMAND ----------

from mlflow.types.schema import ColSpec, Schema

input_schema = Schema([ColSpec(type="string")])
output_schema = Schema([ColSpec(type="double", name=col) for col in target_columns])

signature = mlflow.models.signature.ModelSignature(
    inputs=input_schema,
    outputs=output_schema,
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

import mlflow

# COMMAND ----------

import json
import pickle

# Save target_columns and output_scaler to disk so they can be logged as MLflow artifacts
target_columns_path = os.path.join(cache_full_path, "target_columns.json")
with open(target_columns_path, "w") as f:
    json.dump(target_columns, f)

scaler_path = os.path.join(cache_full_path, "output_scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(output_scaler, f)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

experiment = set_mlflow_experiment(experiment_tag=experiment_name,
                                   user_email=user_email,
                                   host=None,
                                   token=None,
                                   shared=True)

with mlflow.start_run(run_name=f"{model_name}", experiment_id=experiment.experiment_id):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="chemprop_admet",
        python_model=chemprop_pyfunc,
        artifacts={
            "checkpoint": checkpoint_path,
            "target_columns": target_columns_path,
            "output_scaler": scaler_path,
        },
        pip_requirements=[
            "chemprop==2.2.3",
            "rdkit==2025.3.6",
            "torch==2.7.1",
            "torchvision==0.22.1",
            "lightning==2.6.1",
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
                    model_category=ModelCategory.SMALL_MOLECULE,
                    model_uc_name=model_uc_name,
                    model_uc_version=model_version,
                    model_name="Chemprop ADMET",
                    model_display_name="Chemprop Multi-task ADMET Regressor",
                    model_source_version="v2.0",
                    model_description_url="https://github.com/chemprop/chemprop")

# COMMAND ----------

run_id = deploy_model(user_email=user_email,
                gwb_model_id=gwb_model_id,
                deployment_name=f"Chemprop ADMET",
                deployment_description="Chemprop D-MPNN multi-task regression model for 10 ADMET properties",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type=workload_type,
                workload_size="Small")

# COMMAND ----------

result = wait_for_job_run_completion(run_id, timeout=3600)

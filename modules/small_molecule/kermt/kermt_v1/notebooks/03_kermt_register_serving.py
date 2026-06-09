# Databricks notebook source
# MAGIC %md
# MAGIC # KERMT — Register fine-tuned model + deploy serving endpoint
# MAGIC Takes a fine-tuned `ft_id` from `kermt_weights`, wraps the checkpoint in an MLflow PyFunc
# MAGIC that runs KERMT prediction **in-process on the plain-RDKit path** (no cuik_molmaker, so the
# MAGIC Model Serving env stays buildable), registers it to Unity Catalog, imports it into GWB, and
# MAGIC deploys a GPU serving endpoint. Mirrors the ChemProp register/deploy pattern.
# MAGIC
# MAGIC Endpoint contract matches ChemProp ADMET: `inputs=[smiles…]` → `predictions=[{task: val}…]`.

# COMMAND ----------

dbutils.widgets.text("catalog", "srijit_nair_ci_demo_catalog", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("cache_dir", "kermt", "KERMT UC volume name")
dbutils.widgets.text("kermt_src_path", "", "Workspace path to vendored kermt_src")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "User Id/Email")
dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse Id")
dbutils.widgets.text("ft_id", "", "Fine-tuned model id (from kermt_weights) to deploy")
dbutils.widgets.text("model_name", "kermt_admet", "UC model name")
dbutils.widgets.text("workload_type", "GPU_SMALL", "Serving workload type")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %pip install -q torch==2.3.1 rdkit==2025.3.6 descriptastorus==2.8.0 scikit-learn==1.5.2 numpy==1.26.4 pandas==1.5.3 tqdm==4.67.1
# MAGIC %pip install -q mlflow[databricks]==2.22.0 databricks-sdk==0.50.0 databricks-sql-connector==4.0.3

# COMMAND ----------

gwb_library_path = None
for lib in dbutils.fs.ls(f"/Volumes/{dbutils.widgets.get('catalog')}/{dbutils.widgets.get('schema')}/libraries"):
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")
print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, sys, shutil
g = dbutils.widgets.get
catalog, schema, cache_dir = g("catalog"), g("schema"), g("cache_dir")
kermt_src_path, user_email = g("kermt_src_path"), g("user_email")
sql_warehouse_id = g("sql_warehouse_id")
ft_id = g("ft_id")
model_name = g("model_name")
workload_type = g("workload_type")
vol_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"

# Resolve the fine-tuned checkpoint location from kermt_weights.
row = spark.sql(f"""
    SELECT ft_label, dataset_type, task_names, weights_volume_location
    FROM {catalog}.{schema}.kermt_weights WHERE ft_id = {ft_id} AND is_active = true
""").collect()
assert row, f"ft_id {ft_id} not found / inactive in kermt_weights"
ft_label = row[0]["ft_label"]; dataset_type = row[0]["dataset_type"]
task_names = [t for t in row[0]["task_names"].split(",") if t]
weights_loc = row[0]["weights_volume_location"]
print(f"deploying ft_label={ft_label} dataset_type={dataset_type} tasks={task_names}\n  from {weights_loc}")

# Stage vendored source + checkpoint locally.
WS_SRC = "/Workspace" + kermt_src_path
LOCAL_SRC = "/local_disk0/kermt_src"
shutil.rmtree(LOCAL_SRC, ignore_errors=True); shutil.copytree(WS_SRC, LOCAL_SRC)
LOCAL_CKPT = "/local_disk0/kermt_ckpt"
shutil.rmtree(LOCAL_CKPT, ignore_errors=True); shutil.copytree(weights_loc, LOCAL_CKPT)
sys.path.insert(0, LOCAL_SRC)
print("staged source + checkpoint")

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel

class KermtADMETModel(PythonModel):
    """Serves a fine-tuned KERMT model on the plain-RDKit path (no cuik_molmaker).
    Input: SMILES strings. Output: a DataFrame with one column per task -> serving
    returns predictions=[{task: value}, ...] (matches the ChemProp ADMET contract)."""

    def load_context(self, context):
        import sys as _sys, os as _os
        src = context.artifacts["kermt_src"]
        if src not in _sys.path:
            _sys.path.insert(0, src)
        ckpt_dir = context.artifacts["checkpoint_dir"]

        from kermt.util.parsing import parse_args
        from kermt.util.utils import load_args, load_scalars, load_checkpoint_for_prediction
        # Build predict args via KERMT's own parser (walks checkpoint_dir for *.pt).
        _sys.argv = ["kermt", "predict",
                     "--checkpoint_dir", ckpt_dir,
                     "--data_path", "/tmp/_kermt_dummy.csv", "--output", "/tmp/_kermt_out.csv",
                     "--no_features_scaling",
                     "--features_generator", "rdkit_2d_normalized_onthefly",
                     "--batch_size", "16"]
        args = parse_args()
        ckpt0 = args.checkpoint_paths[0]
        scaler, _ = load_scalars(ckpt0)
        train_args = load_args(ckpt0)
        for k, v in vars(train_args).items():
            if not hasattr(args, k):
                setattr(args, k, v)
        args.task_names = train_args.task_names
        args.num_tasks = len(train_args.task_names)
        args.bond_drop_rate = 0
        self._args = args
        self._scaler = scaler if dataset_type == "regression" else None
        self._task_names = list(train_args.task_names)
        self._model = load_checkpoint_for_prediction(ckpt0, cuda=args.cuda, current_args=args, logger=None)
        self._model.eval()

    def predict(self, context, model_input, params=None):
        import pandas as pd
        from kermt.util.utils import get_data_from_smiles
        from task.predict import predict as _kermt_predict
        # Accept a DataFrame (serving maps inputs=[..] to a 'smiles' column) or a raw list.
        if hasattr(model_input, "iloc"):
            col = "smiles" if "smiles" in model_input.columns else model_input.columns[0]
            smiles = model_input[col].astype(str).tolist()
        else:
            smiles = [str(s) for s in model_input]
        data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False, args=self._args)
        preds, _ = _kermt_predict(model=self._model, data=data, args=self._args,
                                  batch_size=self._args.batch_size, loss_func=None,
                                  logger=None, shared_dict={}, scaler=self._scaler)
        return pd.DataFrame(preds, columns=self._task_names)

# COMMAND ----------

# Dry-load + smoke test before registering.
from mlflow.pyfunc import PythonModelContext
artifacts = {"kermt_src": LOCAL_SRC, "checkpoint_dir": LOCAL_CKPT}
_ctx = PythonModelContext(artifacts=artifacts, model_config={})
_m = KermtADMETModel(); _m.load_context(_ctx)
import pandas as pd
_sample = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
_out = _m.predict(_ctx, pd.DataFrame({"smiles": _sample}))
print("smoke test predictions:\n", _out)

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

signature = ModelSignature(
    inputs=Schema([ColSpec("string", "smiles")]),
    outputs=Schema([ColSpec("double", name=t) for t in task_names]),
)

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(run_name=f"register_{model_name}"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path=model_name,
        python_model=KermtADMETModel(),
        artifacts={"kermt_src": LOCAL_SRC, "checkpoint_dir": LOCAL_CKPT},
        code_paths=[f"{LOCAL_SRC}/kermt", f"{LOCAL_SRC}/task"],
        signature=signature,
        input_example=pd.DataFrame({"smiles": _sample}),
        pip_requirements=[
            "torch==2.3.1", "rdkit==2025.3.6", "descriptastorus==2.8.0",
            "scikit-learn==1.5.2", "numpy==1.26.4", "pandas==1.5.3", "tqdm==4.67.1",
            "mlflow==2.22.0",
        ],
        registered_model_name=f"{catalog}.{schema}.{model_name}",
    )
print("registered:", model_info.model_uri)

# COMMAND ----------

# Import into GWB + deploy a serving endpoint (ChemProp pattern).
from genesis_workbench.models import (ModelCategory, import_model_from_uc,
                                      deploy_model, get_latest_model_version)
from genesis_workbench.workbench import initialize, wait_for_job_run_completion

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

model_uc_name = f"{catalog}.{schema}.{model_name}"
model_version = get_latest_model_version(model_uc_name)

gwb_model_id = import_model_from_uc(
    user_email=user_email,
    model_category=ModelCategory.SMALL_MOLECULE,
    model_uc_name=model_uc_name,
    model_uc_version=model_version,
    model_name="KERMT ADMET",
    model_display_name="KERMT ADMET",
    model_source_version=f"ft:{ft_label}",
    model_description_url="https://github.com/NVIDIA-BioNeMo/KERMT",
)

run_id = deploy_model(
    user_email=user_email,
    gwb_model_id=gwb_model_id,
    deployment_name="KERMT ADMET",
    deployment_description=f"KERMT (Kinetic GROVER Multi-Task) fine-tuned on {','.join(task_names)} ({ft_label})",
    input_adapter_str="none",
    output_adapter_str="none",
    sample_input_data_dict_as_json="none",
    sample_params_as_json="none",
    workload_type=workload_type,
    workload_size="Small",
)
result = wait_for_job_run_completion(run_id, timeout=3600)
print("deploy result:", result)

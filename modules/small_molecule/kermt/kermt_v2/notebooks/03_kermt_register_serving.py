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
# Labeled holdout for probability calibration (classification only). Preference order:
# <checkpoint_dir>/calibration.csv (written by the fine-tune job) → this widget → none.
dbutils.widgets.text("calibration_data_location", "", "Calibration holdout CSV (smiles + target); blank = use the checkpoint's calibration.csv")

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
ft_id = g("ft_id").strip()
model_name = g("model_name")
workload_type = g("workload_type")
calibration_data_location = g("calibration_data_location")
vol_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"

# When no ft_id is passed (e.g. the deploy step of the initial submodule deployment,
# which runs right after a finetune), fall back to the most recent active checkpoint.
# ft_id is a monotonic time_ns() value, so ORDER BY ft_id DESC = newest first.
if not ft_id:
    latest = spark.sql(f"""
        SELECT ft_id FROM {catalog}.{schema}.kermt_weights
        WHERE is_active = true ORDER BY ft_id DESC LIMIT 1
    """).collect()
    assert latest, "no active fine-tuned model found in kermt_weights — run a finetune first"
    ft_id = str(latest[0]["ft_id"])
    print(f"ft_id not provided — using latest active ft_id = {ft_id}")

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

        # Probability calibration (classification): a per-task Platt map
        # {task: {"a": ..., "b": ...}} so calibrated = sigmoid(a*logit(p)+b).
        # Stored as plain JSON params (no pickled class) so the serving env needs
        # nothing extra. Empty/absent => raw model probabilities pass through.
        import json as _json
        self._calib = {}
        calib_path = context.artifacts.get("calibration")
        if calib_path and _os.path.exists(calib_path):
            try:
                with open(calib_path) as _f:
                    self._calib = _json.load(_f) or {}
            except Exception:
                self._calib = {}

    @staticmethod
    def _apply_platt(p, a, b):
        import numpy as _np
        p = _np.clip(_np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        logit = _np.log(p / (1.0 - p))
        return 1.0 / (1.0 + _np.exp(-(a * logit + b)))

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
        df = pd.DataFrame(preds, columns=self._task_names)
        # Apply per-task calibration where fitted (classification only).
        for task, cal in (self._calib or {}).items():
            if task in df.columns and cal and "a" in cal and "b" in cal:
                df[task] = self._apply_platt(df[task].values, cal["a"], cal["b"])
        return df

# COMMAND ----------

# Dry-load + smoke test before registering.
from mlflow.pyfunc import PythonModelContext
artifacts = {"kermt_src": LOCAL_SRC, "checkpoint_dir": LOCAL_CKPT}
_ctx = PythonModelContext(artifacts=artifacts, model_config={})
_m = KermtADMETModel(); _m.load_context(_ctx)
import pandas as pd
_sample = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
_out = _m.predict(_ctx, pd.DataFrame({"smiles": _sample}))
print("smoke test predictions (raw, uncalibrated):\n", _out)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Probability calibration (classification only)
# MAGIC Fit a per-task **Platt** map on a labeled holdout (the checkpoint's `calibration.csv`,
# MAGIC written by the fine-tune job, else `calibration_data_location`) so the served
# MAGIC probabilities are comparable to ChemProp's — instead of the raw, miscalibrated sigmoid.
# MAGIC The map is stored as plain JSON params and applied inside the PyFunc.

# COMMAND ----------

import json
import numpy as np
from sklearn.linear_model import LogisticRegression

CALIB_PATH = "/local_disk0/calibration.json"
calib: dict = {}

calib_csv = None
_ckpt_calib = f"{LOCAL_CKPT}/calibration.csv"
if os.path.exists(_ckpt_calib):
    calib_csv = _ckpt_calib
elif calibration_data_location.strip():
    calib_csv = calibration_data_location.strip()

if dataset_type == "classification" and calib_csv:
    print(f"Calibrating on holdout: {calib_csv}")
    try:
        hold = pd.read_csv(calib_csv)
        raw = _m.predict(_ctx, pd.DataFrame({"smiles": hold["smiles"].astype(str).tolist()}))
        for task in task_names:
            if task not in hold.columns:
                print(f"  {task}: not in holdout columns — skip")
                continue
            y = pd.to_numeric(hold[task], errors="coerce").to_numpy()
            p = pd.to_numeric(raw[task], errors="coerce").to_numpy()
            mask = ~(np.isnan(y) | np.isnan(p))
            y, p = y[mask], p[mask]
            n_pos = int((y == 1).sum())
            if len(y) < 20 or n_pos < 5 or n_pos == len(y):
                print(f"  {task}: too few/too-imbalanced ({len(y)} pts, {n_pos} pos) — skip calibration")
                continue
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            x = np.log(pc / (1 - pc)).reshape(-1, 1)
            lr = LogisticRegression(solver="lbfgs").fit(x, y)  # regularized Platt (robust on small data)
            calib[task] = {
                "a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0]),
                "n": int(len(y)), "n_pos": n_pos,
            }
            print(f"  {task}: Platt a={calib[task]['a']:.3f} b={calib[task]['b']:.3f} "
                  f"(n={len(y)}, pos={n_pos})")
    except Exception as e:
        print("Calibration skipped (error):", e)
else:
    print(f"No calibration (dataset_type={dataset_type}, holdout={calib_csv}). Serving raw probabilities.")

with open(CALIB_PATH, "w") as _f:
    json.dump(calib, _f)
artifacts["calibration"] = CALIB_PATH

# Verify the calibrated PyFunc loads + applies the map.
_m2 = KermtADMETModel()
_m2.load_context(PythonModelContext(artifacts=artifacts, model_config={}))
print("smoke test predictions (calibrated):\n", _m2.predict(None, pd.DataFrame({"smiles": _sample})))

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
        artifacts=artifacts,  # kermt_src + checkpoint_dir + calibration (Platt JSON)
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

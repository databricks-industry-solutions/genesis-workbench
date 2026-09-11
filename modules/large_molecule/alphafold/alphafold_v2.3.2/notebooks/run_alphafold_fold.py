# Databricks notebook source
# MAGIC %md
# MAGIC # AlphaFold fold — serverless GPU (JAX, no conda)
# MAGIC Runs the AlphaFold-2.3.2 structure models on **serverless GPU (`GPU_1xA10`)**
# MAGIC with a modern pip JAX stack (`jax[cuda12]==0.4.28`), instead of the classic
# MAGIC Miniconda/`conda env`/`jaxlib-cuda11` bootstrap. Loads the `features.pkl`
# MAGIC produced by the featurize task, runs the **template-free** models, ranks them
# MAGIC by pLDDT, and writes `ranked_0.pdb` (+ ranked_1..) to
# MAGIC `/Volumes/{catalog}/{schema}/{model_volume}/results/{run_id}/{run_id}/`
# MAGIC — the path the GWB app pulls.
# MAGIC
# MAGIC Only template-free models (model_3/4/5) are supported here; templates would
# MAGIC need hhsearch/pdb70 which the serverless-CPU featurize skips. OpenMM relax is
# MAGIC not run — `ranked_0.pdb` is the unrelaxed top model.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("model_volume", "alphafold", "Volume")
dbutils.widgets.text("run_id", "b3c99d3b49ba4893aa402a4342a70cd1", "Run Id")
dbutils.widgets.text("protein_sequence", "", "Protein Sequence")  # unused (features carry it)
dbutils.widgets.text("user_email", "a@b.com", "User Email")
dbutils.widgets.text("fold_models", "model_3_ptm,model_4_ptm,model_5_ptm", "Template-free models")
dbutils.widgets.text("num_recycle", "3", "Recycles")

# COMMAND ----------

# torch/CUDA are preinstalled but AF uses JAX; install jax[cuda12] + AF's deps +
# tensorflow-CPU (feature processing only; JAX owns the GPU). No conda/Miniconda.
# MAGIC %pip install -q "jax[cuda12]==0.4.28" dm-haiku==0.0.12 chex dm-tree ml-collections immutabledict absl-py biopython "tensorflow-cpu==2.18.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, sys, subprocess, tempfile, time, pickle
import numpy as np

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("model_volume")
RUN_ID = dbutils.widgets.get("run_id")
USER_EMAIL = dbutils.widgets.get("user_email")
NUM_RECYCLE = int(dbutils.widgets.get("num_recycle"))
FOLD_MODELS = [m.strip() for m in dbutils.widgets.get("fold_models").split(",") if m.strip()]
# guard: only template-free models are valid for the serverless (no-template) features
FOLD_MODELS = [m for m in FOLD_MODELS if any(m.startswith(f"model_{i}") for i in (3, 4, 5))]
assert FOLD_MODELS, "no template-free models (model_3/4/5) selected"

DATA_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/datasets"
OUTDIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/results/{RUN_ID}/{RUN_ID}"
FEATURES = os.path.join(OUTDIR, "features.pkl")
print("run_id:", RUN_ID, "| models:", FOLD_MODELS, "| recycles:", NUM_RECYCLE)
print("features:", FEATURES)

# COMMAND ----------

# AF source + Biopython SCOPData shim (see featurize notebook for rationale)
import types, Bio.Data
try:
    from Bio.Data import SCOPData  # noqa: F401
except ImportError:
    from Bio.Data import PDBData
    _shim = types.ModuleType("Bio.Data.SCOPData")
    _shim.protein_letters_3to1 = dict(PDBData.protein_letters_3to1_extended)
    sys.modules["Bio.Data.SCOPData"] = _shim
    Bio.Data.SCOPData = _shim

_af = tempfile.mkdtemp(prefix="af2_")
_repo = os.path.join(_af, "alphafold")
subprocess.run(["git", "clone", "--depth", "1", "--branch", "v2.3.2",
                "https://github.com/google-deepmind/alphafold.git", _repo],
               check=True, capture_output=True, text=True)
sys.path.insert(0, _repo)

import jax
from alphafold.model import config, data, model
from alphafold.common import protein, residue_constants
print("jax", jax.__version__, "backend", jax.default_backend(), jax.devices())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load features, run each model, rank by pLDDT

# COMMAND ----------

with open(FEATURES, "rb") as f:
    feature_dict = pickle.load(f)
print("loaded features; seq len", int(feature_dict["seq_length"][0]),
      "| MSA rows", int(feature_dict["num_alignments"][0]))

results = []  # (mean_plddt, model_name, pdb_str)
for model_name in FOLD_MODELS:
    t0 = time.time()
    cfg = config.model_config(model_name)
    cfg.data.common.num_recycle = NUM_RECYCLE
    cfg.model.num_recycle = NUM_RECYCLE
    cfg.data.eval.num_ensemble = 1
    params = data.get_model_haiku_params(model_name=model_name, data_dir=DATA_DIR)
    runner = model.RunModel(cfg, params)
    proc = runner.process_features(feature_dict, random_seed=0)
    pred = runner.predict(proc, random_seed=0)
    mean_plddt = float(np.mean(pred["plddt"]))
    b = np.repeat(pred["plddt"][:, None], residue_constants.atom_type_num, axis=-1)
    prot = protein.from_prediction(features=proc, result=pred, b_factors=b,
                                   remove_leading_feature_dimension=True)
    results.append((mean_plddt, model_name, protein.to_pdb(prot)))
    print(f"{model_name}: mean_plddt {mean_plddt:.1f} | {time.time()-t0:.0f}s")

# rank by mean pLDDT (desc); ranked_0.pdb = best (what the GWB app pulls)
results.sort(key=lambda r: r[0], reverse=True)
for rank, (plddt, model_name, pdb_str) in enumerate(results):
    with open(os.path.join(OUTDIR, f"ranked_{rank}.pdb"), "w") as f:
        f.write(pdb_str)
    print(f"ranked_{rank}.pdb <- {model_name} (plddt {plddt:.1f})")
print("best model:", results[0][1], "plddt", round(results[0][0], 1))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log fold result (fold_complete is also set by the mark_success task)

# COMMAND ----------

import mlflow

try:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    with mlflow.start_run(run_id=RUN_ID):
        mlflow.log_param("fold_results_path", OUTDIR)
        mlflow.log_param("fold_best_model", results[0][1])
        mlflow.log_metric("fold_best_mean_plddt", results[0][0])
        mlflow.set_tag("job_status", "fold_complete")
    print("fold_complete")
except Exception as e:
    print(f"WARN: could not set fold_complete tag on run {RUN_ID}: {e}")

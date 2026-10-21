# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark — RFdiffusion (RFD3 inpainting) · generation speed + GPU memory (H100)
# MAGIC Loads the registered UC model via MLflow and measures per-sample generation time + peak GPU
# MAGIC memory (torch + NVML) over the shared template set. Native mode = **inpainting**: regenerate the
# MAGIC middle span of the template, keep the flanks. Registered on `client '4'` (CUDA 12.x); here we run
# MAGIC it on `GPU_1xH100` / `client '6'` (CUDA 13) for GPU parity, relying on driver forward-compat.

# COMMAND ----------
dbutils.widgets.text("model_uri", "models:/main.genesis_workbench.rfdiffusion_inpainting/4", "Registered model URI")
dbutils.widgets.text("out_dir", "/Volumes/main/genesis_workbench/rfd4_proteina/benchmark_results", "Results dir (UC Volume)")
dbutils.widgets.text("max_samples", "50", "Number of samples to run")

# COMMAND ----------
# DBTITLE 1,Install RFD3 code deps (atomworks pinned per the register notebook). load_model fetches ckpts.
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--extra-index-url", "https://pypi.org/simple",
                       "atomworks[ml]==2.2.1", "rc-foundry", "biopython",
                       "mlflow>=3.12", "databricks-sdk>=0.150.0"])
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./bench_common

# COMMAND ----------
# MAGIC %run ./make_dataset

# COMMAND ----------
# DBTITLE 1,Load from MLflow + CIF-template measured loop
import mlflow, torch
MODEL = "rfdiffusion_inpainting"
MODEL_URI = dbutils.widgets.get("model_uri")
OUT_DIR = dbutils.widgets.get("out_dir")
MAX = int(dbutils.widgets.get("max_samples"))
print("GPU:", gpu_info())

model = mlflow.pyfunc.load_model(MODEL_URI)

_cif = {}
def cif_for(s):                               # proper mmCIF (atomworks bonds), cached per length
    L = s["total_len"]
    if L not in _cif:
        _cif[L] = open(pdb_to_cif(s["template_pdb"], f"/tmp/tmpl_{L}.cif")).read()
    return _cif[L]

def adapt(s):
    # rfdiffusion's pyfunc parses residue numbers PDB-style (_max_resid chokes on CIF tokens like
    # "' . 1'"), so feed the (column-fixed) PDB directly — NOT CIF. Regenerate [regen_start..regen_end].
    return [{"pdb": s["template_pdb"], "start_idx": int(s["regen_start"]), "end_idx": int(s["regen_end"])}]

specs = make_motif_dataset(n=MAX)
run_benchmark(MODEL, model, specs, adapt, torch, OUT_DIR, extra={"mode": "inpaint_cif", "model_uri": MODEL_URI})

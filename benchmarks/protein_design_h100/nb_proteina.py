# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark — Proteina-Complexa (AME motif-scaffolding) · generation speed + GPU memory (H100)
# MAGIC Loads the registered UC model via MLflow and measures per-sample generation time + peak GPU
# MAGIC memory (torch + NVML) over the shared template set. Native mode = **AME motif-scaffolding**: fix the
# MAGIC template motif, scaffold a region of matched length. Registered on `client '4'` (CUDA 12.x, torch
# MAGIC 2.7.1/cu126); here we run on `GPU_1xH100` / `client '6'` (CUDA 13) for parity via driver forward-compat.

# COMMAND ----------
dbutils.widgets.text("model_uri", "models:/main.genesis_workbench.proteina_complexa/6", "Registered model URI")
dbutils.widgets.text("out_dir", "/Volumes/main/genesis_workbench/rfd4_proteina/benchmark_results", "Results dir (UC Volume)")
dbutils.widgets.text("max_samples", "50", "Number of samples to run")

# COMMAND ----------
# DBTITLE 1,Install Proteina-Complexa code deps (torch 2.7.1/cu126 + repo). load_model fetches ckpts.
import subprocess, sys, os
PYG = "https://data.pyg.org/whl/torch-2.7.0+cu126.html"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "databricks-sdk>=0.150.0", "databricks-sql-connector>=4.0.2", "mlflow>=3.12"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--find-links", PYG,
                       "torch==2.7.1", "lightning==2.6.1", "hydra-core==1.3.1", "omegaconf==2.3.0",
                       "torch_geometric==2.7.0", "torch_scatter==2.1.2", "torch_sparse==0.6.18",
                       "torch_cluster==1.6.3", "biotite==1.6.0", "loralib==0.1.2", "einops==0.8.2",
                       "transformers==5.5.0", "jaxtyping"])
_dir, _commit = "/tmp/proteina_complexa_repo", "f95f2d4bbcebcad0613b89a0012edec8637a6334"
if not os.path.exists(_dir):
    subprocess.check_call(["git", "clone", "https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa.git", _dir])
    subprocess.check_call(["git", "-C", _dir, "checkout", _commit])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", _dir])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--find-links", PYG,
                       "atomworks==1.0.3", "jax==0.9.2", "colabdesign==1.1.1", "jaxtyping==0.3.9", "loguru==0.7.3",
                       "biopandas==0.5.1", "biopython==1.87", "cpdb-protein==0.2.1", "deepdiff==9.0.0", "dm-tree==0.1.10",
                       "h5py==3.16.0", "mdtraj==1.11.1.post1", "ml-collections==1.1.0", "modin==0.37.1",
                       "multipledispatch==1.0.0", "plotly==6.7.0", "prody==2.6.1", "pydantic==2.12.5",
                       "python-dotenv==1.2.2", "rich==14.3.4", "rich-click==1.9.7", "seaborn==0.13.2",
                       "wandb==0.25.1", "wget==3.2", "xarray==2026.2.0", "toolz==1.1.0"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "graphein"])
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./bench_common

# COMMAND ----------
# MAGIC %run ./make_dataset

# COMMAND ----------
# DBTITLE 1,Load from MLflow + CIF-template measured loop
import mlflow, torch, pandas as pd
MODEL = "proteina_complexa"   # binder variant: needs a protein target (our template); AME needs a ligand (x_target)
MODEL_URI = dbutils.widgets.get("model_uri")
OUT_DIR = dbutils.widgets.get("out_dir")
MAX = int(dbutils.widgets.get("max_samples"))
print("GPU:", gpu_info())

model = mlflow.pyfunc.load_model(MODEL_URI)

_cif = {}
def cif_for(s):                               # proper mmCIF (atomworks bonds/charges), cached per length
    L = s["total_len"]
    if L not in _cif:
        _cif[L] = open(pdb_to_cif(s["template_pdb"], f"/tmp/tmpl_{L}.cif")).read()
    return _cif[L]

def adapt(s):
    L = int(s["regen_len"])                   # AME motif-scaffolding, scaffold length matched to the span
    # proteina's AME path writes target_pdb to a .pdb and parses ATOM records (_build_motif_atom_spec),
    # so feed PDB — NOT CIF (CIF content yields "No heavy ATOM records"). It passes the mlflow string
    # schema either way, but the model needs real PDB ATOM lines.
    df = pd.DataFrame([{"target_pdb": s["template_pdb"], "target_chain": "A",
                        "binder_length_min": L, "binder_length_max": L, "num_samples": 1,
                        "hotspot_residues": ""}])   # hotspot_residues required by the registered schema
    # proteina's schema is int32 ('integer'); Python ints become int64 and mlflow won't downcast.
    return df.astype({"binder_length_min": "int32", "binder_length_max": "int32", "num_samples": "int32"})

specs = make_motif_dataset(n=MAX)
run_benchmark(MODEL, model, specs, adapt, torch, OUT_DIR, extra={"mode": "binder", "model_uri": MODEL_URI})

# Databricks notebook source
# MAGIC %md
# MAGIC ## Build Transformer Engine Wheel (serverless GPU)
# MAGIC
# MAGIC Builds the `transformer_engine` / `transformer_engine_torch` wheels from source on a
# MAGIC **serverless GPU** node (`GPU_1xA10`) and copies them to the UC `libraries` Volume, so
# MAGIC downstream serverless-GPU workflows (bionemo ESM2 finetune/inference, esm2_embeddings,
# MAGIC sequence_search) install a prebuilt wheel in seconds instead of source-building (~40 min)
# MAGIC on every run.
# MAGIC
# MAGIC Migrated off the classic A10 GPU cluster (can't provision at the workshop's EC2 GPU quota=0).
# MAGIC The serverless base ships `nvcc` + the CUDA runtime but NOT the CUDA math-lib dev headers
# MAGIC (cusparse.h, nvtx3, …) under /usr/local/cuda/include — those come with the pip `nvidia-*`
# MAGIC wheels (installed as torch deps) at `site-packages/nvidia/<lib>/include/`, so we surface them
# MAGIC on CPATH for the build. `NVTE_CUDA_ARCHS=86` targets A10 (Ampere sm_86; the same arch the
# MAGIC downstream jobs run on, so the wheel is binary-compatible).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("te_version", "", "Transformer Engine version to pin (blank = latest compatible)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
te_version = dbutils.widgets.get("te_version").strip()

libraries_volume = f"/Volumes/{catalog}/{schema}/libraries"
print(f"Libraries volume: {libraries_volume}")

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

# MAGIC %md
# MAGIC ### Skip if a wheel already exists

# COMMAND ----------

import os, sys, glob, shutil, subprocess

existing = [f for f in os.listdir(libraries_volume)
            if f.startswith("transformer_engine") and f.endswith(".whl")]
if existing:
    print(f"Transformer Engine wheel(s) already present: {existing}")
    print("Skipping build. Delete them from the volume to force a rebuild.")
    dbutils.notebook.exit(f"SKIPPED: {existing}")
print("No existing wheel found. Building from source...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Surface CUDA dev headers + set build env

# COMMAND ----------

def cuda_include_cpath():
    """':'-joined CUDA dev-header include dirs from the pip nvidia-* wheels."""
    roots = []
    try:
        import nvidia
        roots.append(os.path.dirname(nvidia.__file__))
    except Exception as e:
        print(f"[warn] import nvidia failed: {e}")
    incs = sorted({d for r in roots for d in glob.glob(os.path.join(r, "*", "include")) if os.path.isdir(d)})
    for d in incs:
        print("  include:", d)
    # sanity: the two headers TE's build tends to choke on
    for hdr in ("cusparse.h", "nvtx3/nvToolsExt.h"):
        hit = next((os.path.join(d, hdr) for d in incs if os.path.exists(os.path.join(d, hdr))), None)
        print(f"  {hdr}: {hit or 'NOT FOUND'}")
    return ":".join(incs)


cpath = cuda_include_cpath()
if cpath:
    os.environ["CPATH"] = cpath + (":" + os.environ["CPATH"] if os.environ.get("CPATH") else "")
os.environ["NVTE_CUDA_ARCHS"] = "86"      # A10 = Ampere sm_86
os.environ["NVTE_FRAMEWORK"] = "pytorch"
# Keep parallel nvcc jobs LOW: each compiles large CUDA kernels and uses several GB; too many
# OOM-kill the single serverless node ("cluster is unhealthy"). 2 trades build time for stability.
os.environ["MAX_JOBS"] = "2"
print("CPATH set:", bool(cpath), "| NVTE_CUDA_ARCHS=86 | NVTE_FRAMEWORK=pytorch")
print("nvcc:", shutil.which("nvcc"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build the wheels (source build — the slow step, ~40 min)

# COMMAND ----------

pkg = "transformer_engine[pytorch]" + (f"=={te_version}" if te_version else "")
print("Building:", pkg)

# Install first (compiles + caches build artifacts), then capture the wheels.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-v",
                       "--no-build-isolation", pkg])

# COMMAND ----------

# Capture wheels: rebuild into a dir (fast now — objects are cached), plus scoop pip's wheel cache.
os.makedirs("/tmp/te_wheels", exist_ok=True)
for spec in ("transformer_engine", "transformer_engine_torch"):
    s = spec + (f"=={te_version}" if te_version else "")
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                    "--no-build-isolation", s, "-w", "/tmp/te_wheels"], check=False)

home = os.path.expanduser("~")
for w in glob.glob(f"{home}/.cache/pip/wheels/**/transformer_engine*.whl", recursive=True):
    shutil.copy2(w, "/tmp/te_wheels/")

built = sorted(set(glob.glob("/tmp/te_wheels/transformer_engine*.whl")))
print("Built wheels:", [os.path.basename(w) for w in built])
if not built:
    raise RuntimeError("No transformer_engine wheels were built/captured.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Copy wheels to the libraries Volume

# COMMAND ----------

for w in built:
    dst = os.path.join(libraries_volume, os.path.basename(w))
    shutil.copy2(w, dst)
    print(f"copied {os.path.basename(w)} ({os.path.getsize(w)/1e6:.1f} MB) -> {dst}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bonus: probe Hugging Face egress (does serverless reach the HF CDN?)
# MAGIC Tells bionemo whether the ESM-2 weights must be pre-staged to the Volume.

# COMMAND ----------

try:
    from huggingface_hub import hf_hub_download
    # config.json is tiny (hub API); a weights shard exercises the LFS CDN.
    p = hf_hub_download("facebook/esm2_t33_650M_UR50D", "config.json")
    print("HF hub (config.json): OK ->", p)
    try:
        w = hf_hub_download("facebook/esm2_t33_650M_UR50D", "model.safetensors")
        print("HF LFS (model.safetensors): OK ->", w, f"({os.path.getsize(w)/1e6:.0f} MB)")
        print("HF_EGRESS: FULL (no pre-staging needed)")
    except Exception as e:
        print(f"HF LFS (weights) FAILED: {type(e).__name__}: {str(e)[:200]}")
        print("HF_EGRESS: LFS-BLOCKED (pre-stage ESM-2 weights to the Volume)")
except Exception as e:
    print(f"HF hub FAILED: {type(e).__name__}: {str(e)[:200]}")
    print("HF_EGRESS: BLOCKED (pre-stage ESM-2 to the Volume)")

# COMMAND ----------

wheels = [f for f in os.listdir(libraries_volume) if f.startswith("transformer_engine") and f.endswith(".whl")]
dbutils.notebook.exit(f"SUCCESS: {wheels}")

# Databricks notebook source
# MAGIC %md
# MAGIC # Stage the OpenFold3 checkpoint to the model Volume (deploy-time, serverless CPU)
# MAGIC
# MAGIC Downloads the ~2 GB OpenBind checkpoint ONCE here and persists it to the model Volume, so every
# MAGIC fold run just restores it (sequential copy, FUSE-safe) and never re-downloads. `setup_openfold`
# MAGIC pulls the checkpoint from S3 with boto3 MULTIPART (seeked writes), which a UC Volume FUSE mount
# MAGIC rejects — so we download to LOCAL disk first, then sequential-copy to the Volume. Runs on
# MAGIC serverless CPU (no GPU needed to download), and writes a `.of3_complete` marker the fold job checks.
# MAGIC Idempotent: if the marker already exists, this is a no-op.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_volume", "openfold3", "Model Volume")
dbutils.widgets.text("of3_version", "0.5.1", "openfold3 package version")
dbutils.widgets.text("of3_ckpt_name", "openbind-2025-06-30-174k", "OpenFold3 checkpoint name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("model_volume")
OF3_VERSION = dbutils.widgets.get("of3_version").strip()
CKPT_NAME = dbutils.widgets.get("of3_ckpt_name").strip()

# COMMAND ----------

import os, sys, json, subprocess, shutil

CACHE_DIR = "/tmp/of3_cache"
VENV_DIR = "/tmp/of3venv"
VOLUME_CACHE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/cache"
MARKER = os.path.join(VOLUME_CACHE, ".of3_complete")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(VOLUME_CACHE, exist_ok=True)

if os.path.exists(MARKER):
    print("Checkpoint already staged (.of3_complete present) — nothing to do.")
    dbutils.notebook.exit("already_staged")

# Free-space diagnostics (the venv ~several GB + the ~2 GB checkpoint both live under /tmp).
for p in ("/tmp", CACHE_DIR, VOLUME_CACHE):
    try:
        du = shutil.disk_usage(p)
        print(f"disk {p}: free={du.free/1e9:.1f}GB / total={du.total/1e9:.1f}GB")
    except Exception as e:  # noqa: BLE001
        print(f"disk {p}: n/a ({e})")

# COMMAND ----------

# DBTITLE 1,Build the isolated openfold3 venv
PY = os.path.join(VENV_DIR, "bin", "python")
SETUP_BIN = os.path.join(VENV_DIR, "bin", "setup_openfold")

if not os.path.exists(SETUP_BIN):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    r = subprocess.run(["uv", "venv", "--python", "3.12", VENV_DIR], capture_output=True, text=True)
    assert r.returncode == 0, f"uv venv failed (rc={r.returncode}): {r.stderr[-1500:]}"
    r = subprocess.run(["uv", "pip", "install", "--python", PY, f"openfold3=={OF3_VERSION}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"openfold3 install failed (rc={r.returncode}): {r.stderr[-2000:]}"
print("venv ready at", VENV_DIR)

# COMMAND ----------

# DBTITLE 1,Download the checkpoint to local disk, then sequential-copy to the Volume
SETUP_CFG = os.path.join(CACHE_DIR, "setup_config.json")
with open(SETUP_CFG, "w") as f:
    json.dump({"openfold_cache": CACHE_DIR, "param_directory": CACHE_DIR,
               "selected_parameters": "default", "force_download_parameters": False,
               "run_integration_tests": False, "update_biotite_ccd": False}, f)

env = dict(os.environ)
env["OPENFOLD_CACHE"] = CACHE_DIR  # download target = LOCAL disk (multipart-safe)

r = subprocess.run([SETUP_BIN, "--config", SETUP_CFG, "--skip-biotite-ccd-update"],
                   capture_output=True, text=True, timeout=3600,
                   stdin=subprocess.DEVNULL, env=env)
if r.returncode != 0:
    raise RuntimeError(
        f"setup_openfold failed (rc={r.returncode}).\n--- STDERR head ---\n{r.stderr[:1800]}\n"
        f"--- STDERR tail ---\n{r.stderr[-1800:]}")
print("Downloaded checkpoint to", CACHE_DIR)

# Sequential copy to the Volume (FUSE-safe), then the completion marker.
subprocess.run(["cp", "-r", f"{CACHE_DIR}/.", VOLUME_CACHE + "/"], check=True)
with open(MARKER, "w") as f:
    f.write(f"{CKPT_NAME}\n")
print("Staged checkpoint to", VOLUME_CACHE)
print("Contents:", os.listdir(VOLUME_CACHE))

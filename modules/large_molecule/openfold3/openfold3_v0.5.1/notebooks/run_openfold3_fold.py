# Databricks notebook source
# MAGIC %md
# MAGIC # OpenFold3 structure prediction (fold)
# MAGIC
# MAGIC Serverless-GPU (GPU_1xH100, AI Runtime client '6' = CUDA 13 / py3.12) folding with
# MAGIC **OpenFold3** (`openfold3==0.5.1`, Apache-2.0; default **OpenBind-0** checkpoint
# MAGIC `openbind-2025-06-30-174k`, also Apache-2.0). This replaces the classic-T4 AlphaFold2 job:
# MAGIC no ~2 TB genetic-database download — OpenFold3 pulls only the ~2 GB checkpoint (cached on the
# MAGIC model Volume) and gets its MSA from the remote ColabFold MSA server (toggle `use_msa`).
# MAGIC
# MAGIC Ported from the proven `make_of3_inproc` recipe in the insilico_binder_benchmark:
# MAGIC isolated `uv` venv -> `setup_openfold` (download/resolve ckpt) -> `run_openfold predict`.
# MAGIC
# MAGIC **Input:** `protein_sequence` (a single chain = monomer; multiple chains separated by `:` =
# MAGIC complex, e.g. `TARGET:BINDER`). **Output:** `ranked_0.pdb` (+ model.cif + confidences.json)
# MAGIC written to `/Volumes/{catalog}/{schema}/{model_volume}/results/{run_id}/`, and pTM / ipTM /
# MAGIC avg-pLDDT / min-interface-PAE logged to the MLflow run.

# COMMAND ----------

# DBTITLE 1,Install biotite in the notebook kernel (CIF->PDB parse happens here, not in the of3 venv)
# MAGIC %pip install -q biotite
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_volume", "openfold3", "Model Volume")
dbutils.widgets.text("run_id", "run", "MLflow Run Id")
dbutils.widgets.text("protein_sequence", "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDAATKTFTVTE", "Protein Sequence (':' separates complex chains)")
dbutils.widgets.text("user_email", "a@b.com", "User Email")
# use_msa: true -> query the remote ColabFold MSA server (paper-faithful; the target folds well but
# needs outbound egress and is slower). false -> single-sequence (self-contained, no egress, but the
# target folds poorly). Exposed as a job parameter; the app defaults it to true.
dbutils.widgets.text("use_msa", "true", "Use ColabFold MSA server (true/false)")
dbutils.widgets.text("num_samples", "1", "Number of diffusion samples")
dbutils.widgets.text("of3_version", "0.5.1", "openfold3 package version")
dbutils.widgets.text("of3_ckpt_name", "openbind-2025-06-30-174k", "OpenFold3 checkpoint name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("model_volume")
RUN_ID = dbutils.widgets.get("run_id")
PROTEIN_SEQUENCE = dbutils.widgets.get("protein_sequence").strip()
USER_EMAIL = dbutils.widgets.get("user_email")
USE_MSA = dbutils.widgets.get("use_msa").strip().lower() in ("true", "1", "yes")
NUM_SAMPLES = max(1, int(dbutils.widgets.get("num_samples")))
OF3_VERSION = dbutils.widgets.get("of3_version").strip()
CKPT_NAME = dbutils.widgets.get("of3_ckpt_name").strip()

# COMMAND ----------

import mlflow

# Mark the run started as early as possible so the app's status poll sees progress even while the
# (first-run) venv build + checkpoint download are still going.
with mlflow.start_run(run_id=RUN_ID):
    mlflow.set_tag("job_status", "started")
    mlflow.log_params({
        "model": "OpenFold3",
        "of3_version": OF3_VERSION,
        "of3_checkpoint": CKPT_NAME,
        "use_msa": USE_MSA,
        "num_diffusion_samples": NUM_SAMPLES,
        "protein_sequence": PROTEIN_SEQUENCE,
        "num_chains": len([c for c in PROTEIN_SEQUENCE.split(":") if c.strip()]),
    })

# COMMAND ----------

# DBTITLE 1,Paths — checkpoint cache persists on the model Volume
import os

# setup_openfold downloads the ~2 GB checkpoint from S3 with boto3 MULTIPART (parallel/seeked range
# writes). A UC Volume FUSE mount does NOT support random/seek writes -> OSError [Errno 5]. So the live
# OPENFOLD_CACHE lives on LOCAL disk (multipart-safe), and we persist/restore the checkpoint to a Volume
# dir with plain SEQUENTIAL copies (which FUSE does support) so later cold nodes skip the re-download.
# The isolated venv also lives on local disk (serverless uses /tmp, not /local_disk0).
CACHE_DIR = "/tmp/of3_cache"                                    # OPENFOLD_CACHE (local, multipart-safe)
VOLUME_CACHE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/cache"    # persistent (sequential copy only)
RESULTS_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/results/{RUN_ID}"
VENV_DIR = "/tmp/of3venv"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(VOLUME_CACHE, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

PY = os.path.join(VENV_DIR, "bin", "python")
OF3_BIN = os.path.join(VENV_DIR, "bin", "run_openfold")
SETUP_BIN = os.path.join(VENV_DIR, "bin", "setup_openfold")
RUNNER_YAML = os.path.join(CACHE_DIR, "runner.yml")
SETUP_CFG = os.path.join(CACHE_DIR, "setup_config.json")

def of3_env():
    e = dict(os.environ)
    e["OPENFOLD_CACHE"] = CACHE_DIR  # both setup (download) and predict (resolve) key off this
    return e

# COMMAND ----------

# DBTITLE 1,Build the isolated OpenFold3 venv + download the checkpoint (first run only)
import subprocess, sys, json

if not os.path.exists(OF3_BIN):
    print("Building isolated OpenFold3 venv (py3.12, openfold3==%s) ..." % OF3_VERSION)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    # Isolated venv (NO --system-site-packages): openfold3 brings its own CUDA torch wheel and must not
    # clash with the runtime's torch. uv fetches the py3.12 interpreter if the node lacks it.
    r = subprocess.run(["uv", "venv", "--python", "3.12", VENV_DIR], capture_output=True, text=True)
    assert r.returncode == 0, f"uv venv failed (rc={r.returncode}): {r.stderr[-1500:]}"
    r = subprocess.run(["uv", "pip", "install", "--python", PY, f"openfold3=={OF3_VERSION}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"openfold3 install failed (rc={r.returncode}): {r.stderr[-2000:]}"
    print("venv ready.")
else:
    print("Reusing existing OpenFold3 venv at", VENV_DIR)

# runner.yml: disable the optional fused kernels (cuEquivariance / DeepSpeed / Triton) that aren't
# installed in the venv, so inference falls back to pure torch. Schema matches the package's
# examples/example_runner_yamls (toggles under model_update.custom.settings.memory.eval).
with open(RUNNER_YAML, "w") as f:
    f.write(
        "model_update:\n  presets:\n    - predict\n  custom:\n    settings:\n      memory:\n"
        "        eval:\n          use_cueq_triangle_kernels: false\n"
        "          use_deepspeed_evo_attention: false\n"
        "          use_triton_triangle_kernels: false\n"
    )

# Non-interactive setup: the bare command prompts on stdin and would hang; drive it with a config file
# + stdin=DEVNULL. force_download_parameters=False so an already-cached checkpoint on the Volume is
# reused (first run downloads ~2 GB; later runs are a fast no-op). Skip the biotite CCD S3 fetch
# (proteins only — the bundled CCD suffices).
with open(SETUP_CFG, "w") as f:
    json.dump({"openfold_cache": CACHE_DIR, "param_directory": CACHE_DIR,
               "selected_parameters": "default", "force_download_parameters": False,
               "run_integration_tests": False, "update_biotite_ccd": False}, f)

import shutil as _sh
MARKER = os.path.join(VOLUME_CACHE, ".of3_complete")
# Restore from the Volume ONLY if a prior run marked the cache COMPLETE — never restore a half-written
# checkpoint from a failed run (that would poison the local cache).
if os.path.exists(MARKER):
    subprocess.run(["cp", "-r", f"{VOLUME_CACHE}/.", CACHE_DIR + "/"], check=False)
    print("Restored complete checkpoint cache from Volume ->", CACHE_DIR)

# Free-space diagnostics: the venv (~several GB) + the ~2 GB checkpoint both live under /tmp.
for p in ("/tmp", CACHE_DIR, VOLUME_CACHE):
    try:
        du = _sh.disk_usage(p)
        print(f"disk {p}: free={du.free/1e9:.1f}GB / total={du.total/1e9:.1f}GB")
    except Exception as _e:  # noqa: BLE001
        print(f"disk {p}: n/a ({_e})")

r = subprocess.run([SETUP_BIN, "--config", SETUP_CFG, "--skip-biotite-ccd-update"],
                   capture_output=True, text=True, timeout=2400,
                   stdin=subprocess.DEVNULL, env=of3_env())
if r.returncode != 0:
    raise RuntimeError(
        f"setup_openfold failed (rc={r.returncode}).\n--- STDERR head ---\n{r.stderr[:1800]}\n"
        f"--- STDERR tail ---\n{r.stderr[-1800:]}")
print("OpenFold3 checkpoint ready in", CACHE_DIR)

# Persist to the Volume for later cold nodes (best-effort, sequential copy) + a completion marker so the
# restore above only ever pulls a fully-downloaded checkpoint.
try:
    subprocess.run(["cp", "-r", f"{CACHE_DIR}/.", VOLUME_CACHE + "/"], check=False)
    open(MARKER, "w").write("ok")
    print("Persisted checkpoint cache to Volume ->", VOLUME_CACHE)
except Exception as e:  # noqa: BLE001
    print("cache persist skipped:", e)

# COMMAND ----------

# DBTITLE 1,Fold — build the query, run `run_openfold predict`
import tempfile, glob

# Split the input into chains: one chain = monomer; several (':'-separated) = complex. Label them
# A, B, C, ... in order (matches the AlphaFold2 job's ':' convention so the app contract is unchanged).
chains = [c.strip() for c in PROTEIN_SEQUENCE.split(":") if c.strip()]
assert chains, "protein_sequence is empty"
chain_ids = [chr(ord("A") + i) for i in range(len(chains))]
query = {"queries": {"cplx": {"chains": [
    {"molecule_type": "protein", "chain_ids": [cid], "sequence": seq}
    for cid, seq in zip(chain_ids, chains)]}}}

with mlflow.start_run(run_id=RUN_ID):
    mlflow.set_tag("job_status", "folding")

work = tempfile.mkdtemp(prefix="of3_")
query_json = os.path.join(work, "query.json")
out_dir = os.path.join(work, "out")
with open(query_json, "w") as f:
    json.dump(query, f)

msa_flag = f"--use-msa-server={'True' if USE_MSA else 'False'}"
cmd = [OF3_BIN, "predict", "--query-json", query_json, "--output-dir", out_dir,
       "--inference-ckpt-name", CKPT_NAME, msa_flag,
       "--num-diffusion-samples", str(NUM_SAMPLES), "--runner-yaml", RUNNER_YAML]
# MSA-server runs add a variable queue/search wait on top of inference; be generous.
timeout_s = 1800 + 300 * (NUM_SAMPLES - 1) + (1800 if USE_MSA else 0)
print("Running:", " ".join(cmd), f"(timeout {timeout_s}s)")
p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                   stdin=subprocess.DEVNULL, env=of3_env())
if p.returncode != 0:
    raise RuntimeError(f"run_openfold predict failed (rc={p.returncode}):\n{p.stderr[-3000:]}")
print(p.stdout[-2000:])

# COMMAND ----------

# DBTITLE 1,Parse outputs — CIF -> ranked_0.pdb, confidences -> MLflow metrics
import numpy as np
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb_io

def _find(pattern, exclude=None):
    hits = sorted(glob.glob(os.path.join(out_dir, "**", pattern), recursive=True))
    if exclude:
        hits = [h for h in hits if exclude not in os.path.basename(h)]
    return hits

cif_files = _find("*_model.cif") or _find("*.cif")
agg_files = _find("*_confidences_aggregated.json")
conf_files = _find("*_confidences.json", exclude="aggregated")
assert cif_files, f"No CIF produced by OpenFold3 in {out_dir}; stderr tail:\n{p.stderr[-1500:]}"

def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

agg = json.load(open(agg_files[0])) if agg_files else {}
conf = json.load(open(conf_files[0])) if conf_files else {}

ptm, iptm, avg_plddt = _num(agg.get("ptm")), _num(agg.get("iptm")), _num(agg.get("avg_plddt"))
# binder-chain pTM (chain B), when a complex was folded
cptm = agg.get("chain_ptm")
binder_ptm = None
if isinstance(cptm, dict):
    binder_ptm = _num(cptm.get("B", cptm.get("1", cptm.get(1))))
elif isinstance(cptm, (list, tuple)) and len(cptm) >= 2:
    binder_ptm = _num(cptm[1])

# min interface PAE across the cross-chain block (complex only)
min_ipae = None
pae = conf.get("pae")
if pae is not None and len(chains) == 2:
    m = np.asarray(pae, dtype=float)
    lt, lb = len(chains[0]), len(chains[1])
    if m.ndim == 2 and m.shape[0] == m.shape[1] == lt + lb:
        cross = np.concatenate([m[:lt, lt:].ravel(), m[lt:, :lt].ravel()])
        min_ipae = float(cross.min()) if cross.size else None

# Convert the top model CIF -> PDB (ranked_0.pdb) for the app's viewer; also keep the raw CIF +
# confidences in the results dir. biotite carries the per-atom pLDDT in the B-factor column.
arr = pdbx.get_structure(pdbx.CIFFile.read(cif_files[0]), model=1)
pdb_path = os.path.join(RESULTS_DIR, "ranked_0.pdb")
pf = pdb_io.PDBFile(); pf.set_structure(arr); pf.write(pdb_path)

import shutil
shutil.copy(cif_files[0], os.path.join(RESULTS_DIR, "model.cif"))
if conf_files:
    shutil.copy(conf_files[0], os.path.join(RESULTS_DIR, "confidences.json"))
if agg_files:
    shutil.copy(agg_files[0], os.path.join(RESULTS_DIR, "confidences_aggregated.json"))

print("Wrote", pdb_path)
print(f"ptm={ptm} iptm={iptm} binder_ptm={binder_ptm} avg_plddt={avg_plddt} min_interface_pae={min_ipae}")

# COMMAND ----------

# DBTITLE 1,Log metrics + mark complete
with mlflow.start_run(run_id=RUN_ID):
    for k, v in {"of3_ptm": ptm, "of3_iptm": iptm, "of3_binder_ptm": binder_ptm,
                 "of3_avg_plddt": avg_plddt, "of3_min_interface_pae": min_ipae}.items():
        if v is not None:
            mlflow.log_metric(k, v)
    mlflow.log_param("results_path", RESULTS_DIR)
    mlflow.set_tag("job_status", "fold_complete")

print("OpenFold3 fold complete for run", RUN_ID)

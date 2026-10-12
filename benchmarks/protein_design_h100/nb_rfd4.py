# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark — RFD4-Proteina · generation speed + GPU memory (H100)
# MAGIC Loads the registered UC model via MLflow and measures per-sample generation time + peak GPU
# MAGIC memory (torch + NVML) over the shared template set. Runs on `GPU_1xH100` / `client '6'` (CUDA 13,
# MAGIC rfd4's native runtime). rfd4's registered endpoint exposes de-novo + binder but not single-chain
# MAGIC motif, so we use its native **binder** mode at matched generated-length against the shared template.

# COMMAND ----------
dbutils.widgets.text("model_uri", "models:/main.genesis_workbench.rfd4_proteina/18", "Registered model URI")
dbutils.widgets.text("out_dir", "/Volumes/main/genesis_workbench/rfd4_proteina/benchmark_results", "Results dir (UC Volume)")
dbutils.widgets.text("max_samples", "50", "Number of samples to run")

# COMMAND ----------
# DBTITLE 1,Install RFD4-Proteina code deps (CUDA-13 stack). load_model fetches the checkpoints.
import subprocess, sys
GIT_URL = "github.com/NVIDIA-BioNeMo/RFD4-Proteina-Open-Model-Private.git"
GIT_REF = "84d83b98442e5946a6a626d0d99a19773207e894"
try:
    _gh = dbutils.secrets.get("dbx_genesis_workbench", "rfd4_github_token")
except Exception:
    _gh = None  # repo may be public by now -> unauthenticated clone
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
open("/tmp/bc.txt", "w").write("torch==2.14.1\n")
_cred = f"x-access-token:{_gh}@" if _gh else ""
_spec = f"rfd4-proteina[metrics-cuda13] @ git+https://{_cred}{GIT_URL}@{GIT_REF}"
subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--break-system-packages",
                "--build-constraint", "/tmp/bc.txt", "--torch-backend", "cu132", _spec,
                "mlflow>=3.12", "databricks-sdk>=0.150.0"], check=True)
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./bench_common

# COMMAND ----------
# MAGIC %run ./make_dataset

# COMMAND ----------
# DBTITLE 1,In-process rfd4 motif-scaffolding on the shared CIF + measured loop
# rfd4's registered pyfunc hardcodes writing the input to `target.pdb` and validates by `.cif` suffix,
# so it can't accept an external CIF for conditional design. Drive it in-process from the Volume
# checkpoints instead (mirrors vaccine_immunogen.load_rfd4 + generate_scaffold): preserve a central
# motif (C_CRD+C_SEQ), scaffold the flanks to the shared total length.
import os, json, tempfile, torch
import rfproteina
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from rfproteina.generate import load_ckpt_n_configure_inference, get_contig_or_design_problem_dataset
from rfproteina.datapipes.pipeline import build_design_validation_pipeline
from rfproteina.datapipes.collate import collate_batch

MODEL = "rfd4_proteina"
OUT_DIR = dbutils.widgets.get("out_dir")
MAX = int(dbutils.widgets.get("max_samples"))
VOL = "/Volumes/main/genesis_workbench/rfd4_proteina"
FLOW = f"{VOL}/flow_checkpoints/chk_epoch_00000169_step_000000340000-ema.ckpt"
AE = f"{VOL}/ae_checkpoints/2026_09_04_release_ae_z16_400k-ema.ckpt"
print("GPU:", gpu_info())

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")
pkg_dir = os.path.dirname(rfproteina.__file__)
_monomer = os.path.join(pkg_dir, "data", "benchmarks", "monomer", "monomer-short.cif")
overrides = ["inference_experiment=inference_on_user_inputs", "~metrics/refolding_oracles",
             "metrics_tags=[]", "num_workers=0", "seed=42", f"inputs={_monomer}",
             f"ckpt_path={os.path.dirname(FLOW)}", f"ckpt_name={os.path.basename(FLOW)}",
             f"autoencoder_ckpt_path={AE}", "out_dir=/tmp/rfd4_bench_out"]
GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=os.path.join(pkg_dir, "configs"), version_base="1.3"):
    cfg = compose(config_name="inference", overrides=overrides)
rfd4 = load_ckpt_n_configure_inference(cfg)
rfd4.to(device).eval()
transform = build_design_validation_pipeline(metrics_tags=None)

MOTIF_LEN = 30
class _InProc:
    """A .predict(spec) shim so run_benchmark's timing/NVML/error-logging wraps the in-process gen."""
    def predict(self, s):
        total = int(s["total_len"])
        lo = (total - MOTIF_LEN) // 2 + 1
        hi = lo + MOTIF_LEN - 1
        half = max(1, (total - MOTIF_LEN) // 2)
        seg = f"{half}-{half}"
        with tempfile.TemporaryDirectory() as tmp:
            cif = pdb_to_cif(s["template_pdb"], os.path.join(tmp, "t.cif"))   # real .cif path
            sel = f"res_id>={lo} and res_id<={hi}"
            spec = {"motif": {"input": cif, "contig": f"{seg}, A{lo}-{hi}, {seg}",
                              "conditions": {"C_CRD": {True: [{"select": sel}]},
                                             "C_SEQ": {True: [{"select": sel}]}}}}
            sp = os.path.join(tmp, "spec.json")
            open(sp, "w").write(json.dumps(spec))
            ds = get_contig_or_design_problem_dataset(sp, num_replicates=1, transform=transform)
            batch = collate_batch([ds[i] for i in range(len(ds))],
                                  schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample",
                                  fixed_dimensions=None)
            batch = rfd4.transfer_batch_to_device(batch, torch.device(device), 0)
            with torch.no_grad():
                return rfd4.predict_step(batch, 0)

specs = make_motif_dataset(n=MAX)
run_benchmark(MODEL, _InProc(), specs, lambda s: s, torch, OUT_DIR, extra={"mode": "motif_inproc"})

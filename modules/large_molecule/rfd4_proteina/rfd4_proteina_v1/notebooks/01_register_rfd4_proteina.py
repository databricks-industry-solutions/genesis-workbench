# Databricks notebook source
# MAGIC %md
# MAGIC # RFD4-Proteína — Register + deploy the design serving endpoint (H100)
# MAGIC
# MAGIC Mirrors `small_molecule/proteina_complexa` (its predecessor) but for the flow-matching RFD4-Proteína
# MAGIC foundation model. Runs on a **serverless-GPU H100 (AI Runtime v6 = CUDA 13)** node because the
# MAGIC model + partial-autoencoder need ~40-70 GB VRAM and the `rfproteina` stack is torch 2.14.1 / cu132
# MAGIC with a `TMol` source build + `cuEquivariance-cu13`.
# MAGIC
# MAGIC Pipeline: download checkpoints → MLflow PyFunc `log_model` → **Express** `register_model(env_pack=…)`
# MAGIC (snapshots THIS serverless-GPU/CUDA-13 env for serving) → `serving_endpoints.create_and_wait` on a
# MAGIC **GPU_XLARGE (1× H100)** endpoint. Express is what makes RFD4 servable: it restores this exact
# MAGIC torch-2.14.1/CUDA-13 env at serving instead of rebuilding on the standard torch-2.7.1/CUDA-11.8 base.
# MAGIC
# MAGIC Serving contract is proteina_complexa-compatible (UI drop-in) with an added `task`:
# MAGIC input `[task, target_pdb, binder_length_min, binder_length_max, num_samples, hotspot_residues, target_chain]`
# MAGIC → output `[sample_id, pdb_output, sequence, rewards]`.
# MAGIC
# MAGIC > Confirmed on a real H100 deploy: de-novo generation serves end-to-end via Express. The
# MAGIC > binder/motif condition-spec construction is still first-draft and gets refined in the design workflows.

# COMMAND ----------

# DBTITLE 1,Base tooling (the heavy rfproteina/CUDA-13 stack is installed via uv a few cells down)
# MAGIC %pip install -q "databricks-sdk>=0.50.0" "databricks-sql-connector>=4.0.2" "mlflow>=2.15"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "rfd4_proteina", "Model Name")
dbutils.widgets.text("cache_dir", "rfd4_proteina", "Cache dir (UC volume)")
dbutils.widgets.text("rfd4_git_url", "github.com/NVIDIA-BioNeMo/RFD4-Proteina-Open-Model-Private.git", "RFD4 git host/path")
dbutils.widgets.text("rfd4_git_ref", "84d83b98442e5946a6a626d0d99a19773207e894", "RFD4 git ref (commit/branch/tag)")
dbutils.widgets.text("rfd4_git_token_scope", "dbx_genesis_workbench", "Secret scope for the GitHub PAT")
dbutils.widgets.text("rfd4_git_token_key", "rfd4_github_token", "Secret key for the GitHub PAT")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "MLflow experiment")
dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("workload_type", "GPU_XLARGE", "Serving workload type (GPU_XLARGE = 1x H100)")
dbutils.widgets.text("rfd4_ckpt_base_url", "https://rfd4-proteina-public-e04a.cwobject.com", "Checkpoint base URL")
dbutils.widgets.text("rfd4_flow_ckpt", "chk_epoch_00000169_step_000000340000-ema.ckpt", "Flow checkpoint")
dbutils.widgets.text("rfd4_ae_ckpt", "2026_09_04_release_ae_z16_400k-ema.ckpt", "Autoencoder checkpoint")

g = dbutils.widgets.get
catalog, schema, model_name, cache_dir = g("catalog"), g("schema"), g("model_name"), g("cache_dir")
experiment_name = g("experiment_name")
sql_warehouse_id, user_email, workload_type = g("sql_warehouse_id"), g("user_email"), g("workload_type")
ckpt_base_url, flow_ckpt, ae_ckpt = g("rfd4_ckpt_base_url"), g("rfd4_flow_ckpt"), g("rfd4_ae_ckpt")
git_url, git_ref = g("rfd4_git_url"), g("rfd4_git_ref")
git_token_scope, git_token_key = g("rfd4_git_token_scope"), g("rfd4_git_token_key")

vol_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"
flow_dir, ae_dir = f"{vol_root}/flow_checkpoints", f"{vol_root}/ae_checkpoints"
print(f"catalog={catalog} schema={schema} vol_root={vol_root}\nrfd4_git={git_url}@{git_ref} workload_type={workload_type}")

# COMMAND ----------

# DBTITLE 1,Install the rfproteina CUDA-13 stack (torch 2.14.1/cu132 + TMol source build + cuEquivariance-cu13)
# This is the make-or-break step. client '6' (AI Runtime v6) is CUDA 13 so the cu132 wheels + the TMol
# nvcc build resolve against the base toolkit. uv handles the TMol build isolation. We pip-install
# rfproteina STRAIGHT FROM THE PRIVATE GITHUB REPO — nothing is vendored into GWB and a customer needs
# nothing cloned locally — authenticating with a GitHub PAT read from the GWB secret scope. Pinned to a
# commit for reproducibility. Installing from the real repo means every build file (README/LICENSE/
# .project-root/data) is present, so the hatchling build just works.
import os, subprocess, sys

try:
    gh_token = dbutils.secrets.get(git_token_scope, git_token_key)
except Exception:
    gh_token = None  # OK once the RFD4-Proteina repo is public — fall back to an unauthenticated clone

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
bc = "/tmp/rfd4_build_constraint.txt"
open(bc, "w").write("torch==2.14.1\n")
# Install into THIS kernel's env (sys.executable → /databricks/python, writable + survives the restart
# below — the diffdock serverless pattern). uv --system wrongly targets the READ-ONLY /usr; --python
# sys.executable targets the kernel env so rfproteina's deps (e.g. a newer typing_extensions that the
# base runtime lacks `sentinel` for) overwrite in-place and are live after restartPython. --torch-backend
# cu132 pins CUDA-13 torch; --build-constraint aligns TMol's nvcc build torch.
cred = f"x-access-token:{gh_token}@" if gh_token else ""
spec = f"rfd4-proteina[metrics-cuda13] @ git+https://{cred}{git_url}@{git_ref}"
# Capture uv output: write the FULL log to the Volume (readable via `fs cat`) and embed the stderr tail
# in the exception so the failure cause is visible via get-run-output (which has no driver-logs field).
_r = subprocess.run(
    ["uv", "pip", "install", "--python", sys.executable, "--break-system-packages",
     "--build-constraint", bc, "--torch-backend", "cu132", spec,
     "mlflow>=3.12", "databricks-sdk>=0.150.0"],   # Express deploy API (env_pack) + GPU_XLARGE enum
    capture_output=True, text=True,
)
os.makedirs(f"{vol_root}/register_logs", exist_ok=True)
_log = (_r.stdout or "") + "\n==STDERR==\n" + (_r.stderr or "")
if gh_token:
    _log = _log.replace(gh_token, "<redacted>")  # never persist the PAT
open(f"{vol_root}/register_logs/uv_install.log", "w").write(_log)
print(_log[-4000:])
if _r.returncode != 0:
    hint = ("" if gh_token else
            f"\n(No token at '{git_token_scope}/{git_token_key}'. While the repo is private, set a PAT: "
            f"databricks secrets put-secret {git_token_scope} {git_token_key} --string-value <PAT>)")
    raise RuntimeError(f"uv install failed rc={_r.returncode}; stderr tail:\n{_log[-3000:]}{hint}")
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Re-read widgets after restart + download checkpoints (skip-if-exists)
import os, requests
g = dbutils.widgets.get
catalog, schema, model_name, cache_dir = g("catalog"), g("schema"), g("model_name"), g("cache_dir")
sql_warehouse_id, user_email, workload_type = g("sql_warehouse_id"), g("user_email"), g("workload_type")
experiment_name = g("experiment_name")
ckpt_base_url, flow_ckpt, ae_ckpt = g("rfd4_ckpt_base_url"), g("rfd4_flow_ckpt"), g("rfd4_ae_ckpt")
vol_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"
flow_dir, ae_dir = f"{vol_root}/flow_checkpoints", f"{vol_root}/ae_checkpoints"

def _download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"exists, skip: {dest}"); return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=1800) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=16 << 20):
                f.write(chunk)
    assert os.path.getsize(dest) > 1_000_000, f"download too small: {dest}"

_download(f"{ckpt_base_url}/flow_checkpoints/{flow_ckpt}", f"{flow_dir}/{flow_ckpt}")
_download(f"{ckpt_base_url}/ae_checkpoints/{ae_ckpt}", f"{ae_dir}/{ae_ckpt}")

# COMMAND ----------

# DBTITLE 1,MLflow PyFunc — drives RFD4 exactly as rfd4-generate does (warm, in-process)
import mlflow
from mlflow.pyfunc import PythonModel, PythonModelContext

class RFD4ProteinaModel(PythonModel):
    """RFD4-Proteina design endpoint.

    Inference runs through the model's OWN supported path (mirrors the model repo's
    tests/test_inference_integration.py + rfproteina/generate.py), NOT a hand-rolled batch:

      load_context (once, warm):
        compose the Hydra inference config -> load_ckpt_n_configure_inference() loads the flow + AE
        checkpoints, applies a LoRA adapter if one is attached, and sets the sampling config ->
        build the inference transform pipeline. The model stays resident for low-latency serving.

      predict (per request):
        turn the row into the model's own input -> get_contig_or_design_problem_dataset() (a CIF path
        -> DesignProblemDataset, a condition_spec .json -> ConditionSpecDataset) applies the transform
        -> collate_batch(ProteinaDataSample schema) -> predict_step -> designed structure + sequence.

    Serving contract (proteina_complexa-compatible + a `task` column):
      in : task, target_pdb, binder_length_min, binder_length_max, num_samples, hotspot_residues, target_chain
      out: sample_id, pdb_output (PDB text), sequence, rewards
    Modes:
      - task="de_novo" (or empty target_pdb): length-only monomer generation. RFD4's de-novo input is a
        length-only monomer CIF, so we feed the bundled rfproteina/data/benchmarks/monomer scaffold.
      - task in {"binder","motif"}: builds a condition_spec from target_pdb + a contig + hotspot selects
        (see docs/workflow/condition-spec.md). FIRST-DRAFT conditioning — refined in the design workflows.
    """

    def load_context(self, context):
        # Express packages THIS serverless-GPU/CUDA-13 env (torch 2.14.1/cu132 + cuEquivariance-cu13) at
        # registration and restores it verbatim at serving, so rfproteina imports + loads exactly as it does
        # on this register node — no base-torch compat shims and no load-failure diagnostics needed.
        import os
        import torch
        import rfproteina
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from rfproteina.generate import load_ckpt_n_configure_inference
        from rfproteina.datapipes.pipeline import build_design_validation_pipeline

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_float32_matmul_precision("high")
        self._pkg_dir = os.path.dirname(rfproteina.__file__)   # ships via code_paths -> configs + benchmarks

        flow, ae = context.artifacts["flow_ckpt"], context.artifacts["ae_ckpt"]
        adapter = context.artifacts.get("adapter_dir")         # set only for a fine-tuned deploy (nb 03)
        # `inputs` is a mandatory config key (generation.dataset interpolates ${inputs}); we drive the
        # dataset per-request in predict() instead, so point it at the bundled monomer CIF purely to
        # satisfy config resolution here.
        _monomer_cif = os.path.join(self._pkg_dir, "data", "benchmarks", "monomer", "monomer-short.cif")
        overrides = [
            "inference_experiment=inference_on_user_inputs",
            "~metrics/refolding_oracles",                      # no AF3 refolding oracle at serving time
            "metrics_tags=[]", "num_workers=0", "seed=42",
            f"inputs={_monomer_cif}",
            f"ckpt_path={os.path.dirname(flow)}", f"ckpt_name={os.path.basename(flow)}",
            f"autoencoder_ckpt_path={ae}", "out_dir=/tmp/rfd4_out",
        ]
        if adapter:
            overrides.append(f"adapter_path={adapter}")
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=os.path.join(self._pkg_dir, "configs"), version_base="1.3"):
            cfg = compose(config_name="inference", overrides=overrides)
        self.model = load_ckpt_n_configure_inference(cfg)      # the model's own loader + configurer
        self.model.to(self.device).eval()
        self.transform = build_design_validation_pipeline(metrics_tags=None)

    def _request_to_input_path(self, task, pdb_text, length_min, length_max, hotspots, tmpdir):
        """Return the path to feed get_contig_or_design_problem_dataset: a length-only monomer CIF for
        de-novo, or a condition_spec .json (target + contig + conditions) for binder/motif design."""
        import os, json
        if task == "de_novo" or not pdb_text.strip():
            return os.path.join(self._pkg_dir, "data", "benchmarks", "monomer", "monomer-short.cif")
        target = os.path.join(tmpdir, "target.pdb")
        open(target, "w").write(pdb_text)
        # contig: a fresh binder chain of `length` residues (/0 = chain break), then the whole target chain A.
        contig = f"{int(length_min)}-{int(length_max)}, /0, A"
        conditions = {"C_CRD": {True: [{"select": "sel('A') and is_protein_backbone"}]},  # target backbone
                      "C_SEQ": {True: [{"select": "sel('A')"}]}}                           # target sequence
        if hotspots.strip():
            conditions["C_HOT"] = {True: [{"select": f"sel('A/*/{h.strip()}')"}
                                          for h in hotspots.split(",") if h.strip()]}
        spec_path = os.path.join(tmpdir, "spec.json")
        open(spec_path, "w").write(json.dumps({task: {"input": target, "contig": contig, "conditions": conditions}}))
        return spec_path

    def _structure_to_pdb_and_seq(self, sample):
        import io, numpy as np
        aa = sample["generated_atom_array"]
        seq = ""
        try:
            from biotite.structure import to_sequence
            seqs, _ = to_sequence(aa)
            seq = "".join(str(s) for s in seqs)
        except Exception:
            pass
        # The generated array's b_factor/occupancy/charge can hold out-of-range (even non-finite)
        # values that overflow PDB's fixed column widths; reset them to safe defaults for a clean PDB,
        # and fall back to mmCIF (no column limits) if PDB is still incompatible.
        cats, n = set(aa.get_annotation_categories()), aa.array_length()
        if "b_factor" in cats:
            aa.set_annotation("b_factor", np.zeros(n))
        if "occupancy" in cats:
            aa.set_annotation("occupancy", np.ones(n))
        if "charge" in cats:
            aa.set_annotation("charge", np.zeros(n, dtype=int))
        try:
            from biotite.structure.io.pdb import PDBFile
            buf = io.StringIO(); pf = PDBFile(); pf.set_structure(aa); pf.write(buf)
            return buf.getvalue(), seq
        except Exception:
            from biotite.structure.io.pdbx import CIFFile, set_structure
            cf = CIFFile(); set_structure(cf, aa); buf = io.StringIO(); cf.write(buf)
            return buf.getvalue(), seq

    def predict(self, context, model_input, params=None):
        import pandas as pd
        import tempfile
        from rfproteina.generate import get_contig_or_design_problem_dataset
        from rfproteina.datapipes.collate import collate_batch
        torch = self._torch

        rows, sid = [], 0
        for _, r in model_input.iterrows():
            task = str(r.get("task", "de_novo") or "de_novo")
            n = int(r.get("num_samples", 1) or 1)
            with tempfile.TemporaryDirectory() as tmp:
                inputs = self._request_to_input_path(
                    task, str(r.get("target_pdb", "") or ""),
                    r.get("binder_length_min", 100), r.get("binder_length_max", 100),
                    str(r.get("hotspot_residues", "") or ""), tmp)
                ds = get_contig_or_design_problem_dataset(inputs, num_replicates=n, transform=self.transform)
                batch = collate_batch([ds[i] for i in range(len(ds))],
                                      schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample",
                                      fixed_dimensions=None)
                # Use the model's OWN device transfer (what the Trainer calls) — the batch holds
                # structured containers a plain tensor-recursion misses, which left inputs on CPU.
                batch = self.model.transfer_batch_to_device(batch, torch.device(self.device), 0)
                with torch.no_grad():
                    outs = self.model.predict_step(batch, 0)
            for out in outs:
                pdb_str, seq = self._structure_to_pdb_and_seq(out)
                rows.append({"sample_id": sid, "pdb_output": pdb_str, "sequence": seq, "rewards": None}); sid += 1
        return pd.DataFrame(rows)

# COMMAND ----------

# DBTITLE 1,Signature + dry-load test (de novo, short) on this H100
import sys, pandas as pd
# rfproteina is system-installed from git (install cell above) — importable directly after restart.
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# "long" (int64) matches what pandas produces for Python ints, so the input_example + sample_id output
# pass MLflow schema enforcement (an "integer"/int32 schema rejects the int64 example).
input_schema = Schema([
    ColSpec("string", "task"), ColSpec("string", "target_pdb"),
    ColSpec("long", "binder_length_min"), ColSpec("long", "binder_length_max"),
    ColSpec("long", "num_samples"), ColSpec("string", "hotspot_residues"), ColSpec("string", "target_chain"),
])
output_schema = Schema([ColSpec("long", "sample_id"), ColSpec("string", "pdb_output"),
                        ColSpec("string", "sequence"), ColSpec("double", "rewards")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

artifacts = {"flow_ckpt": f"{flow_dir}/{flow_ckpt}", "ae_ckpt": f"{ae_dir}/{ae_ckpt}"}
_m = RFD4ProteinaModel(); _m.load_context(PythonModelContext(artifacts=artifacts, model_config={}))
_sample = pd.DataFrame([{"task": "de_novo", "target_pdb": "", "binder_length_min": 60,
                         "binder_length_max": 60, "num_samples": 1, "hotspot_residues": "", "target_chain": "A"}])
print(_m.predict(None, _sample)[["sample_id", "sequence"]].head())

# COMMAND ----------

# DBTITLE 1,Register to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user_email}/{experiment_name}")
uc_model_name = f"{catalog}.{schema}.{model_name}"

# Ship rfproteina's package source (+ repo siblings) via code_paths so the model carries its configs +
# data/benchmarks. We deliberately DON'T pin the heavy CUDA-13 deps via pip_requirements anymore: Express
# packaging (env_pack, two cells down) snapshots this exact serverless-GPU env for serving, so there's no
# pip re-resolution at deploy time and the old local-tag gymnastics (torch==2.14.1+cu132, the tmol source
# build, the nvidia index, the dry-run guard) are gone. pip_requirements stays minimal/trivially-resolvable
# — env_pack, not requirements.txt, defines the serving environment.
import importlib
code_paths = []
for _pkg in ("rfproteina", "rfd4_proteina", "script_utils", "gearnet"):
    try:
        code_paths.append(os.path.dirname(importlib.import_module(_pkg).__file__))
    except Exception:
        pass
print("code_paths:", code_paths)

# ~8GB checkpoints: use the fast UC artifact path so log_model doesn't hit MLflow's 5-min upload timeout.
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "false"

# Log WITHOUT registered_model_name — Express requires registering in a SEPARATE register_model() step
# (below) so env_pack can run.
with mlflow.start_run(run_name="rfd4_proteina_register"):
    model_info = mlflow.pyfunc.log_model(
        name=model_name,
        python_model=RFD4ProteinaModel(),
        artifacts=artifacts,
        code_paths=code_paths,
        signature=signature,
        input_example=_sample,
        pip_requirements=["mlflow", "cloudpickle", "pandas", "numpy", "biotite"],
    )
print(f"logged model: {model_info.model_uri}")

# COMMAND ----------

# DBTITLE 1,Express-register (env_pack) — snapshots THIS GPU env for serving (no deploy-time rebuild)
# Express deployments (docs: machine-learning/model-serving/express-deployments) PACKAGE this serverless-
# GPU environment at registration and RESTORE it, as-is, at serving — there is NO deploy-time container
# build and NO torch force-install. That is the ENTIRE reason RFD4 can be served: STANDARD GPU serving
# rebuilds on a torch-2.7.1 / CUDA-11.8 base (a hardcoded build post-step force-reinstalls torch
# 2.7.1+cu118), which the torch-2.14 / CUDA-13 rfproteina stack cannot run (has_static_value /
# torch.optim.Muon / libnvrtc.so.13 all fail). With env_pack the serving container IS this env, so the
# model loads + generates exactly as it does on this register node. Requires mlflow>=3.12 +
# databricks-sdk>=0.102.0 (installed in the uv cell) and that we register FROM serverless GPU.
import torch
from mlflow.utils.env_pack import EnvPackConfig
assert torch.cuda.is_available(), "env_pack must run on serverless GPU so the packaged env is the CUDA-13 GPU env"

# install_dependencies=False: snapshot the already-installed site-packages VERBATIM instead of trying to
# re-resolve them from an index. Our env is full of git/binary/local-tagged packages (rfproteina @ git,
# tmol nvcc build, torch==2.14.1+cu132, cuEquivariance-cu13) that resolve on NO index — a re-resolution
# (the default True) would fail exactly like the old serving build ("No matching distribution for
# torch==2.14.1+cu132"). False packages the exact working bits (serving egress is restricted anyway).
model_version = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=uc_model_name,
    env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False),
).version
print(f"registered {uc_model_name} v{model_version} with Express env_pack (GPU env snapshotted for serving)")

# COMMAND ----------

# DBTITLE 1,Express-deploy the serving endpoint — the packaged GPU env is restored as-is (no rebuild)
from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (EndpointCoreConfigInput, ServedEntityInput,
                                            ServingModelWorkloadType, EndpointTag)

w = WorkspaceClient()
try:
    _pfx = dbutils.secrets.get("dbx_genesis_workbench", "dev_user_prefix")
except Exception:
    _pfx = ""
endpoint_name = f"gwb_{_pfx}_{model_name}_endpoint" if _pfx and _pfx.strip() else f"gwb_{model_name}_endpoint"

served = [ServedEntityInput(
    entity_name=uc_model_name, entity_version=str(model_version), name=model_name,
    workload_type=ServingModelWorkloadType(workload_type),   # GPU_XLARGE = 1x H100 (us-west-2, enrolled)
    workload_size="Small", scale_to_zero_enabled=False,       # GPU_XLARGE does not support scale-to-zero
)]
# application / created_by tags, matching deploy_model_endpoint's convention. create carries them inline;
# update_config_and_wait does NOT touch tags, so (re)apply them with patch() after an update — otherwise
# every redeploy of an existing endpoint drops them.
endpoint_tags = [EndpointTag(key="application", value="genesis_workbench"),
                 EndpointTag(key="created_by", value=user_email)]
print(f"express-deploying {endpoint_name}: {uc_model_name} v{model_version} on {workload_type}")

_exists = True
try:
    w.serving_endpoints.get(endpoint_name)
except Exception:
    _exists = False
if _exists:
    w.serving_endpoints.update_config_and_wait(name=endpoint_name, served_entities=served,
                                               timeout=timedelta(minutes=120))
    w.serving_endpoints.patch(name=endpoint_name, add_tags=endpoint_tags)
else:
    w.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=EndpointCoreConfigInput(name=endpoint_name, served_entities=served),
        tags=endpoint_tags,
        timeout=timedelta(minutes=120))
print(f"endpoint {endpoint_name} is live")

# COMMAND ----------

# DBTITLE 1,Register in the GWB app registry so the UI lists this endpoint (metadata only)
# The endpoint is already live (Express owns it). This resolves the GWB wheel path; the next cell installs
# it (which downgrades mlflow/sdk — fine, serving is already done) and writes app-registry metadata.
gwb_lib = None
for lib in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries"):
    if lib.name.startswith("genesis_workbench"):
        gwb_lib = lib.path.replace("dbfs:", "")
print(gwb_lib)

# COMMAND ----------

# MAGIC %pip install {gwb_lib}
# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Writes GWB app-registry metadata so the UI lists the endpoint. Does NOT call deploy_model — Express
# already created/updated the endpoint above. Re-reads widgets after the restart (which wiped the kernel,
# so names from pre-restart cells like `endpoint_name` are gone — don't reference them here). Wrapped
# non-fatally: the endpoint (the critical deliverable) is already live, so a registry hiccup must NOT fail
# the task and trigger a full-notebook retry (re-install + re-env_pack + re-deploy).
import mlflow, traceback
g = dbutils.widgets.get
catalog, schema, model_name = g("catalog"), g("schema"), g("model_name")
user_email, sql_warehouse_id = g("user_email"), g("sql_warehouse_id")
try:
    from genesis_workbench.models import ModelCategory, import_model_from_uc, get_latest_model_version
    from genesis_workbench.workbench import initialize

    databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
    initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

    uc_model_name = f"{catalog}.{schema}.{model_name}"
    model_version = get_latest_model_version(uc_model_name)
    gwb_model_id = import_model_from_uc(
        user_email=user_email,
        model_category=ModelCategory.LARGE_MOLECULE,
        model_uc_name=uc_model_name,
        model_uc_version=model_version,
        model_name="RFD4-Proteina",
        model_display_name="RFD4-Proteina Design",
        model_source_version="early-access",
        model_description_url="https://github.com/NVIDIA-BioNeMo/RFD4-Proteina-Open-Model-Private",
    )
    print(f"GWB app-registry model_id={gwb_model_id} ({model_name} endpoint already live via Express)")
except Exception:
    print("WARNING: GWB app-registry metadata step failed — the endpoint is already live, so NOT failing "
          "the task. Fix separately if the model should appear in the GWB app UI:\n" + traceback.format_exc())

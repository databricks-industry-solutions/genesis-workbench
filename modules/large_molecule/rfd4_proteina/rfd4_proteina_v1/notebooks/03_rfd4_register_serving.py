# Databricks notebook source
# MAGIC %md
# MAGIC # RFD4-Proteína — Register + deploy a FINE-TUNED model (H100, Express)
# MAGIC
# MAGIC KERMT-v2 deploy flow, RFD4 version, on the SAME Express `env_pack` path as `01_register`: take a
# MAGIC fine-tuned adapter (`ft_id` from `rfd4_weights`; empty = latest active), wrap base-flow + AE + PEFT
# MAGIC adapter in the same PyFunc as `01_register` (the only extra artifact is the LoRA `adapter_dir`),
# MAGIC register to UC, **Express-package** THIS serverless-GPU/CUDA-13 env, and deploy a **GPU_XLARGE
# MAGIC (1× H100)** serving endpoint. Express restores this exact torch-2.14.1/CUDA-13 env at serving (no
# MAGIC deploy-time container rebuild, no torch force-install), which is what makes the model servable —
# MAGIC same reason as the base deploy.
# MAGIC
# MAGIC Serving contract is proteina_complexa-compatible (UI drop-in) with the added `task` column:
# MAGIC input `[task, target_pdb, binder_length_min, binder_length_max, num_samples, hotspot_residues, target_chain]`
# MAGIC → output `[sample_id, pdb_output, sequence, rewards]`. The fine-tuned version is registered as a new
# MAGIC version of the same `rfd4_proteina` UC model and deployed onto the same endpoint as the base
# MAGIC (`01_register`), updating it in place.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("cache_dir", "rfd4_proteina", "Cache dir (UC volume)")
dbutils.widgets.text("rfd4_git_url", "github.com/NVIDIA-BioNeMo/RFD4-Proteina-Open-Model-Private.git", "RFD4 git host/path")
dbutils.widgets.text("rfd4_git_ref", "84d83b98442e5946a6a626d0d99a19773207e894", "RFD4 git ref (commit/branch/tag)")
dbutils.widgets.text("rfd4_git_token_scope", "dbx_genesis_workbench", "Secret scope for the GitHub PAT")
dbutils.widgets.text("rfd4_git_token_key", "rfd4_github_token", "Secret key for the GitHub PAT")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse Id")
dbutils.widgets.text("ft_id", "", "Fine-tuned id (empty = latest active)")
dbutils.widgets.text("model_name", "rfd4_proteina", "UC model name")
dbutils.widgets.text("workload_type", "GPU_XLARGE", "Serving workload type (GPU_XLARGE = 1x H100)")

g = dbutils.widgets.get
catalog, schema, cache_dir = g("catalog"), g("schema"), g("cache_dir")
user_email, sql_warehouse_id = g("user_email"), g("sql_warehouse_id")
git_url, git_ref = g("rfd4_git_url"), g("rfd4_git_ref")
git_token_scope, git_token_key = g("rfd4_git_token_scope"), g("rfd4_git_token_key")
ft_id, model_name, workload_type = g("ft_id"), g("model_name"), g("workload_type")
vol_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"

# COMMAND ----------

# DBTITLE 1,Install the rfproteina CUDA-13 stack + the Express deploy APIs, then restart
import subprocess, sys
# Install rfproteina from the private GitHub repo (nothing vendored; customer needs nothing local),
# authenticating with a GitHub PAT from the GWB secret scope — the same CUDA-13 stack as 01_register. Also
# install mlflow>=3.12 + databricks-sdk>=0.150.0 (the Express env_pack API). restartPython() makes the
# just-installed torch 2.14.1/cu132 live in-kernel so the dry-load + env_pack register FROM this exact env.
try:
    gh_token = dbutils.secrets.get(git_token_scope, git_token_key)
except Exception:
    gh_token = None  # OK once the repo is public — fall back to an unauthenticated clone
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
open("/tmp/bc.txt", "w").write("torch==2.14.1\n")
# Install into THIS kernel's env (sys.executable → /databricks/python, writable + survives the restart; the
# diffdock serverless pattern). uv --system wrongly targets the read-only /usr. --torch-backend cu132 pins
# CUDA-13 torch; --build-constraint aligns TMol's nvcc build torch.
cred = f"x-access-token:{gh_token}@" if gh_token else ""
spec = f"rfd4-proteina[metrics-cuda13] @ git+https://{cred}{git_url}@{git_ref}"
_r = subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--break-system-packages",
                     "--build-constraint", "/tmp/bc.txt", "--torch-backend", "cu132", spec,
                     "mlflow>=3.12", "databricks-sdk>=0.150.0"],   # Express deploy API (env_pack) + GPU_XLARGE enum
                    capture_output=True, text=True)
if _r.returncode != 0:
    _err = (_r.stderr or "")
    if gh_token:
        _err = _err.replace(gh_token, "<redacted>")
    hint = "" if gh_token else f" (no token at {git_token_scope}/{git_token_key}; set a PAT if the repo is still private)"
    raise RuntimeError(f"uv install failed rc={_r.returncode}{hint}; stderr tail:\n{_err[-3000:]}")
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Re-read widgets after restart + resolve ft_id/adapter from rfd4_weights
import glob, os
g = dbutils.widgets.get
catalog, schema, cache_dir = g("catalog"), g("schema"), g("cache_dir")
user_email, sql_warehouse_id = g("user_email"), g("sql_warehouse_id")
ft_id, model_name, workload_type = g("ft_id"), g("model_name"), g("workload_type")
vol_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"

if not ft_id:
    row = spark.sql(f"SELECT ft_id FROM {catalog}.{schema}.rfd4_weights WHERE is_active = true ORDER BY ft_id DESC LIMIT 1").collect()
    assert row, "no active fine-tune found in rfd4_weights — run 02_rfd4_finetune first"
    ft_id = str(row[0]["ft_id"])
meta = spark.sql(f"SELECT ft_label, adapter_volume_location, pretrain_ckpt FROM {catalog}.{schema}.rfd4_weights WHERE ft_id = {ft_id} AND is_active = true").collect()
assert meta, f"ft_id {ft_id} not found/active in rfd4_weights"
ft_label, adapter_loc, pretrain_ckpt = meta[0]["ft_label"], meta[0]["adapter_volume_location"], meta[0]["pretrain_ckpt"]
# AE ckpt lives alongside the base flow ckpt under the cache volume.
ae_ckpt = (glob.glob(f"{vol_root}/ae_checkpoints/*.ckpt") or [""])[0]
print(f"ft_id={ft_id} label={ft_label} adapter={adapter_loc} base={pretrain_ckpt} ae={ae_ckpt}")

# COMMAND ----------

# DBTITLE 1,PyFunc (base + PEFT adapter) — same contract as 01_register
import mlflow, pandas as pd, sys
# rfproteina is system-installed from git (install cell above) — importable directly after restart.
from mlflow.pyfunc import PythonModel, PythonModelContext

class RFD4ProteinaFineTunedModel(PythonModel):
    """Fine-tuned RFD4-Proteina endpoint — identical wrapper to the base model (01_register), the only
    difference being the LoRA `adapter_dir` artifact: load_ckpt_n_configure_inference applies it via its
    `adapter_path` override. Drives inference through the model's own supported path (mirrors the model
    repo's tests/test_inference_integration.py + rfproteina/generate.py), model loaded once and warm.

    Serving contract (proteina_complexa-compatible + a `task` column):
      in : task, target_pdb, binder_length_min, binder_length_max, num_samples, hotspot_residues, target_chain
      out: sample_id, pdb_output (PDB text), sequence, rewards
    """

    def load_context(self, context):
        import os, torch, rfproteina
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from rfproteina.generate import load_ckpt_n_configure_inference
        from rfproteina.datapipes.pipeline import build_design_validation_pipeline

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_float32_matmul_precision("high")
        self._pkg_dir = os.path.dirname(rfproteina.__file__)

        flow, ae = context.artifacts["flow_ckpt"], context.artifacts["ae_ckpt"]
        adapter = context.artifacts.get("adapter_dir")        # the fine-tuned LoRA adapter
        _monomer_cif = os.path.join(self._pkg_dir, "data", "benchmarks", "monomer", "monomer-short.cif")
        overrides = [
            "inference_experiment=inference_on_user_inputs",
            "~metrics/refolding_oracles", "metrics_tags=[]", "num_workers=0", "seed=42",
            f"inputs={_monomer_cif}",                          # satisfies ${inputs}; dataset is driven per-request
            f"ckpt_path={os.path.dirname(flow)}", f"ckpt_name={os.path.basename(flow)}",
            f"autoencoder_ckpt_path={ae}", "out_dir=/tmp/rfd4_out",
        ]
        if adapter:
            overrides.append(f"adapter_path={adapter}")
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=os.path.join(self._pkg_dir, "configs"), version_base="1.3"):
            cfg = compose(config_name="inference", overrides=overrides)
        self.model = load_ckpt_n_configure_inference(cfg)
        self.model.to(self.device).eval()
        self.transform = build_design_validation_pipeline(metrics_tags=None)

    def _request_to_input_path(self, task, pdb_text, length_min, length_max, hotspots, tmpdir):
        import os, json
        if task == "de_novo" or not pdb_text.strip():
            return os.path.join(self._pkg_dir, "data", "benchmarks", "monomer", "monomer-short.cif")
        target = os.path.join(tmpdir, "target.pdb")
        open(target, "w").write(pdb_text)
        contig = f"{int(length_min)}-{int(length_max)}, /0, A"
        conditions = {"C_CRD": {True: [{"select": "sel('A') and is_protein_backbone"}]},
                      "C_SEQ": {True: [{"select": "sel('A')"}]}}
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
        import tempfile, pandas as pd
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
                batch = self.model.transfer_batch_to_device(batch, torch.device(self.device), 0)
                with torch.no_grad():
                    outs = self.model.predict_step(batch, 0)
            for out in outs:
                pdb_str, seq = self._structure_to_pdb_and_seq(out)
                rows.append({"sample_id": sid, "pdb_output": pdb_str, "sequence": seq, "rewards": None}); sid += 1
        return pd.DataFrame(rows)

# COMMAND ----------

# DBTITLE 1,Signature + dry-load test (de novo, short) on this H100 — validates the adapter loads
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

signature = ModelSignature(
    inputs=Schema([ColSpec("string", "task"), ColSpec("string", "target_pdb"),
                   ColSpec("long", "binder_length_min"), ColSpec("long", "binder_length_max"),
                   ColSpec("long", "num_samples"), ColSpec("string", "hotspot_residues"), ColSpec("string", "target_chain")]),
    outputs=Schema([ColSpec("long", "sample_id"), ColSpec("string", "pdb_output"),
                    ColSpec("string", "sequence"), ColSpec("double", "rewards")]))
artifacts = {"flow_ckpt": pretrain_ckpt, "ae_ckpt": ae_ckpt, "adapter_dir": adapter_loc}
_ex = pd.DataFrame([{"task": "de_novo", "target_pdb": "", "binder_length_min": 60, "binder_length_max": 60,
                     "num_samples": 1, "hotspot_residues": "", "target_chain": "A"}])
_m = RFD4ProteinaFineTunedModel(); _m.load_context(PythonModelContext(artifacts=artifacts, model_config={}))
print(_m.predict(None, _ex)[["sample_id", "sequence"]].head())

# COMMAND ----------

# DBTITLE 1,Register to UC — log WITHOUT registered_model_name (Express registers separately)
import os, importlib
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user_email}/dbx_genesis_workbench_modules")
uc_model_name = f"{catalog}.{schema}.{model_name}"

# Ship rfproteina's package source (+ repo siblings) via code_paths so the model carries its configs +
# data/benchmarks. We DON'T pin the heavy CUDA-13 deps via pip_requirements: Express packaging (env_pack,
# two cells down) snapshots this exact serverless-GPU env for serving, so pip_requirements stays
# minimal/trivially-resolvable — env_pack, not requirements.txt, defines the serving environment.
code_paths = []
for _pkg in ("rfproteina", "rfd4_proteina", "script_utils", "gearnet"):
    try:
        code_paths.append(os.path.dirname(importlib.import_module(_pkg).__file__))
    except Exception:
        pass
print("code_paths:", code_paths)

# ~8GB checkpoints + adapter: use the fast UC artifact path so log_model doesn't hit the 5-min upload timeout.
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "false"

# Log WITHOUT registered_model_name — Express requires registering in a SEPARATE register_model() step
# (below) so env_pack can run.
with mlflow.start_run(run_name=f"rfd4_deploy_{ft_label}"):
    model_info = mlflow.pyfunc.log_model(
        name=model_name,
        python_model=RFD4ProteinaFineTunedModel(),
        artifacts=artifacts,
        code_paths=code_paths,
        signature=signature,
        input_example=_ex,
        pip_requirements=["mlflow", "cloudpickle", "pandas", "numpy", "biotite"],
    )
print(f"logged fine-tuned model (ft {ft_label}): {model_info.model_uri}")

# COMMAND ----------

# DBTITLE 1,Express-register (env_pack) — snapshots THIS GPU env for serving (no deploy-time rebuild)
# Express PACKAGES this serverless-GPU/CUDA-13 env at registration and RESTORES it, as-is, at serving — no
# deploy-time container build and no torch force-install. That is what makes the fine-tuned (torch-2.14 /
# CUDA-13) model servable, exactly as for the base model. install_dependencies=False snapshots the installed
# site-packages VERBATIM instead of re-resolving them: our env is full of git/binary/local-tagged packages
# (rfproteina @ git, tmol nvcc build, torch==2.14.1+cu132, cuEquivariance-cu13) that resolve on NO index —
# a re-resolution (the default True) would fail ("No matching distribution for torch==2.14.1+cu132").
# Requires mlflow>=3.12 + databricks-sdk>=0.150.0 (installed in the uv cell) and registering FROM serverless GPU.
import torch
from mlflow.utils.env_pack import EnvPackConfig
assert torch.cuda.is_available(), "env_pack must run on serverless GPU so the packaged env is the CUDA-13 GPU env"

model_version = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=uc_model_name,
    env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False),
).version
print(f"registered {uc_model_name} v{model_version} (fine-tuned ft {ft_label}) with Express env_pack")

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
# update_config_and_wait does NOT touch tags, so (re)apply them with patch() after an update — the
# fine-tuned deploy updates the existing base endpoint, so this keeps it tagged.
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
print(f"endpoint {endpoint_name} is live (fine-tuned ft {ft_label})")

# COMMAND ----------

# DBTITLE 1,Register in the GWB app registry so the UI lists this endpoint (metadata only)
# The endpoint is already live (Express owns it). This resolves the GWB wheel path; the next cell installs
# it (which downgrades mlflow/sdk — fine, serving is already done) and writes app-registry metadata.
gwb_lib = next(l.path.replace("dbfs:", "") for l in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries") if l.name.startswith("genesis_workbench"))
print(gwb_lib)

# COMMAND ----------

# MAGIC %pip install {gwb_lib}
# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Writes GWB app-registry metadata so the UI lists the fine-tuned endpoint. Does NOT call deploy_model —
# Express already created/updated the endpoint above. Re-reads widgets + re-resolves ft_id after the restart
# (which wiped the kernel, so names from pre-restart cells are gone — don't reference them here). Wrapped
# non-fatally: the endpoint (the critical deliverable) is already live, so a registry hiccup must NOT fail
# the task.
import mlflow, traceback
g = dbutils.widgets.get
catalog, schema, model_name = g("catalog"), g("schema"), g("model_name")
user_email, sql_warehouse_id, ft_id = g("user_email"), g("sql_warehouse_id"), g("ft_id")
try:
    from genesis_workbench.models import ModelCategory, import_model_from_uc, get_latest_model_version
    from genesis_workbench.workbench import initialize

    if not ft_id:  # resolve the latest active ft_id for the source_version label (kernel was wiped)
        _r = spark.sql(f"SELECT ft_id FROM {catalog}.{schema}.rfd4_weights WHERE is_active = true ORDER BY ft_id DESC LIMIT 1").collect()
        ft_id = str(_r[0]["ft_id"]) if _r else ""

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
        model_display_name="RFD4-Proteina Design (fine-tuned)",
        model_source_version=f"ft:{ft_id}",
        model_description_url="https://github.com/NVIDIA-BioNeMo/RFD4-Proteina-Open-Model-Private",
    )
    print(f"GWB app-registry model_id={gwb_model_id} (fine-tuned endpoint already live via Express)")
except Exception:
    print("WARNING: GWB app-registry metadata step failed — the endpoint is already live, so NOT failing "
          "the task. Fix separately if the model should appear in the GWB app UI:\n" + traceback.format_exc())

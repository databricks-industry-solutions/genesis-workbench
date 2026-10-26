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

# MAGIC %md
# MAGIC ## 1 · Parameters
# MAGIC `ft_id` selects which fine-tune from the `rfd4_weights` table to deploy (empty = latest active). The
# MAGIC catalog/schema/cache_dir triple + HF repo/revision mirror `01_register`.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("cache_dir", "rfd4_proteina", "Cache dir (UC volume)")
dbutils.widgets.text("rfd4_git_url", "github.com/RosettaCommons/RFD4-Proteina.git", "RFD4 git host/path (public repo)")
dbutils.widgets.text("rfd4_git_ref", "v0.0.1", "RFD4 git ref (release tag/commit)")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse Id")
dbutils.widgets.text("ft_id", "", "Fine-tuned id (empty = latest active)")
dbutils.widgets.text("model_name", "rfd4_proteina", "UC model name")
dbutils.widgets.text("workload_type", "GPU_XLARGE", "Serving workload type (GPU_XLARGE = 1x H100)")
dbutils.widgets.text("rfd4_hf_repo", "nvidia/RFD4-Proteina", "HF repo for weights")
dbutils.widgets.text("rfd4_hf_revision", "a06e2509c7033b010fc43614e9fd840303588502", "HF Hub commit for reproducibility")
dbutils.widgets.text("rfd4_hf_token_scope", "dbx_genesis_workbench", "Secret scope for HF token")
dbutils.widgets.text("rfd4_hf_token_key", "hf_token", "Secret key for HF token")

get_widget = dbutils.widgets.get
catalog, schema, cache_dir = get_widget("catalog"), get_widget("schema"), get_widget("cache_dir")
user_email, sql_warehouse_id = get_widget("user_email"), get_widget("sql_warehouse_id")
git_url, git_ref = get_widget("rfd4_git_url"), get_widget("rfd4_git_ref")
hf_repo, hf_revision = get_widget("rfd4_hf_repo"), get_widget("rfd4_hf_revision")
hf_token_scope, hf_token_key = get_widget("rfd4_hf_token_scope"), get_widget("rfd4_hf_token_key")
ft_id, model_name, workload_type = get_widget("ft_id"), get_widget("model_name"), get_widget("workload_type")
volume_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Install the rfproteina CUDA-13 stack + the Express deploy APIs, then restart
# MAGIC Same CUDA-13 stack as `01_register`, installed from the **public GitHub repo** (nothing vendored;
# MAGIC customer needs nothing local). Also installs `mlflow>=3.12` + `databricks-sdk>=0.150.0` (the Express
# MAGIC `env_pack` API). `--python sys.executable` installs into THIS kernel's env (`/databricks/python`,
# MAGIC writable + surviving the restart — the diffdock serverless pattern); `uv --system` would hit the
# MAGIC read-only `/usr`. `--torch-backend cu132` pins CUDA-13 torch; `--build-constraint` aligns TMol's nvcc
# MAGIC build torch. `restartPython()` makes the just-installed torch 2.14.1/cu132 live so the dry-load +
# MAGIC `env_pack` register FROM this exact env.

# COMMAND ----------

import os, subprocess, sys

# Set HF_TOKEN if available (weights repo may be gated).
try:
    hf_token = dbutils.secrets.get(hf_token_scope, hf_token_key)
    os.environ["HF_TOKEN"] = hf_token
except Exception:
    pass

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
open("/tmp/bc.txt", "w").write("torch==2.14.1\n")
rfd4_package_spec = f"rfd4-proteina[metrics-cuda13] @ git+https://{git_url}@{git_ref}"
install_result = subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--break-system-packages",
                     "--build-constraint", "/tmp/bc.txt", "--torch-backend", "cu132", rfd4_package_spec,
                     "mlflow>=3.12", "databricks-sdk>=0.150.0"],
                    capture_output=True, text=True)
if install_result.returncode != 0:
    install_stderr = (install_result.stderr or "")
    raise RuntimeError(f"uv install failed rc={install_result.returncode}; stderr tail:\n{install_stderr[-3000:]}")
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Re-read parameters after restart + resolve the fine-tune from `rfd4_weights`
# MAGIC `restartPython()` wiped the kernel: re-read the widgets, re-set `HF_TOKEN`, and resolve `ft_id` →
# MAGIC `(ft_label, adapter_volume_location)` from the `rfd4_weights` table. An empty `ft_id` picks the latest
# MAGIC active fine-tune.

# COMMAND ----------

import os
get_widget = dbutils.widgets.get
catalog, schema, cache_dir = get_widget("catalog"), get_widget("schema"), get_widget("cache_dir")
user_email, sql_warehouse_id = get_widget("user_email"), get_widget("sql_warehouse_id")
ft_id, model_name, workload_type = get_widget("ft_id"), get_widget("model_name"), get_widget("workload_type")
hf_repo, hf_revision = get_widget("rfd4_hf_repo"), get_widget("rfd4_hf_revision")
hf_token_scope, hf_token_key = get_widget("rfd4_hf_token_scope"), get_widget("rfd4_hf_token_key")
volume_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"

try:
    hf_token = dbutils.secrets.get(hf_token_scope, hf_token_key)
    os.environ["HF_TOKEN"] = hf_token
except Exception:
    pass

if not ft_id:
    row = spark.sql(f"SELECT ft_id FROM {catalog}.{schema}.rfd4_weights WHERE is_active = true ORDER BY ft_id DESC LIMIT 1").collect()
    assert row, "no active fine-tune found in rfd4_weights — run 02_rfd4_finetune first"
    ft_id = str(row[0]["ft_id"])
meta = spark.sql(f"SELECT ft_label, adapter_volume_location FROM {catalog}.{schema}.rfd4_weights WHERE ft_id = {ft_id} AND is_active = true").collect()
assert meta, f"ft_id {ft_id} not found/active in rfd4_weights"
ft_label, adapter_loc = meta[0]["ft_label"], meta[0]["adapter_volume_location"]
print(f"ft_id={ft_id} label={ft_label} adapter={adapter_loc} hf_repo={hf_repo}@{hf_revision}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · MLflow PyFunc wrapper (base + PEFT adapter) — same contract as `01_register`
# MAGIC Identical wrapper to the base model (`01_register`); the only difference is the LoRA `adapter_dir`
# MAGIC artifact, which `load_ckpt_n_configure_inference` applies via its `adapter_path` override. The base
# MAGIC flow+AE weights are the portable HF bundle (`config.json` + `model.safetensors`), baked into the model
# MAGIC version as a local-dir artifact so serving loads them offline (no HF egress/token on the endpoint). Keep
# MAGIC `configure_inference` (via `load_ckpt_n_configure_inference`) — bypassing it leaves `predict_step` with
# MAGIC no generation config. Model is loaded once and warm.

# COMMAND ----------

import mlflow, pandas as pd, sys
# rfproteina is system-installed from git (§2) — importable directly after restart.
from mlflow.pyfunc import PythonModel, PythonModelContext

class RFD4ProteinaFineTunedModel(PythonModel):
    """Fine-tuned RFD4-Proteina endpoint — identical wrapper to the base model (01_register), the only
    difference being the LoRA `adapter_dir` artifact. Drives inference through the model's own supported
    path (mirrors the model repo's tests/test_inference_integration.py + rfproteina/generate.py), model
    loaded once and warm.

    Serving contract (proteina_complexa-compatible + a `task` column):
      in : task, target_pdb, binder_length_min, binder_length_max, num_samples, hotspot_residues, target_chain
      out: sample_id, pdb_output (PDB text), sequence, rewards
    """

    def load_context(self, context):
        import os, torch
        import rfproteina
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from rfproteina.generate import load_ckpt_n_configure_inference
        from rfproteina.datapipes.pipeline import build_design_validation_pipeline

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_float32_matmul_precision("high")
        self._pkg_dir = os.path.dirname(rfproteina.__file__)

        bundle = context.artifacts["hf_bundle"]               # local base bundle dir, baked at register
        adapter = context.artifacts.get("adapter_dir")        # the fine-tuned LoRA adapter
        monomer_cif_path = os.path.join(self._pkg_dir, "data", "benchmarks", "monomer", "monomer-short.cif")
        overrides = [
            "inference_experiment=inference_on_user_inputs",
            "~metrics/refolding_oracles", "metrics_tags=[]", "num_workers=0", "seed=42",
            f"inputs={monomer_cif_path}",
            f"hf_ckpt_path={bundle}", "out_dir=/tmp/rfd4_out",   # local dir => offline load, revision already pinned
        ]
        if adapter:
            overrides.append(f"adapter_path={adapter}")
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=os.path.join(self._pkg_dir, "configs"), version_base="1.3"):
            cfg = compose(config_name="inference", overrides=overrides)
        self.model = load_ckpt_n_configure_inference(cfg)      # HF from_pretrained + adapter + configure_inference
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
            conditions["C_HOT"] = {True: [{"select": f"sel('A/*/{residue.strip()}')"}
                                          for residue in hotspots.split(",") if residue.strip()]}
        spec_path = os.path.join(tmpdir, "spec.json")
        open(spec_path, "w").write(json.dumps({task: {"input": target, "contig": contig, "conditions": conditions}}))
        return spec_path

    def _structure_to_pdb_and_seq(self, sample):
        import io, numpy as np
        atom_array = sample["generated_atom_array"]
        seq = ""
        try:
            from biotite.structure import to_sequence
            seqs, _ = to_sequence(atom_array)
            seq = "".join(str(s) for s in seqs)
        except Exception:
            pass
        annotation_categories = set(atom_array.get_annotation_categories())
        n_atoms = atom_array.array_length()
        if "b_factor" in annotation_categories:
            atom_array.set_annotation("b_factor", np.zeros(n_atoms))
        if "occupancy" in annotation_categories:
            atom_array.set_annotation("occupancy", np.ones(n_atoms))
        if "charge" in annotation_categories:
            atom_array.set_annotation("charge", np.zeros(n_atoms, dtype=int))
        try:
            from biotite.structure.io.pdb import PDBFile
            buffer = io.StringIO()
            pdb_file = PDBFile()
            pdb_file.set_structure(atom_array)
            pdb_file.write(buffer)
            return buffer.getvalue(), seq
        except Exception:
            from biotite.structure.io.pdbx import CIFFile, set_structure
            cif_file = CIFFile()
            set_structure(cif_file, atom_array)
            buffer = io.StringIO()
            cif_file.write(buffer)
            return buffer.getvalue(), seq

    def predict(self, context, model_input, params=None):
        import tempfile, pandas as pd
        from rfproteina.generate import get_contig_or_design_problem_dataset
        from rfproteina.datapipes.collate import collate_batch
        torch = self._torch
        rows, sample_id = [], 0
        for _, row in model_input.iterrows():
            task = str(row.get("task", "de_novo") or "de_novo")
            num_samples = int(row.get("num_samples", 1) or 1)
            with tempfile.TemporaryDirectory() as tmp:
                inputs = self._request_to_input_path(
                    task, str(row.get("target_pdb", "") or ""),
                    row.get("binder_length_min", 100), row.get("binder_length_max", 100),
                    str(row.get("hotspot_residues", "") or ""), tmp)
                dataset = get_contig_or_design_problem_dataset(inputs, num_replicates=num_samples, transform=self.transform)
                batch = collate_batch([dataset[i] for i in range(len(dataset))],
                                      schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample",
                                      fixed_dimensions=None)
                batch = self.model.transfer_batch_to_device(batch, torch.device(self.device), 0)
                with torch.no_grad():
                    outputs = self.model.predict_step(batch, 0)
            for output in outputs:
                pdb_str, seq = self._structure_to_pdb_and_seq(output)
                rows.append({"sample_id": sample_id, "pdb_output": pdb_str, "sequence": seq, "rewards": None}); sample_id += 1
        return pd.DataFrame(rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Signature + dry-load test (de novo, short) on this H100 — validates the adapter loads
# MAGIC Build the serving signature, download + bake the base HF bundle (`config.json` + `model.safetensors`)
# MAGIC plus the fine-tuned LoRA `adapter_dir` into the artifacts, and run one short de-novo prediction to prove
# MAGIC the adapter loads. (MLflow copies every artifacts *value* as a path — never pass the HF repo id here or
# MAGIC it is mis-resolved as an artifact URI; always bake a real local dir.)

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

signature = ModelSignature(
    inputs=Schema([ColSpec("string", "task"), ColSpec("string", "target_pdb"),
                   ColSpec("long", "binder_length_min"), ColSpec("long", "binder_length_max"),
                   ColSpec("long", "num_samples"), ColSpec("string", "hotspot_residues"), ColSpec("string", "target_chain")]),
    outputs=Schema([ColSpec("long", "sample_id"), ColSpec("string", "pdb_output"),
                    ColSpec("string", "sequence"), ColSpec("double", "rewards")]))
import os
from huggingface_hub import snapshot_download
bundle_dir = snapshot_download(repo_id=hf_repo, revision=hf_revision, token=os.environ.get("HF_TOKEN") or None,
                               allow_patterns=["config.json", "model.safetensors"], local_dir="/tmp/rfd4_bundle")
print("bundle_dir:", bundle_dir, "contents:", sorted(os.listdir(bundle_dir)))
artifacts = {"hf_bundle": bundle_dir, "adapter_dir": adapter_loc}
example_input = pd.DataFrame([{"task": "de_novo", "target_pdb": "", "binder_length_min": 60, "binder_length_max": 60,
                               "num_samples": 1, "hotspot_residues": "", "target_chain": "A"}])
test_model = RFD4ProteinaFineTunedModel(); test_model.load_context(PythonModelContext(artifacts=artifacts, model_config={}))
print(test_model.predict(None, example_input)[["sample_id", "sequence"]].head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Log + register in Unity Catalog — log WITHOUT registered_model_name (Express registers separately)
# MAGIC Ship `rfproteina`'s package source (+ repo siblings) via `code_paths` so the model carries its configs +
# MAGIC `data/benchmarks`. We don't pin the heavy CUDA-13 deps via `pip_requirements`: Express packaging
# MAGIC (`env_pack`, §7) snapshots this exact serverless-GPU env for serving, so `pip_requirements` stays
# MAGIC minimal/trivially-resolvable — `env_pack`, not `requirements.txt`, defines the serving environment. The
# MAGIC new release uses HF Hub for weights (`Proteina.from_pretrained`), so there are no large checkpoints to
# MAGIC upload — only the adapter directory is in `artifacts`. Log WITHOUT `registered_model_name` — Express
# MAGIC requires registering in a separate `register_model()` step (§7) so `env_pack` can run.

# COMMAND ----------

import os, importlib
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user_email}/dbx_genesis_workbench_modules")
uc_model_name = f"{catalog}.{schema}.{model_name}"

code_paths = []
for package_name in ("rfproteina", "rfd4_proteina", "script_utils", "gearnet"):
    try:
        code_paths.append(os.path.dirname(importlib.import_module(package_name).__file__))
    except Exception:
        pass
print("code_paths:", code_paths)

with mlflow.start_run(run_name=f"rfd4_deploy_{ft_label}"):
    model_info = mlflow.pyfunc.log_model(
        name=model_name,
        python_model=RFD4ProteinaFineTunedModel(),
        artifacts=artifacts,
        code_paths=code_paths,
        signature=signature,
        input_example=example_input,
        pip_requirements=["mlflow", "cloudpickle", "pandas", "numpy", "biotite"],
    )
print(f"logged fine-tuned model (ft {ft_label}): {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Express-register (env_pack) — snapshots THIS GPU env for serving (no deploy-time rebuild)
# MAGIC Express PACKAGES this serverless-GPU/CUDA-13 env at registration and RESTORES it, as-is, at serving —
# MAGIC no deploy-time container build and no torch force-install. That is what makes the fine-tuned (torch-2.14
# MAGIC / CUDA-13) model servable, exactly as for the base model. `install_dependencies=False` snapshots the
# MAGIC installed site-packages verbatim instead of re-resolving them: our env is full of git/binary/local-tagged
# MAGIC packages (`rfproteina @ git`, tmol nvcc build, `torch==2.14.1+cu132`, `cuEquivariance-cu13`) that resolve
# MAGIC on NO index — a re-resolution (the default `True`) would fail (`No matching distribution for
# MAGIC torch==2.14.1+cu132`). Requires `mlflow>=3.12` + `databricks-sdk>=0.150.0` (§2) and registering FROM
# MAGIC serverless GPU.

# COMMAND ----------

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

# MAGIC %md
# MAGIC ## 8 · Express-deploy the serving endpoint — the packaged GPU env is restored as-is (no rebuild)
# MAGIC Update (or create) the shared `gwb_<prefix>_rfd4_proteina_endpoint` with this fine-tuned version on
# MAGIC GPU_XLARGE (1× H100; no scale-to-zero). `update_config_and_wait` does NOT touch tags, so we (re)apply the
# MAGIC `application`/`created_by` tags with `patch()` after an update to keep the endpoint tagged.

# COMMAND ----------

from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (EndpointCoreConfigInput, ServedEntityInput,
                                            ServingModelWorkloadType, EndpointTag)

w = WorkspaceClient()
try:
    dev_user_prefix = dbutils.secrets.get("dbx_genesis_workbench", "dev_user_prefix")
except Exception:
    dev_user_prefix = ""
endpoint_name = f"gwb_{dev_user_prefix}_{model_name}_endpoint" if dev_user_prefix and dev_user_prefix.strip() else f"gwb_{model_name}_endpoint"

served = [ServedEntityInput(
    entity_name=uc_model_name, entity_version=str(model_version), name=model_name,
    workload_type=ServingModelWorkloadType(workload_type),   # GPU_XLARGE = 1x H100 (us-west-2, enrolled)
    workload_size="Small", scale_to_zero_enabled=False,       # GPU_XLARGE does not support scale-to-zero
)]
endpoint_tags = [EndpointTag(key="application", value="genesis_workbench"),
                 EndpointTag(key="created_by", value=user_email)]
print(f"express-deploying {endpoint_name}: {uc_model_name} v{model_version} on {workload_type}")

endpoint_exists = True
try:
    w.serving_endpoints.get(endpoint_name)
except Exception:
    endpoint_exists = False
if endpoint_exists:
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

# MAGIC %md
# MAGIC ## 9 · Register in the GWB app registry so the UI lists this endpoint (metadata only)
# MAGIC The endpoint is already live (Express owns it). This cell resolves the GWB wheel path; the next cell
# MAGIC installs it (which downgrades mlflow/sdk — fine, serving is already done) and writes app-registry
# MAGIC metadata.

# COMMAND ----------

gwb_lib = next(l.path.replace("dbfs:", "") for l in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries") if l.name.startswith("genesis_workbench"))
print(gwb_lib)

# COMMAND ----------

# MAGIC %pip install {gwb_lib}
# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 · Write the GWB app-registry metadata (non-fatal)
# MAGIC Writes app-registry metadata so the UI lists the fine-tuned endpoint. Does NOT call `deploy_model` —
# MAGIC Express already created/updated the endpoint. Re-reads widgets + re-resolves `ft_id` after the restart
# MAGIC (the kernel was wiped, so names from pre-restart cells are gone). Wrapped non-fatally: the endpoint (the
# MAGIC critical deliverable) is already live, so a registry hiccup must NOT fail the task.

# COMMAND ----------

import mlflow, traceback
get_widget = dbutils.widgets.get
catalog, schema, model_name = get_widget("catalog"), get_widget("schema"), get_widget("model_name")
user_email, sql_warehouse_id, ft_id = get_widget("user_email"), get_widget("sql_warehouse_id"), get_widget("ft_id")
try:
    from genesis_workbench.models import ModelCategory, import_model_from_uc, get_latest_model_version
    from genesis_workbench.workbench import initialize

    if not ft_id:  # resolve the latest active ft_id for the source_version label (kernel was wiped)
        active_row = spark.sql(f"SELECT ft_id FROM {catalog}.{schema}.rfd4_weights WHERE is_active = true ORDER BY ft_id DESC LIMIT 1").collect()
        ft_id = str(active_row[0]["ft_id"]) if active_row else ""

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
        model_description_url="https://github.com/RosettaCommons/RFD4-Proteina",
    )
    print(f"GWB app-registry model_id={gwb_model_id} (fine-tuned endpoint already live via Express)")
except Exception:
    print("WARNING: GWB app-registry metadata step failed — the endpoint is already live, so NOT failing "
          "the task. Fix separately if the model should appear in the GWB app UI:\n" + traceback.format_exc())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11 · Grant the app SP(s) CAN_QUERY on the (re)deployed endpoint
# MAGIC The Express create/update path above does NOT grant the app SPs (unlike the standard `deploy_model`
# MAGIC path, which calls `set_app_permissions_for_endpoint`), so without this a fine-tuned redeploy can leave
# MAGIC the app's models/settings page showing `PermissionDenied ... 'View' on Endpoint
# MAGIC gwb_*_rfd4_proteina_endpoint`. Independent + non-fatal (the endpoint is already live).
# MAGIC `DATABRICKS_APP_NAMES` defaults to the UI + MCP app pair; the helper try/except-skips an app that isn't
# MAGIC deployed.

# COMMAND ----------

import os
try:
    from genesis_workbench.workbench import set_app_permissions_for_endpoint
    model_name_widget = dbutils.widgets.get("model_name")
    os.environ.setdefault("DATABRICKS_APP_NAMES", "genesis-workbench,mcp-genesis-workbench")
    try:
        dev_user_prefix = dbutils.secrets.get("dbx_genesis_workbench", "dev_user_prefix")
    except Exception:
        dev_user_prefix = ""
    endpoint_name = f"gwb_{dev_user_prefix}_{model_name_widget}_endpoint" if dev_user_prefix and dev_user_prefix.strip() else f"gwb_{model_name_widget}_endpoint"
    set_app_permissions_for_endpoint(endpoint_name)
    print(f"granted app SP(s) CAN_QUERY on {endpoint_name}")
except Exception as error:
    print(f"WARNING: could not grant app perms on the endpoint: {error}")

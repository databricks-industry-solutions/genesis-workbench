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

# MAGIC %md
# MAGIC ## 1 · Install base tooling
# MAGIC Install the lightweight SDK/MLflow clients and restart so they are live in-kernel. The heavy
# MAGIC `rfproteina` / CUDA-13 stack is installed separately (via `uv`) in §3.

# COMMAND ----------

# MAGIC %pip install -q "databricks-sdk>=0.50.0" "databricks-sql-connector>=4.0.2" "mlflow>=2.15"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Parameters
# MAGIC Every input is a notebook widget so the registration job (and a human re-run) can override it. The
# MAGIC `catalog`/`schema`/`cache_dir` triple resolves the UC Volume the weights are cached in; the HF repo +
# MAGIC revision pin the exact weight snapshot; `workload_type=GPU_XLARGE` is a single H100.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "rfd4_proteina", "Model Name")
dbutils.widgets.text("cache_dir", "rfd4_proteina", "Cache dir (UC volume)")
dbutils.widgets.text("rfd4_git_url", "github.com/RosettaCommons/RFD4-Proteina.git", "RFD4 git host/path (public repo)")
dbutils.widgets.text("rfd4_git_ref", "v0.0.1", "RFD4 git ref (release tag/commit)")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "MLflow experiment")
dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("workload_type", "GPU_XLARGE", "Serving workload type (GPU_XLARGE = 1x H100)")
dbutils.widgets.text("rfd4_hf_repo", "nvidia/RFD4-Proteina", "HF repo for weights (matched flow + AE pair)")
dbutils.widgets.text("rfd4_hf_revision", "a06e2509c7033b010fc43614e9fd840303588502", "HF Hub commit for weight reproducibility")
dbutils.widgets.text("rfd4_hf_token_scope", "dbx_genesis_workbench", "Secret scope for HF token")
dbutils.widgets.text("rfd4_hf_token_key", "hf_token", "Secret key for HF token")
dbutils.widgets.text("rfd4_flow_ckpt", "rfd4-proteina-flow-release-2026-10-19.ckpt", "Flow checkpoint filename in HF repo")
dbutils.widgets.text("rfd4_ae_ckpt", "rfd4-proteina-ae-release-2026-10-19.ckpt", "AE checkpoint filename in HF repo")

get_widget = dbutils.widgets.get
catalog, schema, model_name, cache_dir = get_widget("catalog"), get_widget("schema"), get_widget("model_name"), get_widget("cache_dir")
experiment_name = get_widget("experiment_name")
sql_warehouse_id, user_email, workload_type = get_widget("sql_warehouse_id"), get_widget("user_email"), get_widget("workload_type")
hf_repo, hf_revision, flow_ckpt, ae_ckpt = get_widget("rfd4_hf_repo"), get_widget("rfd4_hf_revision"), get_widget("rfd4_flow_ckpt"), get_widget("rfd4_ae_ckpt")
hf_token_scope, hf_token_key = get_widget("rfd4_hf_token_scope"), get_widget("rfd4_hf_token_key")
git_url, git_ref = get_widget("rfd4_git_url"), get_widget("rfd4_git_ref")

volume_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"
print(f"catalog={catalog} schema={schema} vol_root={volume_root}\nrfd4_git={git_url}@{git_ref} hf_repo={hf_repo}@{hf_revision} workload_type={workload_type}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Install the rfproteina CUDA-13 stack
# MAGIC This is the make-or-break step. AI Runtime v6 (`client '6'`) is CUDA 13, so the `cu132` wheels and the
# MAGIC `TMol` nvcc build resolve against the base toolkit. We install `rfproteina` from the **public GitHub
# MAGIC repo** (nothing is vendored into GWB, and a customer needs nothing cloned locally), pinned to a tag for
# MAGIC reproducibility — installing from the real repo means every build file (README/LICENSE/`.project-root`/
# MAGIC data) is present, so the hatchling build just works.
# MAGIC
# MAGIC Install details: `uv` handles the `TMol` build isolation. `--python sys.executable` targets THIS
# MAGIC kernel's env (`/databricks/python`, writable and surviving the restart below — the diffdock serverless
# MAGIC pattern), so `rfproteina`'s deps (e.g. a newer `typing_extensions` the base runtime lacks) overwrite
# MAGIC in-place and are live after `restartPython`. `uv --system` would wrongly target the read-only `/usr`.
# MAGIC `--torch-backend cu132` pins CUDA-13 torch; `--build-constraint` aligns TMol's nvcc build torch. We
# MAGIC also install `mlflow>=3.12` + `databricks-sdk>=0.150.0` (the Express env_pack API + the GPU_XLARGE
# MAGIC enum). The full uv log is written to the Volume (readable via `fs cat`) and its stderr tail is embedded
# MAGIC in the raised error so the failure cause is visible via `get-run-output`.

# COMMAND ----------

import os, subprocess, sys

# Set HF_TOKEN from the secret scope (the weights repo may be gated). Harmless no-op if the repo is public.
try:
    hf_token = dbutils.secrets.get(hf_token_scope, hf_token_key)
    os.environ["HF_TOKEN"] = hf_token
except Exception:
    pass

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
build_constraint_path = "/tmp/rfd4_build_constraint.txt"
open(build_constraint_path, "w").write("torch==2.14.1\n")
rfd4_package_spec = f"rfd4-proteina[metrics-cuda13] @ git+https://{git_url}@{git_ref}"
install_result = subprocess.run(
    ["uv", "pip", "install", "--python", sys.executable, "--break-system-packages",
     "--build-constraint", build_constraint_path, "--torch-backend", "cu132", rfd4_package_spec,
     "mlflow>=3.12", "databricks-sdk>=0.150.0"],
    capture_output=True, text=True,
)
os.makedirs(f"{volume_root}/register_logs", exist_ok=True)
install_log = (install_result.stdout or "") + "\n==STDERR==\n" + (install_result.stderr or "")
open(f"{volume_root}/register_logs/uv_install.log", "w").write(install_log)
print(install_log[-4000:])
if install_result.returncode != 0:
    raise RuntimeError(f"uv install failed rc={install_result.returncode}; stderr tail:\n{install_log[-3000:]}")
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Re-read parameters after restart
# MAGIC `restartPython()` wiped the kernel, so re-read the widgets and re-set `HF_TOKEN` for the weight
# MAGIC download below (the repo may be gated; `Proteina.from_pretrained` uses `HF_TOKEN` to authenticate).

# COMMAND ----------

import os
get_widget = dbutils.widgets.get
catalog, schema, model_name, cache_dir = get_widget("catalog"), get_widget("schema"), get_widget("model_name"), get_widget("cache_dir")
sql_warehouse_id, user_email, workload_type = get_widget("sql_warehouse_id"), get_widget("user_email"), get_widget("workload_type")
experiment_name = get_widget("experiment_name")
hf_repo, hf_revision = get_widget("rfd4_hf_repo"), get_widget("rfd4_hf_revision")
hf_token_scope, hf_token_key = get_widget("rfd4_hf_token_scope"), get_widget("rfd4_hf_token_key")
volume_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"

try:
    hf_token = dbutils.secrets.get(hf_token_scope, hf_token_key)
    os.environ["HF_TOKEN"] = hf_token
except Exception:
    pass
print(f"HF_TOKEN: {'set' if 'HF_TOKEN' in os.environ else 'not set'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · MLflow PyFunc wrapper
# MAGIC The wrapper drives RFD4 exactly as `rfd4-generate` does (warm, in-process), **not** a hand-rolled batch:
# MAGIC
# MAGIC - **`load_context`** (once, warm): compose the Hydra inference config → `load_ckpt_n_configure_inference()`
# MAGIC   loads the flow + AE checkpoints (optionally applies a LoRA adapter) and sets the sampling config →
# MAGIC   build the inference transform pipeline. The model stays resident for low-latency serving.
# MAGIC - **`predict`** (per request): turn the row into the model's own input via
# MAGIC   `get_contig_or_design_problem_dataset()` (a CIF path → `DesignProblemDataset`; a `condition_spec`
# MAGIC   `.json` → `ConditionSpecDataset`) → `collate_batch` → `predict_step` → designed structure + sequence.
# MAGIC
# MAGIC Because Express packages THIS serverless-GPU/CUDA-13 env at registration and restores it verbatim at
# MAGIC serving, `rfproteina` imports and loads exactly as it does on this register node — no base-torch compat
# MAGIC shims. The matched flow+AE weights are a portable HF bundle (`config.json` + `model.safetensors`) BAKED
# MAGIC into the model version at register time and handed to `load_context` as a local-dir artifact, so serving
# MAGIC loads them offline (no HF egress or token on the endpoint). The `inputs` config key is mandatory
# MAGIC (`generation.dataset` interpolates `${inputs}`); we drive the dataset per-request in `predict` instead,
# MAGIC so `inputs` points at the bundled monomer CIF purely to satisfy config resolution.
# MAGIC
# MAGIC Modes: `task="de_novo"` (or empty `target_pdb`) → length-only monomer generation off the bundled monomer
# MAGIC scaffold. `task in {"binder","motif"}` → build a `condition_spec` from `target_pdb` + a contig + hotspot
# MAGIC selects (first-draft conditioning, refined in the design workflows).

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel, PythonModelContext


class RFD4ProteinaModel(PythonModel):
    """RFD4-Proteina design endpoint.

    Inference runs through the model's OWN supported path (mirrors the model repo's
    tests/test_inference_integration.py + rfproteina/generate.py), NOT a hand-rolled batch.

    Serving contract (proteina_complexa-compatible + a `task` column):
      in : task, target_pdb, binder_length_min, binder_length_max, num_samples, hotspot_residues, target_chain
      out: sample_id, pdb_output (PDB text), sequence, rewards
    """

    def load_context(self, context):
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

        bundle = context.artifacts["hf_bundle"]                # local bundle dir, baked at register
        adapter = context.artifacts.get("adapter_dir")         # set only for a fine-tuned deploy (nb 03)
        monomer_cif_path = os.path.join(self._pkg_dir, "data", "benchmarks", "monomer", "monomer-short.cif")
        # Keep configure_inference (via load_ckpt_n_configure_inference); bypassing it (bare from_pretrained)
        # leaves predict_step with no generation config. A local hf_ckpt_path dir => offline load.
        overrides = [
            "inference_experiment=inference_on_user_inputs",
            "~metrics/refolding_oracles",                      # no AF3 refolding oracle at serving time
            "metrics_tags=[]", "num_workers=0", "seed=42",
            f"inputs={monomer_cif_path}",
            f"hf_ckpt_path={bundle}", "out_dir=/tmp/rfd4_out",   # local dir => offline load, revision already pinned
        ]
        if adapter:
            overrides.append(f"adapter_path={adapter}")
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=os.path.join(self._pkg_dir, "configs"), version_base="1.3"):
            cfg = compose(config_name="inference", overrides=overrides)
        self.model = load_ckpt_n_configure_inference(cfg)      # HF from_pretrained + configure_inference
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
        # The generated array's b_factor/occupancy/charge can hold out-of-range (even non-finite) values that
        # overflow PDB's fixed column widths; reset them to safe defaults for a clean PDB.
        annotation_categories = set(atom_array.get_annotation_categories())
        n_atoms = atom_array.array_length()
        if "b_factor" in annotation_categories:
            atom_array.set_annotation("b_factor", np.zeros(n_atoms))
        if "occupancy" in annotation_categories:
            atom_array.set_annotation("occupancy", np.ones(n_atoms))
        if "charge" in annotation_categories:
            atom_array.set_annotation("charge", np.zeros(n_atoms, dtype=int))
        # Fall back to mmCIF (no column limits) if PDB is still incompatible.
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
        import pandas as pd
        import tempfile
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
                # Use the model's OWN device transfer (what the Trainer calls) — the batch holds structured
                # containers a plain tensor-recursion misses, which left inputs on CPU.
                batch = self.model.transfer_batch_to_device(batch, torch.device(self.device), 0)
                with torch.no_grad():
                    outputs = self.model.predict_step(batch, 0)
            for output in outputs:
                pdb_str, seq = self._structure_to_pdb_and_seq(output)
                rows.append({"sample_id": sample_id, "pdb_output": pdb_str, "sequence": seq, "rewards": None}); sample_id += 1
        return pd.DataFrame(rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Signature + dry-load test
# MAGIC Build the serving signature and run one short de-novo prediction on this H100 **before** logging — a
# MAGIC `long` (int64) schema matches what pandas produces for Python ints so the `input_example` and the
# MAGIC `sample_id` output pass MLflow schema enforcement. The weights are downloaded here (into a local dir)
# MAGIC and reused by the register cell below.

# COMMAND ----------

import sys, pandas as pd
# rfproteina is system-installed from git (§3) — importable directly after restart.
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

input_schema = Schema([
    ColSpec("string", "task"), ColSpec("string", "target_pdb"),
    ColSpec("long", "binder_length_min"), ColSpec("long", "binder_length_max"),
    ColSpec("long", "num_samples"), ColSpec("string", "hotspot_residues"), ColSpec("string", "target_chain"),
])
output_schema = Schema([ColSpec("long", "sample_id"), ColSpec("string", "pdb_output"),
                        ColSpec("string", "sequence"), ColSpec("double", "rewards")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Bake the matched flow+AE weights (portable HF bundle: config.json + model.safetensors) into a local dir so
# serving loads them OFFLINE — no HF egress or token on the endpoint. HF_TOKEN is passed only if present.
from huggingface_hub import snapshot_download
bundle_dir = snapshot_download(repo_id=hf_repo, revision=hf_revision, token=os.environ.get("HF_TOKEN") or None,
                               allow_patterns=["config.json", "model.safetensors"], local_dir="/tmp/rfd4_bundle")
print("bundle_dir:", bundle_dir, "contents:", sorted(os.listdir(bundle_dir)))
artifacts = {"hf_bundle": bundle_dir}
test_model = RFD4ProteinaModel(); test_model.load_context(PythonModelContext(artifacts=artifacts, model_config={}))
example_input = pd.DataFrame([{"task": "de_novo", "target_pdb": "", "binder_length_min": 60,
                               "binder_length_max": 60, "num_samples": 1, "hotspot_residues": "", "target_chain": "A"}])
print(test_model.predict(None, example_input)[["sample_id", "sequence"]].head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Log + register in Unity Catalog
# MAGIC Ship `rfproteina`'s package source (+ repo siblings) via `code_paths` so the model carries its configs +
# MAGIC `data/benchmarks`. We deliberately do **not** pin the heavy CUDA-13 deps via `pip_requirements`: Express
# MAGIC packaging (`env_pack`, §7-Express) snapshots this exact serverless-GPU env for serving, so there is no
# MAGIC pip re-resolution at deploy time and the old local-tag gymnastics (`torch==2.14.1+cu132`, the TMol source
# MAGIC build, the NVIDIA index, the dry-run guard) are gone. `pip_requirements` stays minimal/trivially
# MAGIC resolvable — `env_pack`, not `requirements.txt`, defines the serving environment.
# MAGIC
# MAGIC The local HF bundle dir (downloaded in §6) is baked into the model version as an MLflow artifact so
# MAGIC serving loads the weights offline. **Note:** MLflow treats every artifacts *value* as a path/URI to copy
# MAGIC in — passing the HF repo id (`nvidia/RFD4-Proteina`) here would make MLflow try to resolve it as an
# MAGIC artifact URI and fail (`No such artifact`). Always bake a real local dir. We log **without**
# MAGIC `registered_model_name` — Express requires registering in a separate `register_model()` step (§7-Express)
# MAGIC so `env_pack` can run.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user_email}/{experiment_name}")
uc_model_name = f"{catalog}.{schema}.{model_name}"

import importlib
code_paths = []
for package_name in ("rfproteina", "rfd4_proteina", "script_utils", "gearnet"):
    try:
        code_paths.append(os.path.dirname(importlib.import_module(package_name).__file__))
    except Exception:
        pass
print("code_paths:", code_paths)

artifacts = {"hf_bundle": bundle_dir}

with mlflow.start_run(run_name="rfd4_proteina_register"):
    model_info = mlflow.pyfunc.log_model(
        name=model_name,
        python_model=RFD4ProteinaModel(),
        artifacts=artifacts,
        code_paths=code_paths,
        signature=signature,
        input_example=example_input,
        pip_requirements=["mlflow", "cloudpickle", "pandas", "numpy", "biotite"],
    )
print(f"logged model: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7-Express · Express registration (env_pack) — snapshots THIS GPU env for serving
# MAGIC Express deployments (docs: `machine-learning/model-serving/express-deployments`) **package** this
# MAGIC serverless-GPU environment at registration and **restore** it, as-is, at serving — there is NO
# MAGIC deploy-time container build and NO torch force-install. That is the entire reason RFD4 can be served:
# MAGIC standard GPU serving rebuilds on a torch-2.7.1 / CUDA-11.8 base (a hardcoded post-step force-reinstalls
# MAGIC torch 2.7.1+cu118), which the torch-2.14 / CUDA-13 `rfproteina` stack cannot run (`has_static_value` /
# MAGIC `torch.optim.Muon` / `libnvrtc.so.13` all fail). With `env_pack` the serving container IS this env, so
# MAGIC the model loads + generates exactly as on this register node. Requires `mlflow>=3.12` +
# MAGIC `databricks-sdk>=0.102.0` (installed in §3) and that we register FROM serverless GPU.
# MAGIC
# MAGIC `install_dependencies=False` snapshots the already-installed site-packages verbatim instead of
# MAGIC re-resolving them from an index. Our env is full of git/binary/local-tagged packages (`rfproteina @
# MAGIC git`, `tmol` nvcc build, `torch==2.14.1+cu132`, `cuEquivariance-cu13`) that resolve on NO index — a
# MAGIC re-resolution (the default `True`) would fail like the old serving build (`No matching distribution for
# MAGIC torch==2.14.1+cu132`). `False` packages the exact working bits (serving egress is restricted anyway).

# COMMAND ----------

import torch
from mlflow.utils.env_pack import EnvPackConfig
assert torch.cuda.is_available(), "env_pack must run on serverless GPU so the packaged env is the CUDA-13 GPU env"

model_version = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=uc_model_name,
    env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False),
).version
print(f"registered {uc_model_name} v{model_version} with Express env_pack (GPU env snapshotted for serving)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Deploy the serving endpoint (Express) — the packaged GPU env is restored as-is
# MAGIC Create (or update, if it already exists) the `gwb_<prefix>_rfd4_proteina_endpoint` endpoint on
# MAGIC GPU_XLARGE (1× H100; scale-to-zero is unsupported there). The `application`/`created_by` tags match
# MAGIC `deploy_model_endpoint`'s convention — `create` carries them inline, but `update_config_and_wait` does
# MAGIC NOT touch tags, so we (re)apply them with `patch()` after an update or every redeploy would drop them.

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
print(f"endpoint {endpoint_name} is live")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 · Register in the GWB app registry so the UI lists this endpoint (metadata only)
# MAGIC The endpoint is already live (Express owns it). This cell resolves the GWB wheel path; the next cell
# MAGIC installs it (which downgrades mlflow/sdk — fine, serving is already done) and writes the app-registry
# MAGIC metadata. The metadata write is wrapped non-fatally so a registry hiccup does not fail the task and
# MAGIC trigger a wasteful full-notebook retry (re-install + re-env_pack + re-deploy).

# COMMAND ----------

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

# Re-reads widgets after the restart (which wiped the kernel, so names from pre-restart cells like
# `endpoint_name` are gone — don't reference them here). Non-fatal: the endpoint (the critical deliverable)
# is already live, so a registry hiccup must NOT fail the task.
import mlflow, traceback
get_widget = dbutils.widgets.get
catalog, schema, model_name = get_widget("catalog"), get_widget("schema"), get_widget("model_name")
user_email, sql_warehouse_id = get_widget("user_email"), get_widget("sql_warehouse_id")
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
        model_source_version="v0.0.1",
        model_description_url="https://github.com/RosettaCommons/RFD4-Proteina",
    )
    print(f"GWB app-registry model_id={gwb_model_id} ({model_name} endpoint already live via Express)")
except Exception:
    print("WARNING: GWB app-registry metadata step failed — the endpoint is already live, so NOT failing "
          "the task. Fix separately if the model should appear in the GWB app UI:\n" + traceback.format_exc())

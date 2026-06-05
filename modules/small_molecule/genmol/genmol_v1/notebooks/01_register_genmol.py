# Databricks notebook source

# MAGIC %md
# MAGIC ### GenMol — generative small-molecule design
# MAGIC
# MAGIC GenMol (NVIDIA) is a masked-diffusion model over **SAFE** (Sequential
# MAGIC Attachment-based Fragment Embedding) strings. It is a *generalist* generator
# MAGIC for de novo generation, scaffold decoration/morphing, linker design, motif
# MAGIC extension, hit generation and lead optimization.
# MAGIC
# MAGIC This notebook:
# MAGIC  1. downloads the open weights `nvidia/NV-GenMol-89M-v2` from Hugging Face,
# MAGIC  2. wraps generation in an MLflow PyFunc (SMILES/SAFE seed + params → SMILES),
# MAGIC  3. registers the model in Unity Catalog,
# MAGIC  4. imports it into Genesis Workbench (SMALL_MOLECULE) and deploys an endpoint.
# MAGIC
# MAGIC It closes the small-molecule "where do candidate ligands come from?" gap:
# MAGIC **GenMol generates** candidates → DiffDock docks them into the target → KERMT/
# MAGIC Chemprop profile ADMET.
# MAGIC
# MAGIC **Licensing:** weights are under the NVIDIA Open Model License (commercial use
# MAGIC permitted; not for life-critical use cases); the GenMol code is Apache-2.0.

# COMMAND ----------

dbutils.widgets.text("catalog", "srijit_nair_ci_demo_catalog", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("model_name", "genmol", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "045df48d4afed522", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "genmol", "Cache dir")
dbutils.widgets.text("workload_type", "GPU_SMALL", "Workload Type for endpoints")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installing dependencies

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3
# MAGIC %pip install torch==2.7.1 huggingface_hub==0.27.1 safe-mol==0.1.13 rdkit==2025.3.6
# MAGIC # GenMol generation/decoding code (Apache-2.0). The weights are pulled
# MAGIC # separately from HF below; this package supplies the Sampler.
# MAGIC %pip install git+https://github.com/NVIDIA-Digital-Bio/GenMol.git
# MAGIC %pip install mlflow[databricks]==2.22.0

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if(lib.name.startswith("genesis_workbench")):
        gwb_library_path = lib.path.replace("dbfs:","")

print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
experiment_name = dbutils.widgets.get("experiment_name")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
cache_dir = dbutils.widgets.get("cache_dir")
workload_type = dbutils.widgets.get("workload_type")

cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"
print(f"Cache full path: {cache_full_path}")

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{cache_dir}")

# COMMAND ----------

# Initialize Genesis Workbench
from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download the open GenMol checkpoint from Hugging Face

# COMMAND ----------

import os
from huggingface_hub import snapshot_download

HF_REPO = "nvidia/NV-GenMol-89M-v2"
weights_dir = os.path.join(cache_full_path, "weights")
os.makedirs(weights_dir, exist_ok=True)

snapshot_download(repo_id=HF_REPO, local_dir=weights_dir)
print("Downloaded GenMol weights to:", weights_dir)
print(os.listdir(weights_dir))

# The repo ships the diffusion checkpoint (model.ckpt / model_v2.ckpt). Resolve
# whichever checkpoint file is present so the artifact path is deterministic.
ckpt_candidates = [f for f in os.listdir(weights_dir) if f.endswith(".ckpt")]
assert ckpt_candidates, f"No .ckpt found in {weights_dir}: {os.listdir(weights_dir)}"
checkpoint_path = os.path.join(weights_dir, sorted(ckpt_candidates)[-1])
print("Using checkpoint:", checkpoint_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wrap GenMol generation in an MLflow PyFunc
# MAGIC
# MAGIC Input: a single-column DataFrame of seed fragments. An empty string ⇒ de novo
# MAGIC generation; a SMILES/SAFE fragment ⇒ fragment-constrained generation (scaffold
# MAGIC decoration / growing). Params control how many molecules, sampling temperature,
# MAGIC randomness, diffusion steps, and which property to score/rank by.
# MAGIC
# MAGIC Output: one row per generated molecule — `seed`, `smiles`, `safe`, `score`.

# COMMAND ----------

import mlflow


class GenMolGenerator(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper around the GenMol masked-diffusion sampler.

    NOTE: GenMol ships as command-line scripts rather than a documented inline
    Python API. The generation entry point used here is `genmol.sampler.Sampler`
    from https://github.com/NVIDIA-Digital-Bio/GenMol — confirm the class/method
    names against the pinned commit at deploy time and adjust `_generate` if the
    upstream API differs. Everything else (I/O contract, SAFE<->SMILES handling,
    scoring, MLflow signature) is stable.
    """

    def load_context(self, context):
        import torch
        from genmol.sampler import Sampler  # GenMol Apache-2.0 package

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Sampler loads the diffusion checkpoint and the SAFE tokenizer.
        self.sampler = Sampler(context.artifacts["checkpoint"], device=self.device)

    def _score(self, smiles, scoring):
        """Rank generated molecules by a cheap RDKit drug-likeness proxy."""
        from rdkit import Chem
        from rdkit.Chem import QED, Crippen

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if scoring == "logp":
            return float(Crippen.MolLogP(mol))
        return float(QED.qed(mol))  # default: QED drug-likeness 0..1

    def _generate(self, seed, n, temperature, randomness, n_steps):
        """Dispatch de novo vs fragment-constrained generation; return SAFE strings."""
        import safe as sf

        kwargs = dict(
            num_samples=n,
            softmax_temp=temperature,
            randomness=randomness,
            num_steps=n_steps,
        )
        seed = (seed or "").strip()
        if not seed:
            return self.sampler.de_novo_generation(**kwargs)

        # Accept a SMILES seed for convenience; convert to SAFE for the model.
        frag = seed
        if "%" not in seed and "." not in seed:
            try:
                frag = sf.encode(seed)
            except Exception:
                frag = seed
        return self.sampler.fragment_completion(frag, **kwargs)

    def predict(self, context, model_input, params=None):
        import pandas as pd
        import safe as sf
        from rdkit import Chem

        params = params or {}
        n = int(params.get("num_molecules", 20))
        temperature = float(params.get("temperature", 1.0))
        randomness = float(params.get("randomness", 0.5))
        n_steps = int(params.get("n_steps", 100))
        scoring = str(params.get("scoring", "qed")).lower()
        unique = bool(params.get("unique", True))

        # Normalize input to a list of seed fragments.
        if hasattr(model_input, "iloc"):
            seeds = model_input.iloc[:, 0].tolist()
        elif hasattr(model_input, "tolist"):
            seeds = model_input.tolist()
        else:
            seeds = list(model_input)
        if not seeds:
            seeds = [""]

        rows = []
        seen = set()
        for seed in seeds:
            for safe_str in self._generate(seed, n, temperature, randomness, n_steps):
                try:
                    smiles = sf.decode(safe_str)
                except Exception:
                    continue
                mol = Chem.MolFromSmiles(smiles) if smiles else None
                if mol is None:
                    continue
                canonical = Chem.MolToSmiles(mol)
                if unique and canonical in seen:
                    continue
                seen.add(canonical)
                rows.append(
                    {
                        "seed": seed or "(de novo)",
                        "smiles": canonical,
                        "safe": safe_str,
                        "score": self._score(canonical, scoring),
                    }
                )

        result = pd.DataFrame(rows, columns=["seed", "smiles", "safe", "score"])
        # Highest score first (QED ↑ is better; for logp the UI can re-sort).
        if not result.empty:
            result = result.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
        return result

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model signature + smoke test

# COMMAND ----------

from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema

input_schema = Schema([ColSpec(type="string", name="fragment")])
output_schema = Schema(
    [
        ColSpec(type="string", name="seed"),
        ColSpec(type="string", name="smiles"),
        ColSpec(type="string", name="safe"),
        ColSpec(type="double", name="score"),
    ]
)
param_schema = ParamSchema(
    [
        ParamSpec("num_molecules", "integer", 20),
        ParamSpec("temperature", "double", 1.0),
        ParamSpec("randomness", "double", 0.5),
        ParamSpec("n_steps", "integer", 100),
        ParamSpec("scoring", "string", "qed"),
        ParamSpec("unique", "boolean", True),
    ]
)
signature = mlflow.models.signature.ModelSignature(
    inputs=input_schema, outputs=output_schema, params=param_schema
)

# Seed examples: "" → de novo; a fragment SMILES → scaffold decoration / growing.
import pandas as pd
input_example = pd.DataFrame({"fragment": ["", "c1ccc(cc1)C(=O)N"]})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register in Unity Catalog, import into Genesis Workbench, deploy

# COMMAND ----------

from genesis_workbench.models import (ModelCategory,
                                      import_model_from_uc,
                                      get_latest_model_version,
                                      deploy_model,
                                      set_mlflow_experiment)
from genesis_workbench.workbench import wait_for_job_run_completion

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

experiment = set_mlflow_experiment(experiment_tag=experiment_name,
                                   user_email=user_email,
                                   host=None,
                                   token=None,
                                   shared=True)

genmol_pyfunc = GenMolGenerator()

with mlflow.start_run(run_name=f"{model_name}", experiment_id=experiment.experiment_id):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="genmol",
        python_model=genmol_pyfunc,
        artifacts={"checkpoint": checkpoint_path},
        pip_requirements=[
            "torch==2.7.1",
            "safe-mol==0.1.13",
            "rdkit==2025.3.6",
            "git+https://github.com/NVIDIA-Digital-Bio/GenMol.git",
            "numpy",
            "pandas",
        ],
        input_example=input_example,
        signature=signature,
        registered_model_name=f"{catalog}.{schema}.{model_name}",
    )

# COMMAND ----------

model_uc_name = f"{catalog}.{schema}.{model_name}"
model_version = get_latest_model_version(model_uc_name)

gwb_model_id = import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.SMALL_MOLECULE,
                    model_uc_name=model_uc_name,
                    model_uc_version=model_version,
                    model_name="GenMol",
                    model_display_name="GenMol Molecule Generator",
                    model_source_version="NV-GenMol-89M-v2",
                    model_description_url="https://huggingface.co/nvidia/NV-GenMol-89M-v2")

# COMMAND ----------

run_id = deploy_model(user_email=user_email,
                gwb_model_id=gwb_model_id,
                deployment_name=f"GenMol",
                deployment_description="GenMol masked-diffusion generator for de novo / fragment-based small-molecule design",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type=workload_type,
                workload_size="Small")

# COMMAND ----------

result = wait_for_job_run_completion(run_id, timeout=3600)

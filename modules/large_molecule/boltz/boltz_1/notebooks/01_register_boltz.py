# Databricks notebook source
# MAGIC %md
# MAGIC ## Register Boltz-2 on serverless GPU
# MAGIC Boltz-2 (AlphaFold3-class biomolecular complex structure predictor). Runs on
# MAGIC the serverless GPU AI runtime; Boltz-2 weights auto-download into a fast local
# MAGIC cache and are packaged INTO the model so the serving container needs no
# MAGIC /Volumes reads or internet at inference time.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "boltz", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "boltz_cache_dir", "Cache dir")
dbutils.widgets.text("workload_type", "GPU_MEDIUM", "Workload Type for endpoints")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")


# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0
# MAGIC # dbboltz pulls boltz==2.2.1 (+ numpy<2.0, hydra-core, pytorch-lightning, rdkit …).
# MAGIC # On the serverless GPU AI runtime torch 2.7.1+cu126 is preinstalled and satisfies
# MAGIC # boltz's torch>=2.2, so torch is NOT reinstalled. Boltz-2 has its own attention
# MAGIC # path, so it needs no flash_attn (the [gpu] extra is intentionally empty).
# MAGIC %pip install ../dbboltz[gpu]

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/libraries")
for lib in libraries:
    if(lib.name.startswith("genesis_workbench")):
        gwb_library_path = lib.path.replace("dbfs:","")

print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
USER_EMAIL = dbutils.widgets.get("user_email")
SQL_WAREHOUSE_ID = dbutils.widgets.get("sql_warehouse_id")
CACHE_DIR = dbutils.widgets.get("cache_dir")
WORKLOAD_TYPE = dbutils.widgets.get("workload_type")

# COMMAND ----------

#Initialize Genesis Workbench
from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name = CATALOG, core_schema_name = SCHEMA, sql_warehouse_id = SQL_WAREHOUSE_ID, token = databricks_token)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Boltz-2 weights → fast LOCAL cache (not a /Volumes FUSE path)
# MAGIC `boltz predict` auto-downloads the Boltz-2 checkpoint + CCD ("mols") into its
# MAGIC `--cache` dir on first run. We point that at a fast local temp dir — writing
# MAGIC GBs through the /Volumes FUSE mount is pathologically slow (see the esmfold
# MAGIC migration notes). The populated cache is packaged into the model below, so the
# MAGIC serving container has the weights locally (Model Serving does not read /Volumes).

# COMMAND ----------

import os, tempfile
# Route the large UC model upload through the presigned-URL/S3 path (boto3
# multipart, no 5-min cap). The Databricks-SDK models artifact repo otherwise wraps
# the upload in a 5-min retry that a multi-GB model exceeds. (Same fix as esmfold.)
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "false"

local_cache = tempfile.mkdtemp(prefix="boltz2_cache_")
print(f"Boltz-2 cache: {local_cache}")

# COMMAND ----------

import mlflow
from dbboltz.boltz import run_boltz, Boltz
import yaml

mlflow.autolog(disable=True)

# COMMAND ----------

def get_model_config():
    model_config = {}
    # jackhmmer MSA is NOT used at serving (serving uses no_msa / the mmseqs
    # server), so no jackhmmer binary is installed. Left as None; the 'jh' MSA
    # code path that would need it is never taken at serving time.
    model_config['jackhmmer_binary_path'] = None
    model_config['compute_type'] = 'gpu'
    return model_config

model_config = get_model_config()
model_config

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize the model (cache points at our local temp dir)

# COMMAND ----------

model = Boltz()
context = mlflow.pyfunc.PythonModelContext(
    artifacts = {
        "CACHE_DIR": local_cache
    },
    model_config = model_config
)
model.load_context(context)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper: map dict-type input to the string input format used for serving

# COMMAND ----------

def convert_input_to_serving_input(inputs):
    out_dict = dict()
    for k, v in inputs.items():
        for in_seqs in v:
            chain_ids = ','.join(in_seqs[0])
            sequence = in_seqs[1]
            out_dict[k+'_'+chain_ids] = sequence
    out_str = ""
    for k,v in out_dict.items():
        out_str += k+':'+v+';'
    out_str = out_str.rstrip(';')
    return out_str

# COMMAND ----------

inputs = {
    'protein':[
        ( ('A'),"GTGAMWLTKLVLNPASRAARRDLANPYEMHRTLSKAVSRALEEGRERLLWRLEPARGLEPPVVLVQTLTEPDWSVLDEGYAQVFPPKPFHPALKPGQRLRFRLRANPAKRLAATGKRVALKTPAEKVAWLERRLEEGGFRLLEGERGPWVQILQDTFLEVRRKKDGEEAGKLLQVQAVLFEGRLEVVDPERALATLRRGVGPGKALGLGLLSVAP"),
    ],
    'rna': [
        ( ('B'), "UCCCCACGCGUGUGGGGAU")
    ]
}

# COMMAND ----------

# serving endpoint expects this dict format
model_input = {
    'input': convert_input_to_serving_input(inputs),
    'msa': 'no_msa',
}
print(model_input)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Smoke-test the model
# MAGIC First call also triggers the Boltz-2 weight download into `local_cache`.

# COMMAND ----------

result = model.predict(context, [model_input])
print(result[0].keys())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ship dbboltz to the serving container via `code_paths`
# MAGIC `code_paths` points straight at the package source, so the whole `dbboltz`
# MAGIC package (incl. the `dbboltz.alphafold` submodule) is copied into the model and
# MAGIC is importable at serving. Serving deps (boltz==2.2.1, rdkit, absl-py) come from
# MAGIC `conda_env.yml` as normal PyPI packages — NOT a `/model/artifacts/dbboltz` pip
# MAGIC line, whose absolute path does not exist during the serving image build.

# COMMAND ----------

# code_paths copies this directory into the model as code/dbboltz -> `import dbboltz`.
dbboltz_code_path = "../dbboltz/src/dbboltz"
print("dbboltz code path:", os.path.abspath(dbboltz_code_path), "->", os.listdir(dbboltz_code_path))

# COMMAND ----------

from genesis_workbench.models import (ModelCategory,
                                      import_model_from_uc,
                                      deploy_model,
                                      get_latest_model_version,
                                      set_mlflow_experiment)

from genesis_workbench.workbench import wait_for_job_run_completion

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log and register Boltz-2 on Unity Catalog

# COMMAND ----------

from mlflow.models.signature import infer_signature

registered_model_name = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

signature = infer_signature([model_input], result)
print(signature)

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

experiment = set_mlflow_experiment(experiment_tag=EXPERIMENT_NAME,
                                   user_email=USER_EMAIL,
                                   host=None,
                                   token=None,
                                   shared=True)

with mlflow.start_run(run_name=f"{MODEL_NAME}", experiment_id=experiment.experiment_id):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=Boltz(),
        artifacts={
            "CACHE_DIR": local_cache
        },
        code_paths=[dbboltz_code_path],
        model_config=model_config,
        input_example=[model_input],
        signature=signature,
        conda_env="conda_env.yml",
        registered_model_name=registered_model_name
    )

# COMMAND ----------

model_version = get_latest_model_version(registered_model_name)
model_uri = f"models:/{registered_model_name}/{model_version}"

gwb_model_id = import_model_from_uc(user_email=USER_EMAIL,
                    model_category=ModelCategory.LARGE_MOLECULE,
                    model_uc_name=registered_model_name,
                    model_uc_version=model_version,
                    model_name=MODEL_NAME,
                    model_display_name="Boltz-2",
                    model_source_version="v2.2.1",
                    model_description_url="https://github.com/jwohlwend/boltz")

# COMMAND ----------

run_id = deploy_model(user_email=USER_EMAIL,
                gwb_model_id=gwb_model_id,
                deployment_name=f"boltz-2",
                deployment_description="Boltz-2 biomolecular complex structure predictor (MIT). AlphaFold3-class diffusion + Pairformer; predicts protein monomers, multimers, and protein-ligand/nucleic-acid complexes from sequence input — returns PDB + confidence scores.",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type=WORKLOAD_TYPE,
                workload_size="Small")

# COMMAND ----------

result = wait_for_job_run_completion(run_id, timeout = 7200)

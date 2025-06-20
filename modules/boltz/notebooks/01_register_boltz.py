# Databricks notebook source
# MAGIC %md
# MAGIC ## Download model weights and JackHMMer binary and place in Unity Catalog Volume

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("model_name", "boltz", "Model Name")
dbutils.widgets.text("model_name", "boltz", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "boltz_cache_dir", "Cache dir")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")


# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0
# MAGIC %pip install ../dbboltz[gpu]
# MAGIC %pip install py3Dmol

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

print(f"Cache dir: {CACHE_DIR}")
cache_full_path = f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}"
print(f"Cache full path: {cache_full_path}")

# COMMAND ----------

WEIGHTS_VOLUME_LOCATION = f"{cache_full_path}/weights"
BINARIES_VOLUME_LOCATION = f"{cache_full_path}/binaries"

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{CACHE_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download Weights place in Volumes

# COMMAND ----------

import urllib
from pathlib import Path
CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MODEL_URL = (
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"
)

def download(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        # click.echo(
        #     f"Downloading the CCD dictionary to {ccd}. You may "
        #     "change the cache directory with the --cache flag."
        # )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        # click.echo(
        #     f"Downloading the model weights to {model}. You may "
        #     "change the cache directory with the --cache flag."
        # )
        urllib.request.urlretrieve(MODEL_URL, str(model)) 

# COMMAND ----------

import os

if not os.path.exists(WEIGHTS_VOLUME_LOCATION):
  os.makedirs(WEIGHTS_VOLUME_LOCATION)

if not os.path.exists(BINARIES_VOLUME_LOCATION):
  os.makedirs(BINARIES_VOLUME_LOCATION)
  

if not os.path.exists(WEIGHTS_VOLUME_LOCATION):
  os.makedirs(WEIGHTS_VOLUME_LOCATION)

if not os.path.exists(BINARIES_VOLUME_LOCATION):
  os.makedirs(BINARIES_VOLUME_LOCATION)
  
download(Path(WEIGHTS_VOLUME_LOCATION))

# COMMAND ----------

os.environ["BINARIES_VOLUME_LOCATION"] = BINARIES_VOLUME_LOCATION

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC if [ ! -d "/miniconda3" ]; then
# MAGIC   mkdir -p /miniconda3
# MAGIC
# MAGIC   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh
# MAGIC   bash /miniconda3/miniconda.sh -b -u -p /miniconda3
# MAGIC   rm -rf /miniconda3/miniconda.sh
# MAGIC fi
# MAGIC
# MAGIC source /miniconda3/bin/activate
# MAGIC conda config --remove channels defaults
# MAGIC
# MAGIC conda create -n jackhmmer_env python=3.8 --yes
# MAGIC conda activate jackhmmer_env
# MAGIC conda install -y bioconda::hmmer
# MAGIC
# MAGIC cp /miniconda3/envs/jackhmmer_env/bin/jackhmmer ${BINARIES_VOLUME_LOCATION}/

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and register Boltz-1 model on Unity Catalog
# MAGIC
# MAGIC - note that with dbboltz version we suport JackHMMer option for MSA when running in notebook (but for serving the MSA with JackHMMer will take too long)

# COMMAND ----------

import mlflow
from dbboltz.boltz import run_boltz, Boltz
import yaml

mlflow.autolog(disable=True)

mlflow.autolog(disable=True)

# COMMAND ----------

def get_model_config():
    model_config = {}
    model_config['jackhmmer_binary_path'] = "/miniconda3/envs/jackhmmer_env/bin/jackhmmer"
    model_config['compute_type'] = 'gpu'
    return model_config

model_config = get_model_config()

# COMMAND ----------

model_config

# COMMAND ----------

model_config

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize the model

# COMMAND ----------

model = Boltz()
context = mlflow.pyfunc.PythonModelContext(
    artifacts = {
        "CACHE_DIR": f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}"
        "CACHE_DIR": f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}"
    },
    model_config = model_config
)
model.load_context(context)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper function to map between dictionary type input and string input for serving

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
params = {
    'msa': 'no_msa',
    'msa_depth': 20,
    'diffusion_samples': 1,
    'recycling_steps': 3,
    'sampling_steps': 200,
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test the model out

# COMMAND ----------

# serving enpoint expects this format...maybe just standardize to this? like everywhere?
model_input = {
    'input': convert_input_to_serving_input(inputs),
    'msa': 'no_msa',
    'use_msa_server': 'True'
}
print(model_input)

# COMMAND ----------

result = model.predict(context, [model_input])

# COMMAND ----------

# MAGIC %md
# MAGIC #### See what the output looks like

# COMMAND ----------

import py3Dmol

view = py3Dmol.view(width=800, height=300)

view.addModel(
    result[0]['pdb'],
    'pdb'
)
view.setStyle({'chain': 'A'}, {'cartoon': {'color': 'blue'}})
view.setStyle({'chain': 'B'}, {'cartoon': {'color': 'red'}})

view.zoomTo()
html = view._make_html()
displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's log our model too

# COMMAND ----------

# MAGIC %sh
# MAGIC # move a copy of our code base to "local" machine and then register it with the model
# MAGIC # this will make a copy of our codebase that we can then install on the server for model serving
# MAGIC mkdir -p /local_disk0/dbboltz
# MAGIC cp -r ../dbboltz/src /local_disk0/dbboltz
# MAGIC cp ../dbboltz/pyproject.toml /local_disk0/dbboltz

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /local_disk0/dbboltz
# MAGIC echo " -- "
# MAGIC ls /local_disk0/dbboltz/src/dbboltz

# COMMAND ----------

result[0]

# COMMAND ----------

from genesis_workbench.models import (ModelCategory, 
                                      import_model_from_uc,
                                      get_latest_model_version,
                                      set_mlflow_experiment)

from genesis_workbench.workbench import AppContext

# COMMAND ----------

from mlflow.types.schema import ColSpec, Schema
from mlflow.models.signature import infer_signature

registered_model_name = f"{CATALOG}.{SCHEMA}.boltz"

signature = infer_signature([model_input], result)
print(signature)

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

experiment = set_mlflow_experiment(experiment_tag=experiment_name, user_email=user_email)

with mlflow.start_run(run_name=f"{model_name}", experiment_id=experiment.experiment_id):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=Boltz(),
        artifacts={
            "CACHE_DIR": f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}",
            "repo_path": "/local_disk0/dbboltz"
            "CACHE_DIR": f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}",
            "repo_path": "/local_disk0/dbboltz"
        },
        model_config=model_config,
        input_example=[model_input],
        signature=signature,
        conda_env="conda_env.yml",
        registered_model_name=registered_model_name
    )

# COMMAND ----------

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
os.environ["SQL_WAREHOUSE"]=SQL_WAREHOUSE_ID
os.environ["IS_TOKEN_AUTH"]="Y"
os.environ["DATABRICKS_TOKEN"]=databricks_token


# COMMAND ----------

model_version = get_latest_model_version(registered_model_name)
model_uri = f"models:/{registered_model_name}/{model_version}"

app_context = AppContext(
        core_catalog_name=CATALOG,
        core_schema_name=SCHEMA
    )

import_model_from_uc(app_context,user_email=USER_EMAIL,
                    model_category=ModelCategory.PROTEIN_STUDIES,
                    model_uc_name=registered_model_name,
                    model_uc_version=model_version,
                    model_name="boltz",
                    model_display_name="Boltz-1",
                    model_source_version="v1.0.0",
                    model_description_url="https://huggingface.co/boltz-community/boltz-1")

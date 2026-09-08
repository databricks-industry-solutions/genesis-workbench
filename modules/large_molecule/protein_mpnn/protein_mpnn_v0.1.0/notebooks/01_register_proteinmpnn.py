# Databricks notebook source
# MAGIC %md
# MAGIC ## Register ProteinMPNN on serverless GPU
# MAGIC ProteinMPNN inverse-folding / sequence design. Runs on the serverless GPU AI
# MAGIC runtime (torch preinstalled); weights + package are copied to fast local temp
# MAGIC and packaged INTO the model so serving needs no /Volumes reads or internet.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt
# MAGIC %pip install ../proteinmpnn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "proteinmpnn", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "protein_mpnn_cache_dir", "Cache dir")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
USER_EMAIL = dbutils.widgets.get("user_email")
SQL_WAREHOUSE_ID = dbutils.widgets.get("sql_warehouse_id")
CACHE_DIR = dbutils.widgets.get("cache_dir")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fetch ProteinMPNN weights → fast local temp
# MAGIC Clone the official repo (weights are bundled, only a few MB) into a local temp
# MAGIC dir via subprocess — no `%sh` and no writes to `/` (unwritable on serverless).

# COMMAND ----------

import os, tempfile, subprocess

REPO_DIR = tempfile.mkdtemp(prefix="proteinmpnn_repo_")
subprocess.run(
    ["git", "clone", "--depth", "1",
     "https://github.com/dauparas/ProteinMPNN.git", REPO_DIR],
    check=True,
)
# The vanilla weights are what we serve (v_48_002/010/020/030.pt; default v_48_020)
VANILLA_WEIGHTS = os.path.join(REPO_DIR, "vanilla_model_weights")
print("weights:", os.listdir(VANILLA_WEIGHTS))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grab an example PDB and reduce to backbone (for the smoke test)

# COMMAND ----------

import Bio.PDB as PDB
def convert_to_backbone(pdb_file_path, output_file_path, chain_id='A'):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)

    new_structure = PDB.Structure.Structure('backbone')
    new_model = PDB.Model.Model(0)
    new_chain = PDB.Chain.Chain(chain_id)
    new_structure.add(new_model)
    new_model.add(new_chain)

    residue_id = 1
    for model in structure:
        for chain in model:
            if chain.id==chain_id:
                for residue in chain:
                    if residue.id[0] != ' ' or residue.resname == 'HOH':
                        continue
                    first_atom = True
                    for atom in residue:
                        if atom.altloc == 'A' or atom.altloc == ' ':
                            if atom.name in ['N', 'CA', 'C', 'O']:
                                if first_atom:
                                    new_residue = PDB.Residue.Residue((' ', residue_id, ' '), 'GLY', ' ')
                                    first_atom = False
                                new_residue.add(atom)
                    try:
                        last_one = [r for r in new_chain.get_residues()][-1]
                    except:
                        last_one = None
                    if last_one!=new_residue:
                        new_chain.add(new_residue)
                    residue_id += 1

    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_file_path)

# COMMAND ----------

import requests

EXAMPLE_DIR = tempfile.mkdtemp(prefix="proteinmpnn_example_")
url = "https://files.rcsb.org/download/5yd3.pdb"
response = requests.get(url)
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_file:
    temp_file.write(response.content)
    pdb_file_path = temp_file.name
example_backbone = os.path.join(EXAMPLE_DIR, "5yd3.pdb")
convert_to_backbone(pdb_file_path, example_backbone)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the ProteinMPNN model as an mlflow PythonModel

# COMMAND ----------

from proteinmpnn.run import main, get_argparser
from proteinmpnn.parse_multiple_chains import main as pdb_main
from proteinmpnn.parse_multiple_chains import get_argparser as pdb_get_argparser

from typing import Optional, List

import mlflow
from mlflow.types.schema import ColSpec, Schema
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

class ProteinMPNN(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model_dir = context.artifacts['model_dir']

    def _prepare_pdb_input(self,pdb_str:str,outdir:str):

        from proteinmpnn.parse_multiple_chains import main as pdb_main
        from proteinmpnn.parse_multiple_chains import get_argparser as pdb_get_argparser
        import tempfile
        parser = pdb_get_argparser()

        with tempfile.TemporaryDirectory() as temp_dir:
            with open(temp_dir + "/my_pdb.pdb", "w") as f:
                f.write(pdb_str)

            arg_list = []
            # arg_list.extend(['--ca_only'])
            arg_list.extend(['--input_path', temp_dir])
            arg_list.extend(['--output_path', f'{outdir}/inputs.jsonl'])
            args = parser.parse_args(arg_list)
            pdb_main(args)
        return None

    def _run_proteinmpnn(self, input_path, output_dir, fixed_positions_jsonl=None):
        from proteinmpnn.run import main, get_argparser
        import os, json

        parser = get_argparser()
        arg_list = []
        arg_list.extend(['--suppress_print', "1"])
        # arg_list.extend(['--ca_only'])
        arg_list.extend(['--jsonl_path', input_path])
        arg_list.extend(['--out_folder', output_dir])
        arg_list.extend(['--num_seq_per_target', "3"])
        arg_list.extend(['--sampling_temp', "0.1"])
        arg_list.extend(['--batch_size', "1"])
        arg_list.extend(['--path_to_model_weights', self.model_dir])
        if fixed_positions_jsonl is not None:
            arg_list.extend(['--fixed_positions_jsonl', fixed_positions_jsonl])

        # ─── Debug: surface what we're about to feed ProteinMPNN ─────────────
        print(f"[DEBUG] arg_list passed to ProteinMPNN: {arg_list}")
        if os.path.isfile(input_path):
            with open(input_path) as f:
                print(f"[DEBUG] inputs.jsonl ({os.path.getsize(input_path)} bytes):")
                for i, line in enumerate(f):
                    obj = json.loads(line)
                    obj_summary = {k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                                   for k, v in obj.items()}
                    print(f"[DEBUG]   line {i}: {obj_summary}")
        if fixed_positions_jsonl is not None and os.path.isfile(fixed_positions_jsonl):
            with open(fixed_positions_jsonl) as f:
                print(f"[DEBUG] fixed_positions.jsonl: {f.read().strip()}")

        args = parser.parse_args(arg_list)
        main(args)

        # Surface the FASTA output so we can see what MPNN actually wrote.
        fa_path = os.path.join(output_dir, "seqs", "my_pdb.fa")
        if os.path.isfile(fa_path):
            with open(fa_path) as f:
                print(f"[DEBUG] output FASTA contents:\n{f.read()}")
        return None


    def predict(self, context, inputs, params=None) -> List[str]:
        """
        Parameters
        ----------
        inputs : single-entry DataFrame / list. Accepted column shapes:
            1. one-column ``pdb`` string — legacy: redesign every residue.
            2. two-column ``pdb`` + ``fixed_positions`` — ``fixed_positions`` is
               a JSON-encoded ``{chain_id: [residue_numbers]}`` dict (1-indexed
               within the chain). Listed positions keep their input AA identity;
               everything else is redesigned. Empty string / null = no fix.

        ``fixed_positions`` is sent as a JSON string (not a nested dict) because
        MLflow's ColSpec schema enforcement is the only shape that survives
        Databricks Model Serving's input projection — nested dicts in unnamed
        columns get silently dropped by ``_enforce_schema``.
        """
        import tempfile, json, os
        import pandas as pd

        pdb_str = None
        fixed_positions = None  # parsed dict {chain: [residues]}, or None

        def _parse_fp(fp):
            """Accept dict (already parsed), JSON string, None, NaN, or empty string."""
            if fp is None:
                return None
            if isinstance(fp, float) and pd.isna(fp):
                return None
            if isinstance(fp, str):
                fp = fp.strip()
                if not fp:
                    return None
                return json.loads(fp)
            if isinstance(fp, dict):
                return fp
            raise TypeError(f"Unexpected fixed_positions type: {type(fp).__name__}")

        if isinstance(inputs, pd.DataFrame):
            if len(inputs) != 1:
                raise ValueError(f"Expected exactly one input row; got {len(inputs)}")
            row = inputs.iloc[0]
            if "pdb" in inputs.columns:
                pdb_str = str(row["pdb"])
                if "fixed_positions" in inputs.columns:
                    fixed_positions = _parse_fp(row["fixed_positions"])
            elif inputs.shape[1] == 1:
                pdb_str = str(row.iloc[0])
            else:
                raise ValueError(
                    f"DataFrame input missing 'pdb' column; got columns {list(inputs.columns)}"
                )
        elif isinstance(inputs, list):
            if len(inputs) != 1:
                raise ValueError(f"Expected exactly one input; got {len(inputs)}")
            first = inputs[0]
            if isinstance(first, dict):
                pdb_str = first.get("pdb") or first.get("pdb_str")
                fixed_positions = _parse_fp(first.get("fixed_positions"))
            else:
                pdb_str = str(first)
        else:
            raise TypeError(f"Unexpected inputs type: {type(inputs).__name__}")

        if not pdb_str:
            raise ValueError("Could not extract a 'pdb' string from inputs")

        with tempfile.TemporaryDirectory() as tmpdir:
            self._prepare_pdb_input(pdb_str, tmpdir)

            # Build the upstream-format JSONL: {"my_pdb": {chain_id: [residues]}}
            fixed_positions_jsonl = None
            if fixed_positions:
                fp_dict = {
                    "my_pdb": {str(chain): list(residues)
                               for chain, residues in fixed_positions.items()}
                }
                fixed_positions_jsonl = os.path.join(tmpdir, "fixed_positions.jsonl")
                with open(fixed_positions_jsonl, "w") as f:
                    f.write(json.dumps(fp_dict) + "\n")

            with tempfile.TemporaryDirectory() as outdir:
                self._run_proteinmpnn(
                    tmpdir + '/inputs.jsonl', outdir,
                    fixed_positions_jsonl=fixed_positions_jsonl,
                )
                with open(outdir+'/seqs/my_pdb.fa', 'r') as f:
                    lines = f.readlines()
                seqs = lines[3::2]
        return [s.strip() for s in seqs]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke-test the model

# COMMAND ----------

model = ProteinMPNN()

artifacts={
    "model_dir" : VANILLA_WEIGHTS,
}
context=mlflow.pyfunc.PythonModelContext(artifacts=artifacts, model_config=dict())
model.load_context(context)

with open(example_backbone, 'r') as f:
    in_pdb_str = f.read()

seqs = model.predict(
    context,
    [in_pdb_str]
)
print(seqs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC Ship the vendored `proteinmpnn` package to the serving container via mlflow
# MAGIC `code_paths` (copied into the model as code/proteinmpnn -> importable at serving).
# MAGIC Its runtime deps come from `conda_env.yaml` as PyPI packages — NOT a
# MAGIC `/model/artifacts/package` pip line, whose absolute path does not exist during
# MAGIC the serving image build (that broke the container build).

# COMMAND ----------

# code_paths copies this dir into the model as code/proteinmpnn -> `import proteinmpnn`.
proteinmpnn_code_path = "../proteinmpnn/src/proteinmpnn"
print("proteinmpnn code path:", os.path.abspath(proteinmpnn_code_path), "->", os.listdir(proteinmpnn_code_path))

# COMMAND ----------

# Route the (small) UC model upload through the presigned-URL/S3 path — consistent
# with the other large_molecule models; avoids the Databricks-SDK 5-min upload cap.
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "false"

from databricks.sdk import WorkspaceClient

signature = mlflow.models.signature.ModelSignature(
    inputs = Schema([
        ColSpec(type="string", name="pdb"),
        ColSpec(type="string", name="fixed_positions", required=False),
    ]),
    outputs = Schema([ColSpec(type="string")]),
    params = None
)

import pandas as _pd_for_example
_input_example_df = _pd_for_example.DataFrame([{"pdb": in_pdb_str, "fixed_positions": ""}])

def set_mlflow_experiment(experiment_tag, user_email):
    w = WorkspaceClient()
    mlflow_experiment_base_path = "Shared/dbx_genesis_workbench_models"
    w.workspace.mkdirs(f"/Workspace/{mlflow_experiment_base_path}")
    experiment_path = f"/{mlflow_experiment_base_path}/{experiment_tag}"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    return mlflow.set_experiment(experiment_path)

experiment = set_mlflow_experiment(experiment_tag=EXPERIMENT_NAME, user_email=USER_EMAIL)

with mlflow.start_run(run_name='protein_mpnn',experiment_id=experiment.experiment_id):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ProteinMPNN(),
        artifacts={
            "model_dir" : VANILLA_WEIGHTS
        },
        code_paths=[proteinmpnn_code_path],
        input_example=_input_example_df,
        signature=signature,
        conda_env="conda_env.yaml",
        registered_model_name=f"{CATALOG}.{SCHEMA}.proteinmpnn"
    )

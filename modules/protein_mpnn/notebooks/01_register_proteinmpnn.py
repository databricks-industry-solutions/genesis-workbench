# Databricks notebook source
# MAGIC %pip install -r requirements.txt
# MAGIC %pip install ../proteinmpnn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p /proteinmpnn
# MAGIC mkdir -p /proteinmpnn/example_data
# MAGIC mkdir -p /proteinmpnn/repos
# MAGIC cd /proteinmpnn/repos
# MAGIC git clone https://github.com/dauparas/ProteinMPNN.git

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("model_name", "rfdiffusion", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "protein_mpnn_cache_dir", "Cache dir")

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

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{CACHE_DIR}")

# COMMAND ----------

import shutil

shutil.copytree("/proteinmpnn/repos/ProteinMPNN/ca_model_weights", f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}/ca_model_weights", dirs_exist_ok=True)
shutil.copytree("/proteinmpnn/repos/ProteinMPNN/vanilla_model_weights", f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}/vanilla_model_weights", dirs_exist_ok=True)
shutil.copytree("/proteinmpnn/repos/ProteinMPNN/soluble_model_weights", f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}/soluble_model_weights", dirs_exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grab example PDB and convert to backbone only
# MAGIC - Additionally, run the data preparation to JSONL per ProteinMPNN package

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
                    print(residue)
                    first_atom = True
                    for atom in residue:
                        if atom.altloc == 'A' or atom.altloc == ' ':
                            if atom.name in ['N', 'CA', 'C', 'O']:
                                # print(residue, atom)
                                if first_atom:
                                    # print("is first")
                                    new_residue = PDB.Residue.Residue((' ', residue_id, ' '), 'GLY', ' ')
                                    first_atom = False
                                new_residue.add(atom)  # Add the backbone atom to the new residue
                    try:
                        last_one = [r for r in new_chain.get_residues()][-1]
                    except:
                        last_one = None
                    if last_one!=new_residue:
                        new_chain.add(new_residue)
                    residue_id += 1  # Increment residue ID

    # Step 3: Write the modified alpha carbons to a new PDB file
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_file_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Now actually donwload a PDB and process to backbone

# COMMAND ----------

# Download the PDB file
import requests

url = "https://files.rcsb.org/download/5yd3.pdb"
import tempfile

response = requests.get(url)
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_file:
    temp_file.write(response.content)
    init_pdb_str = response.content.decode('utf-8')
    pdb_file_path = temp_file.name

    convert_to_backbone(pdb_file_path,'/proteinmpnn/example_data/5yd3.pdb')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Convert the pdb backbone to JSONL format

# COMMAND ----------

import requests
from proteinmpnn.parse_multiple_chains import main, get_argparser
url = "https://files.rcsb.org/download/5yd3.pdb"
import tempfile
import shutil

parser = get_argparser()

with tempfile.TemporaryDirectory() as temp_dir:
    response = requests.get(url)

    # pre-clean the structure to a given chain, the actual removal of CA not necessary...
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_file:
        temp_file.write(response.content)
        pdb_file_path = temp_file.name
        convert_to_backbone(pdb_file_path,'/proteinmpnn/example_data/5yd3_backbone.pdb')
        
        shutil.copy(
            '/proteinmpnn/example_data/5yd3_backbone.pdb', 
            temp_dir + "/5yd3.pdb"
        )

    arg_list = []
    arg_list.extend(['--ca_only'])
    arg_list.extend(['--input_path', temp_dir])
    arg_list.extend(['--output_path', '/proteinmpnn/example_data/example_inputs.jsonl'])
    args = parser.parse_args(arg_list)
    main(args)

    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and Register the ProteinMPNN model to Unity Catalog
# MAGIC

# COMMAND ----------

from proteinmpnn.run import main, get_argparser
from proteinmpnn.parse_multiple_chains import main as pdb_main
from proteinmpnn.parse_multiple_chains import get_argparser as pdb_get_argparser
import tempfile

from typing import Optional,List

import mlflow
from mlflow.types.schema import ColSpec, Schema
mlflow.set_registry_uri("databricks-uc")
        

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define our model as a subclass of mlflow's PythonModel 
# MAGIC  - internally has a pointer to the model weights (stored as artifact with the model)
# MAGIC  - runs the main proteinmpnn inference command on predict, after preparing the input to correct format

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

    def _run_proteinmpnn(self,input_path, output_dir):
        from proteinmpnn.run import main, get_argparser

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
        args = parser.parse_args(arg_list)

        main(args)
        return None


    def predict(self, context, inputs : List[str], params=None) -> List[str]:
        """
        parameters: 
        -----------
        inputs: single entry list (currently) of pdb string for a PDB with only backbone atoms
        """
        import tempfile
        if len(inputs)!= 1:
            raise ValueError("Expected exactly one input")
        pdb_str= inputs[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._prepare_pdb_input(pdb_str,tmpdir)
            with tempfile.TemporaryDirectory() as outdir:
                self._run_proteinmpnn(tmpdir+'/inputs.jsonl', outdir)
                with open(outdir+'/seqs/my_pdb.fa', 'r') as f:
                    lines = f.readlines()
                seqs = lines[3::2]
        return [s.strip() for s in seqs]




# COMMAND ----------

# MAGIC %md
# MAGIC ## Test running the model with the test input we prepared in the previous notebook

# COMMAND ----------

model = ProteinMPNN()

artifacts={
    "model_dir" : f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}/vanilla_model_weights/",
}
context=mlflow.pyfunc.PythonModelContext(artifacts=artifacts, model_config=dict())
model.load_context(context)

with open('/proteinmpnn/example_data/5yd3.pdb', 'r') as f:
    in_pdb_str = f.read()

seqs = model.predict(
    context,
    [in_pdb_str]
)

# COMMAND ----------

seqs

# COMMAND ----------

# MAGIC %md
# MAGIC ## register our model to Unity Catalog
# MAGIC  - first we make a copy of the proteinmpnn repo to the compute's local disk and then add it to the model so that the source code is stored along with the model
# MAGIC  - this allows us to use a path to that artifact in the conda environment we specify with the model so that we can pip install the package along with the model (even though the package is not on pypi and is locally defined).

# COMMAND ----------

# MAGIC %sh
# MAGIC # move a copy of our code base to "local" machine and then register it with the model
# MAGIC # this will make a copy of our codebase that we can then install on the server for model serving
# MAGIC mkdir -p /proteinmpnn/package
# MAGIC cp -r ../proteinmpnn/src /proteinmpnn/package
# MAGIC cp ../proteinmpnn/pyproject.toml /proteinmpnn/package

# COMMAND ----------

signature = mlflow.models.signature.ModelSignature(
    inputs = Schema([ColSpec(type="string")]),
    outputs = Schema([ColSpec(type="string")]),
    params = None
)

with mlflow.start_run(run_name='protein_mpnn'):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ProteinMPNN(),
        artifacts={
            "model_dir" : f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}/vanilla_model_weights/",
            "repo_path": "/proteinmpnn/package"
        },
        input_example=[in_pdb_str],
        signature=signature,
        conda_env="conda_env.yaml",
        registered_model_name=f"{CATALOG}.{SCHEMA}.proteinmpnn"
    )

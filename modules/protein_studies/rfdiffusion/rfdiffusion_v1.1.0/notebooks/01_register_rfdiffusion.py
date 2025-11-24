# Databricks notebook source
# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "rfdiffusion", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "rfdiffussion_cache_dir", "Cache dir")

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

import os

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
os.environ["SQL_WAREHOUSE"]=SQL_WAREHOUSE_ID
os.environ["IS_TOKEN_AUTH"]="Y"
os.environ["DATABRICKS_TOKEN"]=databricks_token

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p /rfd
# MAGIC cd /rfd
# MAGIC rm -rf RFdiffusion
# MAGIC git clone https://github.com/RosettaCommons/RFdiffusion.git
# MAGIC cd RFdiffusion
# MAGIC git checkout e22092420281c644b928e64d490044dfca4f9175

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download the RFDiffusion code and model weights to Unity Catalog

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /rfd/RFdiffusion
# MAGIC mkdir models && cd models
# MAGIC wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
# MAGIC wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
# MAGIC wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt
# MAGIC wget http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt
# MAGIC wget http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt
# MAGIC wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
# MAGIC wget http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{CACHE_DIR}")

# COMMAND ----------

import shutil

shutil.copytree("/rfd/", f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}", dirs_exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model definition for RFDiffusion for Unconstrained Problem
# MAGIC  - this is for predicting a backbone with only the protein legth being a constraint.
# MAGIC  - We use the mlflow PythonModel as the base class
# MAGIC  - Although RFDiffusion expects to run from command line, we set Hydra config within python to be able to run the main function from within our python code

# COMMAND ----------

import subprocess
import os
import tempfile
import mlflow
from mlflow.types.schema import ColSpec, Schema
from typing import Any, Dict, List, Optional

import logging

# COMMAND ----------

class RFDiffusionUnconditional(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model_path = context.artifacts['model_path']
        self.script_path = context.artifacts['script_path']
        self.example_path = context.artifacts['example_path']
        
        import sys
        import os
        self.config_path = context.artifacts['config_path']
        self.rel_config_path = os.path.relpath("/", sys.argv[0])[:-3] + self.config_path

    def _validate_input(self,plen):
        if not isinstance(plen,int):
            try:
                plen = int(plen)
            except:
                raise TypeError("plen should be an int (and less than 180)")
        if plen>180:
            raise ValueError("plen must be less than 180, {plen} was passed")
        if plen==0:
            raise ValueError("protein length must be greater than 0")
        return plen
    
    def _make_config(self,plen:int,outpath:str='out'):
        """ creates a hydra config file for the run script"""
        import hydra
        # with hydra.initialize(version_base=None, config_path=self.rel_config_path):
        cfg = hydra.compose(
            config_name="base", 
            overrides=[
                f'contigmap.contigs=[{plen}-{plen}]',
                f'inference.output_prefix={outpath}/output',
                'inference.num_designs=1',
                f'inference.model_directory_path={self.model_path}',
                f'inference.input_pdb={self.example_path}/input_pdbs/1qys.pdb',
                f'diffuser.T=20'
            ],
            return_hydra_config=True,
            )
        return cfg
    
    def _dummy_hydra(self):
        import os
        from omegaconf import OmegaConf
        hydra_runtime = OmegaConf.create({
            "runtime": {
                "output_dir": "/path/to/outputs",  
                "cwd": os.getcwd()
            },
            "job": {
                "name": "manual_job",
                "num": 0
            }
        })
        return hydra_runtime

    def _run_inference(self,plen:int):
        """ runs inference script with fixed environment 
        
        parameters
        -----------
        plen:
            The length of protein to generate
        """
        import sys
        sys.path.append(self.script_path)
        import hydra
        from run_inference import main as mn
        from omegaconf import OmegaConf
        from hydra.core.hydra_config import HydraConfig

        plen = self._validate_input(plen)
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            with hydra.initialize(version_base=None, config_path=self.rel_config_path):
                cfg = self._make_config(plen=plen,outpath=tmpdirname)
                # add dummy hydra pieces and Merge with existing config
                cfg = OmegaConf.merge(
                    {"hydra": self._dummy_hydra()},
                    cfg
                )
                HydraConfig.instance().set_config(cfg)
                mn(cfg)
            with open('{}/output_0.pdb'.format(tmpdirname),'r') as f:
                pdbtext = f.read()
        return pdbtext
    
    def predict(self, context, model_input : List[str], params=None) -> List[str]:
        """ Generate structure of protein of given length
        parameters
        --------

        context:
            The mlflow context of the model. Gathered by load_context()
        
        model_input:
            A list of strings of protein lengths. Should only contain one entry in the list.
            The string of protein length, e.g "10" will internally be converted to int.

        params: Optional[Dict[str, Any]]
            Additional parameters
        """
        if len(model_input)>1:
            raise ValueError("input must be a list with a single integer as string")

        # convert to int (str input is easier to manage on server side)
        plen = int(model_input[0])
        pdb = self._run_inference(plen)
        return pdb

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inpainting version of RFDiffusion

# COMMAND ----------

class RFDiffusionInpainting(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model_path = context.artifacts['model_path']
        self.script_path = context.artifacts['script_path']
        
        import sys
        import os
        self.config_path = context.artifacts['config_path']
        self.rel_config_path = os.path.relpath("/", sys.argv[0])[:-3] + self.config_path

        # more than this is too slow for serving...
        self.num_designs=1
        self.steps=20 # rfdiffusion repo suggests 20steps is usually sufficient

    def _validate_input(self,pdb_str):
        return True
    
    def _make_config(self,contig_statement:str,pdb_path:str,outpath:str='out'):
        """ creates a hydra config file for the run script"""
        import hydra
        # with hydra.initialize(version_base=None, config_path=self.rel_config_path):
        cfg = hydra.compose(
            config_name="base", 
            overrides=[
                f'contigmap.contigs=[{contig_statement}]',
                f'inference.output_prefix={outpath}/output',
                f'inference.num_designs={self.num_designs}',
                f'inference.model_directory_path={self.model_path}',
                f'inference.input_pdb={pdb_path}',
                f'diffuser.T={self.steps}'
            ],
            return_hydra_config=True,
        )
        return cfg
    
    def _dummy_hydra(self):
        import os
        from omegaconf import OmegaConf
        hydra_runtime = OmegaConf.create({
            "runtime": {
                "output_dir": "/path/to/outputs",  
                "cwd": os.getcwd()
            },
            "job": {
                "name": "manual_job",
                "num": 0
            }
        })
        return hydra_runtime

    def _run_inference(self,input_pdb:str, start_idx:int, end_idx:int):
        """ runs inference script with fixed environment 
        
        parameters
        -----------
        input_pdb:
            the pdb string to generate backbone for

        idxs are inclusive (of mask) and based on indexing in the pdb file
        ie idx for start and end will both be generated
        """
        import sys
        sys.path.append(self.script_path)
        from run_inference import main as mn
        from Bio.PDB.Polypeptide import d3_to_index, dindex_to_1
        from Bio.PDB import PDBParser
        import hydra
        from omegaconf import OmegaConf
        from hydra.core.hydra_config import HydraConfig
        
        with (
            tempfile.TemporaryDirectory() as tmpdirname,
            tempfile.TemporaryDirectory() as in_tmpdirname):

            input_pdb_path = os.path.join(in_tmpdirname, 'input.pdb')
            with open(input_pdb_path, 'w') as f:
                f.write(input_pdb)

            x_len = end_idx - start_idx + 1

            mysplit = input_pdb.split('\n')[:-1]
            idxs = set()
            for i,v in enumerate(mysplit):
                if v.startswith('ATOM'):
                    idxs.add(int(v[22:26].strip()))
            seq_final_pos = max(idxs)

            contigmap = f"A1-{start_idx-1}/{x_len}-{x_len}/A{end_idx+1}-{seq_final_pos}"
            print(contigmap)
                
            with hydra.initialize(version_base=None, config_path=self.rel_config_path):
                cfg = self._make_config(contigmap, input_pdb_path, outpath=tmpdirname)
                cfg = OmegaConf.merge(
                        {"hydra": self._dummy_hydra()},
                        cfg
                    )
                HydraConfig.instance().set_config(cfg)
                mn(cfg)
            texts = []
            for i in range(self.num_designs):
                with open(f'{tmpdirname}/output_{i}.pdb','r') as f:
                    pdbtext = f.read()
                    texts.append(pdbtext)
        return texts
    
    def predict(self, context, model_input : List[Dict[str,str]], params=None) -> List[str]:
        """ Generate structure of protein of given length
        parameters
        --------

        context:
            The mlflow context of the model. Gathered by load_context()
        
        model_input:
            A list of dicts (pdb, start_idx, end_idx). Should only contain one entry in the list.
            start_idx and end_idx positions are 1-indexed and are includive to the mask for inpaint
            the pdb string should be of a pdb file that is 1-indexed for residues, no hetatm, single A chain

        params: Optional[Dict[str, Any]]
            Additional parameters
        """
        if len(model_input)>1:
            raise ValueError("input must be a list with a single entry")

        # pdb_texts = self._run_inference(model_input[0])
        pdb_texts = self._run_inference(model_input[0]['pdb'], int(model_input[0]['start_idx']), int(model_input[0]['end_idx']))
        return pdb_texts

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Unconditioned version

# COMMAND ----------

model = RFDiffusionUnconditional()
repo_path = f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}/RFdiffusion/"
artifacts={
    "script_path" : os.path.join(repo_path,"scripts"),
    "model_path" : os.path.join(repo_path,"models"),
    "example_path" : os.path.join(repo_path,"examples"),
    "config_path" : os.path.join(repo_path,"config/inference"),
}

model.load_context(mlflow.pyfunc.PythonModelContext(artifacts=artifacts, model_config=dict()))
pdb = model._run_inference(100)
pdb

# COMMAND ----------

# MAGIC %md
# MAGIC #### test the inpainting version
# MAGIC  - first make a dummy pdb that's been formatted correctly 

# COMMAND ----------

from Bio.PDB import PDBList
from Bio.PDB import PDBParser
from Bio import PDB
parser = PDBParser()

import requests
with tempfile.TemporaryDirectory() as tmpdirname:
    response = requests.get("https://files.rcsb.org/download/8dgr.pdb")
    pdb_file_path = f"{tmpdirname}/8dgr.pdb"
    with open(pdb_file_path, 'wb') as file:
        file.write(response.content)
    structure = parser.get_structure("8DGR", pdb_file_path)

def extract_chain_reindex(structure, chain_id='A'):
    # Extract chain A
    chain = structure[0][chain_id]
    
    # Create a new structure with only chain A & 1-indexed
    new_structure = PDB.Structure.Structure('new_structure')
    new_model = PDB.Model.Model(0)
    new_chain = PDB.Chain.Chain(chain_id)
    
    # Reindex residues starting from 1
    for i, residue in enumerate(chain, start=1):
        if residue.id[0] == ' ':  # Ensure no HETATM
            residue.id = (' ', i, ' ')
            new_chain.add(residue)
    
    new_model.add(new_chain)
    new_structure.add(new_model)
    
    # Save the new structure to a PDB file
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    with tempfile.NamedTemporaryFile(suffix='.pdb') as f:
        io.save(f.name)
        with open(f.name, 'r') as f_handle:
            pdb_text = f_handle.read()
    return pdb_text

# COMMAND ----------

model = RFDiffusionInpainting()
repo_path = f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}/RFdiffusion/"
artifacts={
    "script_path" : os.path.join(repo_path,"scripts"),
    "model_path" : os.path.join(repo_path,"models"),
    "example_path" : os.path.join(repo_path,"examples"),
    "config_path" : os.path.join(repo_path,"config/inference"),
}

model.load_context(mlflow.pyfunc.PythonModelContext(artifacts=artifacts, model_config=dict()))

# mask our pdb at residues 12-22 inclusive and generate new protein backbone
pdbs = model._run_inference( extract_chain_reindex(structure), 12, 22 )

# COMMAND ----------

pdbs[0].split('\n')[:10]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Begin the Model registration
# MAGIC  - first set input examples (to keep with the model)
# MAGIC  - and the model signatures

# COMMAND ----------

signature = mlflow.models.signature.ModelSignature(
    inputs = Schema([ColSpec(type="string")]),
    outputs = Schema([ColSpec(type="string")]),
    params = None
)


context = mlflow.pyfunc.PythonModelContext(artifacts=artifacts, model_config=dict())
input_example=[
    {
        'pdb':extract_chain_reindex(structure),
        'start_idx' : 12,
        'end_idx': 22 
    }
]
inpaint_signature = mlflow.models.infer_signature(
    input_example,
    model.predict(context, input_example)
)
print(inpaint_signature)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Perform the model registration

# COMMAND ----------

from databricks.sdk import WorkspaceClient

def set_mlflow_experiment(experiment_tag, user_email):    
    w = WorkspaceClient()
    mlflow_experiment_base_path = "Shared/dbx_genesis_workbench_models"
    w.workspace.mkdirs(f"/Workspace/{mlflow_experiment_base_path}")
    experiment_path = f"/{mlflow_experiment_base_path}/{experiment_tag}"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    return mlflow.set_experiment(experiment_path)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

repo_path = f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}/RFdiffusion/"

experiment = set_mlflow_experiment(experiment_tag=EXPERIMENT_NAME, user_email=USER_EMAIL)

with mlflow.start_run(run_name='rfdiffusion_unconditional', experiment_id=experiment.experiment_id):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="rfdiffusion",
        python_model=RFDiffusionUnconditional(),
        artifacts={
            "script_path" : os.path.join(repo_path,"scripts"),
            "model_path" : os.path.join(repo_path,"models"),
            "example_path" : os.path.join(repo_path,"examples"),
            "config_path" : os.path.join(repo_path,"config/inference"),
        },
        input_example=["100"],
        signature=signature,
        conda_env='rfd_env.yml',
        registered_model_name=f"{CATALOG}.{SCHEMA}.rfdiffusion_unconditional"
    )

with mlflow.start_run(run_name='rfdiffusion_inpainting', experiment_id=experiment.experiment_id):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="rfdiffusion",
        python_model=RFDiffusionInpainting(),
        artifacts={
            "script_path" : os.path.join(repo_path,"scripts"),
            "model_path" : os.path.join(repo_path,"models"),
            "example_path" : os.path.join(repo_path,"examples"),
            "config_path" : os.path.join(repo_path,"config/inference"),
        },
        input_example=input_example,
        signature=inpaint_signature,
        conda_env='rfd_env.yml',
        registered_model_name=f"{CATALOG}.{SCHEMA}.rfdiffusion_inpainting"
    )

import os
import requests
import json
import logging
import mlflow
import pandas as pd
from typing import Tuple, Optional
import tempfile
from Bio.PDB import PDBParser
from .structure_utils import _cif_to_pdb_str, select_and_align
from databricks.sdk import WorkspaceClient
from genesis_workbench.models import set_mlflow_experiment
from genesis_workbench.workbench import UserInfo, execute_workflow


def start_run_alphafold_job(protein_sequence:str,
                            mlflow_experiment_name:str,
                            mlflow_run_name:str,
                            user_info:UserInfo):
    
    experiment = set_mlflow_experiment(experiment_tag = mlflow_experiment_name,
                                       user_email = user_info.user_email,
                                       host = None, #host_name, Until we a have a way to grant user token permission to workspace api
                                       token = None #user_info.user_access_token
                                       )
    mlflow_run_id = ""
    job_run_id = ""
    with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("protein_sequence",protein_sequence)           
    
        job_run_id = execute_workflow(
            job_id=os.environ["RUN_ALPHAFOLD_JOB_ID"],
            params={
                "catalog" : os.environ["CORE_CATALOG_NAME"],
                "schema" : os.environ["CORE_SCHEMA_NAME"],
                "run_id" : mlflow_run_id,
                "protein_sequence" : protein_sequence,
                "user_email" : user_info.user_email
            }
        )
        mlflow.set_tag("origin","genesis_workbench") 
        mlflow.set_tag("feature", "alphafold")       
        mlflow.set_tag("created_by",user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status","started")

    return job_run_id


def search_alphafold_runs_by_run_name(user_email: str, run_name:str):
    """
    Searches for runs with the given run name and returns a pandas dataframe """

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    #find all experiments thats used by genesis workbench
    experiment_list = mlflow.search_experiments(filter_string=f"tags.used_by_genesis_workbench='yes' ")
    if len(experiment_list)==0:
        return pd.DataFrame()
    
    experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(experiments.keys())

    #find all runs with the given run name
    experiment_run_search_results = mlflow.search_runs(
        filter_string=f"tags.feature='alphafold' AND \
            tags.created_by='{user_email}' AND \
                tags.origin='genesis_workbench'", 
        experiment_ids=experiment_ids)
    
    #filter the runs with the given run name anywhere in the run_name
    filtered_runs = experiment_run_search_results[experiment_run_search_results['tags.mlflow.runName'].str.contains(run_name, case=False, na=False)]
    if len(filtered_runs)==0:
        return pd.DataFrame()
    
    #format the results
    filtered_runs['experiment_name'] = filtered_runs['experiment_id'].map(experiments)
    return_results = filtered_runs[['run_id','tags.mlflow.runName','experiment_name','params.protein_sequence','start_time','tags.job_status']]
    return_results.columns = ['run_id','run_name','experiment_name','protein_sequence','start_time','status']
    return return_results

def search_alphafold_runs_by_experiment_name(user_email: str, experiment_name:str):
    """ Searches for runs with the given experiment name and returns a pandas dataframe """

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    #find all experiments thats used by genesis workbench
    experiment_list = mlflow.search_experiments(filter_string=f"tags.used_by_genesis_workbench='yes' ")
    experiments = {exp.experiment_id: exp.name.split("/")[-1] 
                   for exp in experiment_list 
                    if experiment_name.upper() in exp.name.split("/")[-1].upper()}
    
    if len(experiments)==0:
        return pd.DataFrame()
    
    experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(experiments.keys())

    #return all alphafold runs in the experiments
    experiment_run_search_results = mlflow.search_runs(
        filter_string=f"tags.feature='alphafold' AND \
            tags.created_by='{user_email}' AND \
                tags.origin='genesis_workbench'", 
        experiment_ids=experiment_ids)
    
    if len(experiment_run_search_results) == 0:
        return pd.DataFrame()
    
    #format the results
    experiment_run_search_results['experiment_name'] = experiment_run_search_results['experiment_id'].map(experiments)
    return_results = experiment_run_search_results[['run_id','tags.mlflow.runName','experiment_name','params.protein_sequence','start_time','tags.job_status']]
    return_results.columns = ['run_id','run_name','experiment_name','protein_sequence','start_time','status']
    return return_results


def pull_alphafold_run(run_id : str ='run') -> str:
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]

    print(f"Fetching result for run id: {run_id}")
    w = WorkspaceClient()
    response = w.files.download(
        f'/Volumes/{catalog}/{schema}/alphafold/results/{run_id}/{run_id}/ranked_0.pdb'
    )
    pdb_str = str(response.contents.read(), encoding='utf-8')
    return pdb_str

def pull_pdbmmcif(pdb_code : str ='4ykk') -> str:
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    pdb_code = pdb_code.lower()
    w = WorkspaceClient()
    response = w.files.download(
        f'/Volumes/{catalog}/{schema}/alphafold/datasets/pdb_mmcif/mmcif_files/{pdb_code}.cif'
    )
    cif_str = str(response.contents.read(), encoding='utf-8')
    return _cif_to_pdb_str(cif_str)
  
def apply_pdb_header(pdb_str: str, name: str) -> str:
    header = f"""HEADER    "{name}"                           00-JAN-00   0XXX 
    TITLE     "{name}"                         
    COMPND    MOL_ID: 1;                                                            
    COMPND   2 MOLECULE: {name};                          
    COMPND   3 CHAIN: A;""" 
    return header + pdb_str
  
def af_collect_and_align(run_id:str, 
                         run_name : str, 
                         pdb_code : Optional[str] = None, 
                         include_pdb : bool = False) -> Tuple[str]:
    
    logging.info("collect run")
    pdb_run = pull_alphafold_run(run_id=run_id)
    logging.info("add header")
    pdb_run = apply_pdb_header(pdb_run, run_name)
    true_structure_str = ""
    af_structure_str = pdb_run
    if include_pdb:
      logging.info("collect PDB entry")
      pdb_mmcif = pull_pdbmmcif(pdb_code)
      # strings to biopdb structures
      with tempfile.TemporaryDirectory() as tmpdir:
          true_pdb_path = os.path.join(tmpdir, 'true_pdb.pdb')
          af_pdb_path = os.path.join(tmpdir, 'af_pdb.pdb') 
          with open(true_pdb_path, 'w') as f:
              f.write(pdb_mmcif)
          with open(af_pdb_path, 'w') as f:
              f.write(pdb_run)
          
          true_structure = PDBParser().get_structure('true',true_pdb_path)
          af_structure = PDBParser().get_structure('af',af_pdb_path)

          logging.info("do slect and align")
          true_structure_str, af_structure_str = select_and_align(
              true_structure, af_structure
          )

          logging.info("more headers")          
          true_structure_str = apply_pdb_header(true_structure_str, run_name)
          af_structure_str = apply_pdb_header(af_structure_str, "alphafold2 prediction")
    return pdb_run, true_structure_str, af_structure_str

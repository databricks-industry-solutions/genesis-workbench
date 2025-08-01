import os
import requests
import json
import logging
import mlflow

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
                "schema" : os.environ["CORE_CATALOG_NAME"],
                "run_id" : mlflow_run_id,
                "protein_sequence" : protein_sequence,
                "user_email" : user_info.user_email
            }
        )
        mlflow.set_tag("origin","genesis_workbench")        
        mlflow.set_tag("created_by",user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status","started")
        
    return job_run_id


   
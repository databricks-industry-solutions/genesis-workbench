import os
import requests
import json
import logging
import mlflow
import pandas as pd
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
    return_results = filtered_runs[['tags.mlflow.runName','experiment_name','params.protein_sequence','start_time','tags.job_status']]
    return_results.columns = ['run_name','experiment_name','protein_sequence','start_time','status']
    return return_results

def search_alphafold_runs_by_experiment_name(user_email: str, experiment_name:str):
    """ Searches for runs with the given experiment name and returns a pandas dataframe """

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
    return_results = experiment_run_search_results[['tags.mlflow.runName','experiment_name','params.protein_sequence','start_time','tags.job_status']]
    return_results.columns = ['run_name','experiment_name','protein_sequence','start_time','status']
    return return_results
from databricks.sdk import WorkspaceClient
from utils.streamlit_helper import UserInfo
from genesis_workbench.models import set_mlflow_experiment
import os

def start_scanpy_job(
    data_path: str,
    mlflow_experiment: str,
    mlflow_run_name: str,
    gene_name_column: str,
    min_genes: int,
    min_cells: int,
    pct_counts_mt: float,
    n_genes_by_counts: int,
    target_sum: int,
    n_top_genes: int,
    n_pcs: int,
    leiden_resolution: float,
    user_info: UserInfo
):
    """
    Trigger the scanpy analysis job with specified parameters
    Returns: (job_id, run_id) tuple
    """
    w = WorkspaceClient()
    
    # Get job ID from settings table (loaded into env vars by initialize())
    scanpy_job_id = os.environ.get("RUN_SCANPY_JOB_ID")
    if not scanpy_job_id:
        raise RuntimeError("Scanpy job not registered. Please deploy the single_cell module first.")
    
    # Get full experiment path using user's mlflow folder
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment,  # Simple name from user
        user_email=user_info.user_email
    )
    
    job_run = w.jobs.run_now(
        job_id=scanpy_job_id,
        job_parameters={
            "catalog": os.environ["CORE_CATALOG_NAME"],
            "schema": os.environ["CORE_SCHEMA_NAME"],
            "user_email": user_info.user_email,
            "data_path": data_path,
            "mlflow_experiment": experiment.name,  # Full path like /Users/email/folder/experiment_name
            "mlflow_run_name": mlflow_run_name,
            "gene_name_column": gene_name_column,
            "min_genes": str(min_genes),
            "min_cells": str(min_cells),
            "pct_counts_mt": str(pct_counts_mt),
            "n_genes_by_counts": str(n_genes_by_counts),
            "target_sum": str(target_sum),
            "n_top_genes": str(n_top_genes),
            "n_pcs": str(n_pcs),
            "leiden_resolution": str(leiden_resolution),
        }
    )
    
    return scanpy_job_id, job_run.run_id


def start_rapids_singlecell_job(
    data_path: str,
    mlflow_experiment: str,
    mlflow_run_name: str,
    gene_name_column: str,
    min_genes: int,
    min_cells: int,
    pct_counts_mt: float,
    n_genes_by_counts: int,
    target_sum: int,
    n_top_genes: int,
    n_pcs: int,
    leiden_resolution: float,
    user_info: UserInfo
):
    """
    Trigger the rapids-singlecell analysis job with specified parameters
    Returns: (job_id, run_id) tuple
    
    TODO: Implement this function when rapids-singlecell job is deployed
    """
    raise NotImplementedError("rapids-singlecell mode is not yet implemented")
    
    # When implemented, follow this pattern (similar to scanpy):
    # w = WorkspaceClient()
    # 
    # # Get job ID from settings table (loaded into env vars by initialize())
    # rapids_job_id = os.environ.get("RUN_RAPIDS_SINGLECELL_JOB_ID")
    # if not rapids_job_id:
    #     raise RuntimeError("rapids-singlecell job not registered. Please deploy the rapids-singlecell module first.")
    # 
    # # Get full experiment path using user's mlflow folder
    # experiment = set_mlflow_experiment(
    #     experiment_tag=mlflow_experiment,
    #     user_email=user_info.user_email
    # )
    # 
    # job_run = w.jobs.run_now(
    #     job_id=rapids_job_id,
    #     job_parameters={
    #         "catalog": os.environ["CORE_CATALOG_NAME"],
    #         "schema": os.environ["CORE_SCHEMA_NAME"],
    #         "user_email": user_info.user_email,
    #         "data_path": data_path,
    #         "mlflow_experiment": experiment.name,
    #         "mlflow_run_name": mlflow_run_name,
    #         "gene_name_column": gene_name_column,
    #         "min_genes": str(min_genes),
    #         "min_cells": str(min_cells),
    #         "pct_counts_mt": str(pct_counts_mt),
    #         "n_genes_by_counts": str(n_genes_by_counts),
    #         "target_sum": str(target_sum),
    #         "n_top_genes": str(n_top_genes),
    #         "n_pcs": str(n_pcs),
    #         "leiden_resolution": str(leiden_resolution),
    #     }
    # )
    # 
    # return rapids_job_id, job_run.run_id

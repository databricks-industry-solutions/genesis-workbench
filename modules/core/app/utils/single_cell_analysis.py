from databricks.sdk import WorkspaceClient
from utils.streamlit_helper import UserInfo
from genesis_workbench.models import set_mlflow_experiment
from mlflow.tracking import MlflowClient
import mlflow
import pandas as pd
import tempfile
import os

def start_scanpy_job(
    data_path: str,
    mlflow_experiment: str,
    mlflow_run_name: str,
    gene_name_column: str,
    species: str,
    min_genes: int,
    min_cells: int,
    pct_counts_mt: float,
    n_genes_by_counts: int,
    target_sum: int,
    n_top_genes: int,
    n_pcs: int,
    cluster_resolution: float,
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
            "species": species,
            "min_genes": str(min_genes),
            "min_cells": str(min_cells),
            "pct_counts_mt": str(pct_counts_mt),
            "n_genes_by_counts": str(n_genes_by_counts),
            "target_sum": str(target_sum),
            "n_top_genes": str(n_top_genes),
            "n_pcs": str(n_pcs),
            "cluster_resolution": str(cluster_resolution),
        }
    )
    
    return scanpy_job_id, job_run.run_id


def start_rapids_singlecell_job(
    data_path: str,
    mlflow_experiment: str,
    mlflow_run_name: str,
    gene_name_column: str,
    species: str,
    min_genes: int,
    min_cells: int,
    pct_counts_mt: float,
    n_genes_by_counts: int,
    target_sum: int,
    n_top_genes: int,
    n_pcs: int,
    cluster_resolution: float,
    user_info: UserInfo
):
    """
    Trigger the rapids-singlecell analysis job with specified parameters
    Returns: (job_id, run_id) tuple
    """
    w = WorkspaceClient()
    
    # Get job ID from settings table (loaded into env vars by initialize())
    rapids_job_id = os.environ.get("RUN_RAPIDSSINGLECELL_JOB_ID")
    if not rapids_job_id:
        raise RuntimeError("rapids-singlecell job not registered. Please deploy the rapids-singlecell module first.")
    
    # Get full experiment path using user's mlflow folder
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment,  # Simple name from user
        user_email=user_info.user_email
    )
    
    job_run = w.jobs.run_now(
        job_id=rapids_job_id,
        job_parameters={
            "catalog": os.environ["CORE_CATALOG_NAME"],
            "schema": os.environ["CORE_SCHEMA_NAME"],
            "user_email": user_info.user_email,
            "data_path": data_path,
            "mlflow_experiment": experiment.name,  # Full path like /Users/email/folder/experiment_name
            "mlflow_run_name": mlflow_run_name,
            "gene_name_column": gene_name_column,
            "species": species,
            "min_genes": str(min_genes),
            "min_cells": str(min_cells),
            "pct_counts_mt": str(pct_counts_mt),
            "n_genes_by_counts": str(n_genes_by_counts),
            "target_sum": str(target_sum),
            "n_top_genes": str(n_top_genes),
            "n_pcs": str(n_pcs),
            "cluster_resolution": str(cluster_resolution),
        }
    )
    
    return rapids_job_id, job_run.run_id


def download_singlecell_markers_df(run_id: str) -> pd.DataFrame:
    """
    Download the markers_flat.parquet from an MLflow single-cell analysis run
    
    Works with scanpy, rapids-singlecell, or any tool that produces the standard
    markers_flat.parquet artifact.
    
    Args:
        run_id: The MLflow run ID
    
    Returns:
        DataFrame with cells, embeddings, and marker expression
    """
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()
    
    # Download the parquet artifact
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = "markers_flat.parquet"
        local_file = client.download_artifacts(run_id, artifact_path, dst_path=tmpdir)
        df = pd.read_parquet(local_file)
    
    return df


def download_cluster_markers_mapping(run_id: str) -> pd.DataFrame:
    """
    Download the top_markers_per_cluster.csv from an MLflow run.
    
    Returns a DataFrame where columns are cluster IDs and values are ranked marker genes
    (as determined by Wilcoxon rank-sum test in scanpy).
    
    Args:
        run_id: The MLflow run ID
    
    Returns:
        DataFrame with columns as cluster IDs, rows as gene rankings
    """
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()
    
    # Download the CSV artifact
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = "top_markers_per_cluster.csv"
        local_file = client.download_artifacts(run_id, artifact_path, dst_path=tmpdir)
        df = pd.read_csv(local_file)
    
    return df


def get_mlflow_run_url(run_id: str) -> str:
    """
    Construct the URL to an MLflow run
    
    Args:
        run_id: The MLflow run ID
    
    Returns:
        Full URL to the MLflow run in Databricks
    """
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()
    
    # Get run info to get experiment ID
    run = client.get_run(run_id)
    experiment_id = run.info.experiment_id
    
    # Construct URL
    host_name = os.getenv("DATABRICKS_HOSTNAME", "")
    if not host_name.startswith("https://"):
        host_name = "https://" + host_name
    
    url = f"{host_name}/ml/experiments/{experiment_id}/runs/{run_id}"
    return url


def search_singlecell_runs(user_email: str, processing_mode: str = None, days_back: int = None) -> pd.DataFrame:
    """
    Search for single-cell processing runs created by the user
    
    Works across all single-cell processing modes (scanpy, rapids-singlecell, etc.)
    
    Args:
        user_email: Email of the user
        processing_mode: Optional filter for specific mode (e.g., 'scanpy', 'rapids-singlecell')
        days_back: Optional filter to only show runs from last N days (None = all time)
    
    Returns:
        DataFrame with run information (run_id, run_name, experiment_name, start_time, status)
    """
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    
    # Find all experiments used by genesis workbench
    experiment_list = mlflow.search_experiments(
        filter_string=f"tags.used_by_genesis_workbench='yes'"
    )
    
    if len(experiment_list) == 0:
        return pd.DataFrame()
    
    # Get experiment IDs and names
    experiments = {exp.experiment_id: exp.name for exp in experiment_list}
    experiment_ids = list(experiments.keys())
    
    # Build filter string - only show successful runs
    filter_parts = [
        f"tags.created_by='{user_email}'", 
        "tags.origin='genesis_workbench'",
        "attributes.status='FINISHED'"  # Only successful runs
    ]
    if processing_mode:
        filter_parts.append(f"tags.processing_mode='{processing_mode}'")
    
    # Search for single-cell runs created by this user
    try:
        runs = mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string=" AND ".join(filter_parts),
            order_by=["start_time DESC"],
            max_results=100
        )
    except Exception:
        # Fallback if tag filtering doesn't work
        runs = mlflow.search_runs(
            experiment_ids=experiment_ids,
            order_by=["start_time DESC"],
            max_results=100
        )
    
    if len(runs) == 0:
        return pd.DataFrame()
    
    # Apply date filter if specified
    if days_back is not None and days_back >= 0:
        if days_back == 0:
            # For "Today", use start of today (midnight)
            cutoff_date = pd.Timestamp.now().normalize()  # Sets time to 00:00:00
        else:
            # For other filters, go back N days from now
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_back)
        
        # Handle timezone-aware vs timezone-naive timestamps
        # Convert both to timezone-naive for safe comparison
        if hasattr(runs['start_time'].iloc[0], 'tz') and runs['start_time'].iloc[0].tz is not None:
            # Timestamps are timezone-aware, convert to timezone-naive (local time)
            runs_start_time = runs['start_time'].dt.tz_localize(None)
        else:
            # Already timezone-naive
            runs_start_time = runs['start_time']
        
        runs = runs[runs_start_time >= cutoff_date]
        
        if len(runs) == 0:
            return pd.DataFrame()
    
    # Format the results
    runs['experiment_name'] = runs['experiment_id'].map(experiments)
    runs['experiment_simple'] = runs['experiment_name'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
    
    # Select and rename columns
    result_columns = ['run_id', 'tags.mlflow.runName', 'experiment_simple', 'start_time', 'status']
    available_columns = [c for c in result_columns if c in runs.columns]
    
    result = runs[available_columns].copy()
    result.columns = ['run_id', 'run_name', 'experiment', 'start_time', 'status'][:len(available_columns)]
    
    return result

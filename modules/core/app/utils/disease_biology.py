
import os
import mlflow
import pandas as pd
from genesis_workbench.workbench import UserInfo, execute_workflow, execute_select_query
from genesis_workbench.models import set_mlflow_experiment


def start_parabricks_alignment(user_info: UserInfo,
                                fastq_r1: str,
                                fastq_r2: str,
                                reference_genome_path: str,
                                output_volume_path: str,
                                mlflow_experiment_name: str,
                                mlflow_run_name: str):
    """Start a Parabricks germline alignment job.

    Creates an MLflow run, then triggers the Databricks workflow.
    Returns the Databricks job run id.
    """
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None
    )

    with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("fastq_r1", fastq_r1)
        mlflow.log_param("fastq_r2", fastq_r2)
        mlflow.log_param("reference_genome", reference_genome_path)

        job_run_id = execute_workflow(
            job_id=os.environ["PARABRICKS_ALIGNMENT_JOB_ID"],
            params={
                "catalog": os.environ["CORE_CATALOG_NAME"],
                "schema": os.environ["CORE_SCHEMA_NAME"],
                "fastq_r1": fastq_r1,
                "fastq_r2": fastq_r2,
                "reference_genome_path": reference_genome_path,
                "output_volume_path": output_volume_path,
                "mlflow_run_id": mlflow_run_id,
                "user_email": user_info.user_email
            }
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "gwas_alignment")
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")

    return job_run_id


def start_gwas_analysis(user_info: UserInfo,
                         vcf_path: str,
                         phenotype_path: str,
                         phenotype_column: str,
                         contigs: str,
                         hwe_cutoff: str,
                         pvalue_threshold: str,
                         mlflow_experiment_name: str,
                         mlflow_run_name: str):
    """Start a GWAS analysis job using Glow.

    Creates an MLflow run, then triggers the Databricks workflow.
    Returns the Databricks job run id.
    """
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None
    )

    with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("vcf_path", vcf_path)
        mlflow.log_param("phenotype_path", phenotype_path)
        mlflow.log_param("phenotype_column", phenotype_column)
        mlflow.log_param("contigs", contigs)
        mlflow.log_param("hwe_cutoff", hwe_cutoff)

        job_run_id = execute_workflow(
            job_id=os.environ["GWAS_ANALYSIS_JOB_ID"],
            params={
                "catalog": os.environ["CORE_CATALOG_NAME"],
                "schema": os.environ["CORE_SCHEMA_NAME"],
                "vcf_path": vcf_path,
                "phenotype_path": phenotype_path,
                "phenotype_column": phenotype_column,
                "contigs": contigs,
                "hwe_cutoff": hwe_cutoff,
                "pvalue_threshold": pvalue_threshold,
                "mlflow_run_id": mlflow_run_id,
                "user_email": user_info.user_email
            }
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "gwas")
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")

    return job_run_id


def search_gwas_runs_by_run_name(user_email: str, run_name: str) -> pd.DataFrame:
    """Search GWAS runs by run name substring."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(experiments.keys())

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='gwas' AND "
            f"tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    filtered = runs[runs['tags.mlflow.runName'].str.contains(run_name, case=False, na=False)]
    if len(filtered) == 0:
        return pd.DataFrame()

    filtered['experiment_name'] = filtered['experiment_id'].map(experiments)
    result = filtered[['run_id', 'tags.mlflow.runName', 'experiment_name',
                        'params.vcf_path', 'start_time', 'tags.job_status']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'vcf_path', 'start_time', 'status']
    return result


def search_gwas_runs_by_experiment_name(user_email: str, experiment_name: str) -> pd.DataFrame:
    """Search GWAS runs by experiment name substring."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    matching = {exp.experiment_id: exp.name.split("/")[-1]
                for exp in experiment_list
                if experiment_name.upper() in exp.name.split("/")[-1].upper()}

    if len(matching) == 0:
        return pd.DataFrame()

    all_experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(matching.keys())

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='gwas' AND "
            f"tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    runs['experiment_name'] = runs['experiment_id'].map(all_experiments)
    result = runs[['run_id', 'tags.mlflow.runName', 'experiment_name',
                    'params.vcf_path', 'start_time', 'tags.job_status']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'vcf_path', 'start_time', 'status']
    return result


def search_variant_calling_runs_by_run_name(user_email: str, run_name: str) -> pd.DataFrame:
    """Search variant calling runs by run name substring."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(experiments.keys())

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='gwas_alignment' AND "
            f"tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    filtered = runs[runs['tags.mlflow.runName'].str.contains(run_name, case=False, na=False)]
    if len(filtered) == 0:
        return pd.DataFrame()

    filtered['experiment_name'] = filtered['experiment_id'].map(experiments)
    result = filtered[['run_id', 'tags.mlflow.runName', 'experiment_name',
                        'params.fastq_r1', 'start_time', 'tags.job_status']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'fastq_r1', 'start_time', 'status']
    return result


def search_variant_calling_runs_by_experiment_name(user_email: str, experiment_name: str) -> pd.DataFrame:
    """Search variant calling runs by experiment name substring."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    matching = {exp.experiment_id: exp.name.split("/")[-1]
                for exp in experiment_list
                if experiment_name.upper() in exp.name.split("/")[-1].upper()}

    if len(matching) == 0:
        return pd.DataFrame()

    all_experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(matching.keys())

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='gwas_alignment' AND "
            f"tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    runs['experiment_name'] = runs['experiment_id'].map(all_experiments)
    result = runs[['run_id', 'tags.mlflow.runName', 'experiment_name',
                    'params.fastq_r1', 'start_time', 'tags.job_status']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'fastq_r1', 'start_time', 'status']
    return result


def list_successful_variant_calling_runs(user_email: str) -> pd.DataFrame:
    """List all successful variant calling runs for the GWAS run picker."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if len(experiment_list) == 0:
        return pd.DataFrame()

    experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    experiment_ids = list(experiments.keys())

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='gwas_alignment' AND "
            f"tags.job_status='alignment_complete' AND "
            f"tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    runs['experiment_name'] = runs['experiment_id'].map(experiments)
    result = runs[['run_id', 'tags.mlflow.runName', 'experiment_name',
                    'params.output_vcf', 'start_time']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'output_vcf', 'start_time']
    return result


def pull_gwas_results(run_id: str) -> pd.DataFrame:
    """Pull GWAS results from the Delta table for a given run."""
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    table = f"gwas_results_{run_id.replace('-', '_')}"

    query = f"""
        SELECT contigName, start, pvalue, referenceAllele, alternateAlleles,
               -log(10, pvalue) as neg_log_pval
        FROM {catalog}.{schema}.{table}
        WHERE pvalue IS NOT NULL
        ORDER BY pvalue ASC
    """
    return execute_select_query(query)

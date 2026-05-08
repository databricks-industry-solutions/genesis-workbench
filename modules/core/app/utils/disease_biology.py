
import os
import mlflow
import pandas as pd
from genesis_workbench.workbench import UserInfo, execute_workflow, execute_select_query
from genesis_workbench.models import set_mlflow_experiment


def _derive_status(row):
    """Derive display status from job_status tag.

    The mark_success/mark_failure tasks in each job workflow update the
    job_status tag to the final value (e.g. 'alignment_complete', 'failed').
    'started' means the job is still running or the completion task hasn't run yet.
    """
    job_status = row.get("tags.job_status", "")
    if job_status:
        return job_status
    return "unknown"


# Progress visualization for search results
_PROGRESS_MAP = {
    # Variant Calling (2 steps)
    "alignment_complete":   "🟩🟩",
    # GWAS Analysis (3 main tasks)
    "phenotype_prepared":   "🟩⬜⬜",
    "gwas_complete":        "🟩🟩🟩",
    # VCF Ingestion (1 main task)
    "ingestion_complete":   "🟩🟩",
    # Variant Annotation (3 main tasks)
    "annotation_complete":  "🟩🟩🟩",
    # Terminal states
    "failed":               "🟥",
    "unknown":              "⬜",
}


def add_progress_column(df, total_steps=2):
    """Add a visual progress column to a search results DataFrame."""
    if df.empty or "status" not in df.columns:
        return df
    df = df.copy()
    df["progress"] = df["status"].map(
        lambda s: _PROGRESS_MAP.get(s, f"🟩{'⬜' * (total_steps - 1)}" if s == "started" else "⬜" * total_steps)
    )
    # Reorder so progress is right after status
    cols = list(df.columns)
    if "progress" in cols and "status" in cols:
        cols.remove("progress")
        idx = cols.index("status") + 1
        cols.insert(idx, "progress")
        df = df[cols]
    return df


_BLINKING_DOT_CSS = """
<style>
@keyframes blink-orange { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }
.dot-in-progress { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                   background-color: #FF8C00; animation: blink-orange 1.2s infinite;
                   margin-right: 6px; vertical-align: middle; }
.dot-complete { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                background-color: #22C55E; margin-right: 6px; vertical-align: middle; }
.dot-failed { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
              background-color: #EF4444; margin-right: 6px; vertical-align: middle; }
.dot-unknown { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
               background-color: #9CA3AF; margin-right: 6px; vertical-align: middle; }
.run-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.run-table th { text-align: left; padding: 8px 12px; border-bottom: 2px solid #444;
                font-weight: 600; color: #999; }
.run-table td { padding: 8px 12px; border-bottom: 1px solid #333; }
.run-table tr:hover { background-color: rgba(255,255,255,0.03); }
</style>
"""

_IN_PROGRESS_STATUSES = {"started", "phenotype_prepared"}
_COMPLETE_STATUSES = {"alignment_complete", "gwas_complete", "ingestion_complete", "annotation_complete"}
_FAILED_STATUSES = {"failed"}


def _status_dot(status):
    if status in _IN_PROGRESS_STATUSES:
        return '<span class="dot-in-progress"></span>'
    elif status in _COMPLETE_STATUSES:
        return '<span class="dot-complete"></span>'
    elif status in _FAILED_STATUSES:
        return '<span class="dot-failed"></span>'
    return '<span class="dot-unknown"></span>'


def render_runs_html_table(df, hidden_columns=None):
    """Render a search results DataFrame as an HTML table with status dots."""
    if df.empty:
        return ""
    hidden_columns = hidden_columns or []
    display_cols = [c for c in df.columns if c not in hidden_columns]

    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for col in display_cols:
            val = row.get(col, "")
            if col == "status":
                dot = _status_dot(str(val))
                label = str(val).replace("_", " ").title()
                cells.append(f"<td>{dot}{label}</td>")
            else:
                cells.append(f"<td>{val}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    header_labels = [c.replace("_", " ").title() for c in display_cols]
    header = "".join(f"<th>{h}</th>" for h in header_labels)

    return (
        _BLINKING_DOT_CSS
        + f'<table class="run-table"><thead><tr>{header}</tr></thead>'
        + f'<tbody>{"".join(rows_html)}</tbody></table>'
    )


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
    fq_table = f"{catalog}.{schema}.{table}"

    # First check what columns exist
    try:
        cols_df = execute_select_query(f"DESCRIBE {fq_table}")
        available_cols = cols_df["col_name"].tolist() if "col_name" in cols_df.columns else []
        print(f"[pull_gwas_results] Table {fq_table} columns: {available_cols}")
    except Exception as e:
        print(f"[pull_gwas_results] Table {fq_table} not found: {e}")
        return pd.DataFrame()

    # Check row counts
    try:
        count_df = execute_select_query(f"SELECT count(*) as total, count(pvalue) as non_null_pvalue FROM {fq_table}")
        print(f"[pull_gwas_results] Rows: total={count_df['total'].iloc[0]}, non_null_pvalue={count_df['non_null_pvalue'].iloc[0]}")
    except Exception as e:
        print(f"[pull_gwas_results] Count query failed: {e}")

    query = f"""
        SELECT contigName, start, pvalue, referenceAllele, alternateAlleles, effect, phenotype,
               CASE WHEN pvalue IS NOT NULL AND pvalue > 0 THEN -log(10, pvalue) ELSE NULL END as neg_log_pval
        FROM {fq_table}
        WHERE pvalue IS NOT NULL
        ORDER BY pvalue ASC
        LIMIT 10000
    """
    return execute_select_query(query)


# ── VCF Ingestion ──

def start_vcf_ingestion(user_info: UserInfo,
                         vcf_path: str,
                         output_table_name: str,
                         mlflow_experiment_name: str,
                         mlflow_run_name: str):
    """Start a VCF-to-Delta ingestion job.

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
        mlflow.log_param("output_table_name", output_table_name)

        job_run_id = execute_workflow(
            job_id=os.environ["VCF_INGESTION_JOB_ID"],
            params={
                "catalog": os.environ["CORE_CATALOG_NAME"],
                "schema": os.environ["CORE_SCHEMA_NAME"],
                "sql_warehouse_id": os.environ["SQL_WAREHOUSE"],
                "vcf_path": vcf_path,
                "output_table_name": output_table_name,
                "mlflow_run_id": mlflow_run_id,
                "user_email": user_info.user_email
            }
        )
        catalog = os.environ["CORE_CATALOG_NAME"]
        schema = os.environ["CORE_SCHEMA_NAME"]
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "vcf_ingestion")
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")
        mlflow.set_tag("output_table", f"{catalog}.{schema}.{output_table_name}")

    return job_run_id


def search_vcf_ingestion_runs_by_run_name(user_email: str, run_name: str) -> pd.DataFrame:
    """Search VCF ingestion runs by run name substring."""
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
            f"tags.feature='vcf_ingestion' AND "
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


def search_vcf_ingestion_runs_by_experiment_name(user_email: str, experiment_name: str) -> pd.DataFrame:
    """Search VCF ingestion runs by experiment name substring."""
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
            f"tags.feature='vcf_ingestion' AND "
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


def list_successful_vcf_ingestion_runs(user_email: str) -> pd.DataFrame:
    """List all successful VCF ingestion runs for the annotation tab picker."""
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
            f"tags.feature='vcf_ingestion' AND "
            f"tags.job_status='ingestion_complete' AND "
            f"tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    runs['experiment_name'] = runs['experiment_id'].map(experiments)
    result = runs[['run_id', 'tags.mlflow.runName', 'experiment_name',
                    'tags.output_table', 'start_time']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'output_table', 'start_time']
    return result


# ── Variant Annotation ──

def start_variant_annotation(user_info: UserInfo,
                              variants_table: str,
                              gene_regions: str,
                              pathogenic_vcf_path: str,
                              mlflow_experiment_name: str,
                              mlflow_run_name: str,
                              gene_panel_mode: str = "custom"):
    """Start a variant annotation job.

    Creates an MLflow run, then triggers the Databricks workflow.
    Returns the Databricks job run id.

    Args:
        gene_panel_mode: "custom" for JSON gene regions, "acmg" for ACMG SF v3.2 panel.
    """
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None
    )

    with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("variants_table", variants_table)
        mlflow.log_param("gene_panel_mode", gene_panel_mode)
        if gene_panel_mode != "acmg":
            mlflow.log_param("gene_regions", gene_regions)

        job_run_id = execute_workflow(
            job_id=os.environ["VARIANT_ANNOTATION_JOB_ID"],
            params={
                "catalog": os.environ["CORE_CATALOG_NAME"],
                "schema": os.environ["CORE_SCHEMA_NAME"],
                "sql_warehouse_id": os.environ["SQL_WAREHOUSE"],
                "variants_table": variants_table,
                "gene_regions": gene_regions,
                "gene_panel_mode": gene_panel_mode,
                "pathogenic_vcf_path": pathogenic_vcf_path,
                "mlflow_run_id": mlflow_run_id,
                "run_name": mlflow_run_name,
                "user_email": user_info.user_email
            }
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "variant_annotation")
        mlflow.set_tag("run_name", mlflow_run_name)
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")

    return job_run_id


def search_variant_annotation_runs_by_run_name(user_email: str, run_name: str) -> pd.DataFrame:
    """Search variant annotation runs by run name substring."""
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
            f"tags.feature='variant_annotation' AND "
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
                        'params.variants_table', 'start_time', 'tags.job_status']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'variants_table', 'start_time', 'status']
    return result


def search_variant_annotation_runs_by_experiment_name(user_email: str, experiment_name: str) -> pd.DataFrame:
    """Search variant annotation runs by experiment name substring."""
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
            f"tags.feature='variant_annotation' AND "
            f"tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=experiment_ids
    )

    if len(runs) == 0:
        return pd.DataFrame()

    runs['experiment_name'] = runs['experiment_id'].map(all_experiments)
    result = runs[['run_id', 'tags.mlflow.runName', 'experiment_name',
                    'params.variants_table', 'start_time', 'tags.job_status']]
    result.columns = ['run_id', 'run_name', 'experiment_name', 'variants_table', 'start_time', 'status']
    return result


def pull_annotation_results(run_id: str, run_name: str = "") -> pd.DataFrame:
    """Pull pathogenic variant results for a given annotation run.

    Each variant_annotation run owns its own per-run pathogenic table
    (named `<base>__<sanitized_run_name>_<run_id_prefix>`). Prefer the
    `pathogenic_table` MLflow tag/param set by `03_save_results.py` —
    falls back to reconstructing the name from `(run_name, run_id)` if
    the tag is missing on older runs.
    """
    import re
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]

    run = mlflow.get_run(run_id)
    if not run_name:
        run_name = run.data.tags.get("run_name", "")

    # Trust the orchestrator-logged table name first; reconstruct only if missing.
    table = run.data.tags.get("pathogenic_table") or run.data.params.get("pathogenic_table")
    if not table:
        safe = re.sub(r"[^a-z0-9_]", "_", (run_name or "").lower())
        safe = re.sub(r"_+", "_", safe).strip("_")[:40] or "unnamed"
        suffix = f"{safe}_{(run_id or 'norun')[:8]}"
        table = f"{catalog}.{schema}.variant_annotation_pathogenic__{suffix}"

    # Check if ACMG columns (category, condition) exist in the table
    try:
        cols_df = execute_select_query(f"DESCRIBE {table}")
        available_cols = cols_df["col_name"].tolist() if "col_name" in cols_df.columns else []
    except Exception:
        available_cols = []

    extra_cols = ""
    if "category" in available_cols:
        extra_cols = "category, condition, "

    query = f"""
        SELECT gene, {extra_cols}chromosome, start as position, ref, alt, zygosity,
               array_join(clinical_significance, ', ') as clinical_significance,
               array_join(disease_name, ', ') as disease_name
        FROM {table}
        ORDER BY gene, position
    """
    return execute_select_query(query)

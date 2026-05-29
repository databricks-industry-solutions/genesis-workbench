"""Disease Biology — async-job dispatchers + MLflow result loaders for the
four workflows (Variant Calling / GWAS / VCF Ingestion / Variant Annotation).

Each workflow follows the AlphaFold2 pattern: dispatch a Databricks job
via job_id sourced from env, pre-create the MLflow run, search past runs
by run-name or experiment-name, and load results from Delta tables on
demand. Ported from modules/core/app/utils/disease_biology.py."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import mlflow
import pandas as pd
from genesis_workbench.models import set_mlflow_experiment
from genesis_workbench.workbench import UserInfo, execute_workflow
from mlflow.tracking import MlflowClient

from app.services.databricks_links import job_run_url
from app.services.workbench import execute_select_query

logger = logging.getLogger(__name__)


# ─── Common search-results dataclass ──────────────────────────────────────


@dataclass(frozen=True)
class DBRun:
    run_id: str
    run_name: str
    experiment_name: str
    status: str
    progress: str
    start_time_ms: int | None
    # Extra workflow-specific cell — fastq_r1 / vcf_path / variants_table.
    detail: str
    # Workspace UI URL for the dispatched job's run page. Empty when the
    # job_run_id tag is missing or the deploy didn't expose the orchestrator
    # job-id env var (in which case the frontend renders the name as text).
    run_url: str = ""


# MLflow feature tag → env var holding the orchestrator job id used to
# dispatch that feature's runs. Lookup is per-row so we can build a working
# "Open in workspace" link for every search hit.
_FEATURE_TO_JOB_ENV: dict[str, str] = {
    "gwas_alignment": "PARABRICKS_ALIGNMENT_JOB_ID",
    "gwas": "GWAS_ANALYSIS_JOB_ID",
    "vcf_ingestion": "VCF_INGESTION_JOB_ID",
    "variant_annotation": "VARIANT_ANNOTATION_JOB_ID",
}


_IN_PROGRESS = {"started", "phenotype_prepared"}
_COMPLETE = {
    "alignment_complete",
    "gwas_complete",
    "ingestion_complete",
    "annotation_complete",
}
_FAILED = {"failed"}

_PROGRESS_MAP = {
    "started": "🟩⬜⬜",
    "phenotype_prepared": "🟩🟩⬜",
    "alignment_complete": "🟩🟩🟩",
    "gwas_complete": "🟩🟩🟩",
    "ingestion_complete": "🟩🟩🟩",
    "annotation_complete": "🟩🟩🟩",
    "failed": "🟥",
}


def _progress(status: str) -> str:
    return _PROGRESS_MAP.get(status, "⬜⬜⬜")


def _start_time_ms(v) -> int | None:
    try:
        if pd.isna(v):
            return None
        if hasattr(v, "value"):
            return int(v.value // 1_000_000)
        return int(v)
    except (ValueError, TypeError, AttributeError):
        return None


def _experiment_map() -> dict[str, str]:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    experiments = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    return {e.experiment_id: e.name.split("/")[-1] for e in experiments}


def _search(
    feature: str,
    user_email: str,
    by: str,
    text: str,
    detail_col: str,
) -> list[DBRun]:
    """Generic MLflow-runs search filtered to a Disease Biology feature
    tag for the requesting user. `detail_col` is the workflow-specific
    parameter we surface in the UI table (e.g. `params.fastq_r1`)."""
    exp_map = _experiment_map()
    if not exp_map:
        return []

    if by == "experiment_name":
        needle = text.upper()
        exp_map = {
            eid: name for eid, name in exp_map.items()
            if needle in name.upper()
        }
        if not exp_map:
            return []

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='{feature}' AND "
            f"tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=list(exp_map.keys()),
    )
    if runs.empty:
        return []

    if by == "run_name":
        runs = runs[
            runs["tags.mlflow.runName"].astype(str).str.contains(text, case=False, na=False)
        ]
        if runs.empty:
            return []

    runs["experiment_name"] = runs["experiment_id"].map(exp_map)
    job_id_env = _FEATURE_TO_JOB_ENV.get(feature, "")
    job_id = os.environ.get(job_id_env, "") if job_id_env else ""
    out: list[DBRun] = []
    for _, r in runs.iterrows():
        status = str(r.get("tags.job_status", "") or "")
        detail = str(r.get(detail_col, "") or "")
        job_run_id = str(r.get("tags.job_run_id", "") or "")
        out.append(
            DBRun(
                run_id=str(r.get("run_id", "")),
                run_name=str(r.get("tags.mlflow.runName", "") or ""),
                experiment_name=str(r.get("experiment_name", "") or ""),
                status=status,
                progress=_progress(status),
                start_time_ms=_start_time_ms(r.get("start_time")),
                detail=detail,
                run_url=job_run_url(job_id, job_run_id),
            )
        )
    return out


# ─── Variant Calling (Parabricks alignment) ────────────────────────────────


def start_parabricks_alignment(
    user_info: UserInfo,
    fastq_r1: str,
    fastq_r2: str,
    reference_genome_path: str,
    output_volume_path: str,
    mlflow_experiment_name: str,
    mlflow_run_name: str,
) -> int:
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None,
    )
    with mlflow.start_run(
        run_name=mlflow_run_name, experiment_id=experiment.experiment_id
    ) as run:
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
                "user_email": user_info.user_email,
            },
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "gwas_alignment")
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")
    return int(job_run_id)


def search_variant_calling(user_email: str, by: str, text: str) -> list[DBRun]:
    return _search("gwas_alignment", user_email, by, text, "params.fastq_r1")


def list_successful_variant_calling(user_email: str) -> list[dict]:
    """Picker rows for the GWAS form. Each row exposes the output VCF that
    GWAS will consume next."""
    exp_map = _experiment_map()
    if not exp_map:
        return []
    runs = mlflow.search_runs(
        filter_string=(
            "tags.feature='gwas_alignment' AND "
            "tags.job_status='alignment_complete' AND "
            f"tags.created_by='{user_email}' AND "
            "tags.origin='genesis_workbench'"
        ),
        experiment_ids=list(exp_map.keys()),
    )
    if runs.empty:
        return []
    runs["experiment_name"] = runs["experiment_id"].map(exp_map)
    return [
        {
            "run_id": str(r.get("run_id", "")),
            "run_name": str(r.get("tags.mlflow.runName", "") or ""),
            "experiment_name": str(r.get("experiment_name", "") or ""),
            "output_vcf": str(r.get("params.output_vcf", "") or ""),
            "start_time_ms": _start_time_ms(r.get("start_time")),
        }
        for _, r in runs.iterrows()
    ]


# ─── GWAS ───────────────────────────────────────────────────────────────────


def start_gwas_analysis(
    user_info: UserInfo,
    vcf_path: str,
    phenotype_path: str,
    phenotype_column: str,
    contigs: str,
    hwe_cutoff: str,
    pvalue_threshold: str,
    mlflow_experiment_name: str,
    mlflow_run_name: str,
) -> int:
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None,
    )
    with mlflow.start_run(
        run_name=mlflow_run_name, experiment_id=experiment.experiment_id
    ) as run:
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
                "user_email": user_info.user_email,
            },
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "gwas")
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")
    return int(job_run_id)


def search_gwas(user_email: str, by: str, text: str) -> list[DBRun]:
    return _search("gwas", user_email, by, text, "params.vcf_path")


def pull_gwas_results(run_id: str) -> pd.DataFrame:
    """Top 10k variants by pvalue from the per-run `gwas_results_<run_id>`
    Delta table."""
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    table = f"gwas_results_{run_id.replace('-', '_')}"
    fq = f"{catalog}.{schema}.{table}"
    try:
        execute_select_query(f"DESCRIBE {fq}")
    except Exception as e:
        logger.info("gwas results table %s not found: %s", fq, e)
        return pd.DataFrame()
    query = (
        f"SELECT contigName, start, pvalue, referenceAllele, alternateAlleles, "
        f"effect, phenotype, "
        f"CASE WHEN pvalue IS NOT NULL AND pvalue > 0 THEN -log(10, pvalue) "
        f"ELSE NULL END as neg_log_pval "
        f"FROM {fq} WHERE pvalue IS NOT NULL "
        f"ORDER BY pvalue ASC LIMIT 10000"
    )
    return execute_select_query(query)


# ─── VCF Ingestion ─────────────────────────────────────────────────────────


def start_vcf_ingestion(
    user_info: UserInfo,
    vcf_path: str,
    output_table_name: str,
    mlflow_experiment_name: str,
    mlflow_run_name: str,
) -> int:
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None,
    )
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    with mlflow.start_run(
        run_name=mlflow_run_name, experiment_id=experiment.experiment_id
    ) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("vcf_path", vcf_path)
        mlflow.log_param("output_table_name", output_table_name)
        job_run_id = execute_workflow(
            job_id=os.environ["VCF_INGESTION_JOB_ID"],
            params={
                "catalog": catalog,
                "schema": schema,
                "sql_warehouse_id": os.environ["SQL_WAREHOUSE"],
                "vcf_path": vcf_path,
                "output_table_name": output_table_name,
                "mlflow_run_id": mlflow_run_id,
                "user_email": user_info.user_email,
            },
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "vcf_ingestion")
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")
        mlflow.set_tag("output_table", f"{catalog}.{schema}.{output_table_name}")
    return int(job_run_id)


def search_vcf_ingestion(user_email: str, by: str, text: str) -> list[DBRun]:
    return _search("vcf_ingestion", user_email, by, text, "params.vcf_path")


def list_successful_vcf_ingestion(user_email: str) -> list[dict]:
    """Picker rows for the Variant Annotation form."""
    exp_map = _experiment_map()
    if not exp_map:
        return []
    runs = mlflow.search_runs(
        filter_string=(
            "tags.feature='vcf_ingestion' AND "
            "tags.job_status='ingestion_complete' AND "
            f"tags.created_by='{user_email}' AND "
            "tags.origin='genesis_workbench'"
        ),
        experiment_ids=list(exp_map.keys()),
    )
    if runs.empty:
        return []
    runs["experiment_name"] = runs["experiment_id"].map(exp_map)
    return [
        {
            "run_id": str(r.get("run_id", "")),
            "run_name": str(r.get("tags.mlflow.runName", "") or ""),
            "experiment_name": str(r.get("experiment_name", "") or ""),
            "output_table": str(r.get("tags.output_table", "") or ""),
            "start_time_ms": _start_time_ms(r.get("start_time")),
        }
        for _, r in runs.iterrows()
    ]


# ─── Variant Annotation ────────────────────────────────────────────────────


def start_variant_annotation(
    user_info: UserInfo,
    variants_table: str,
    gene_regions: str,
    pathogenic_vcf_path: str,
    mlflow_experiment_name: str,
    mlflow_run_name: str,
    gene_panel_mode: str = "custom",
) -> int:
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None,
    )
    with mlflow.start_run(
        run_name=mlflow_run_name, experiment_id=experiment.experiment_id
    ) as run:
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
                "user_email": user_info.user_email,
            },
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "variant_annotation")
        mlflow.set_tag("run_name", mlflow_run_name)
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")
    return int(job_run_id)


def search_variant_annotation(user_email: str, by: str, text: str) -> list[DBRun]:
    return _search("variant_annotation", user_email, by, text, "params.variants_table")


def pull_annotation_results(run_id: str, run_name: str = "") -> pd.DataFrame:
    """Pathogenic variant rows for the View dialog. Each variant_annotation
    run has its own table named via the `pathogenic_table` MLflow tag/param;
    fall back to reconstructing the name if older runs lack the tag."""
    import re
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]

    run = mlflow.get_run(run_id)
    if not run_name:
        run_name = run.data.tags.get("run_name", "")
    table = run.data.tags.get("pathogenic_table") or run.data.params.get("pathogenic_table")
    if not table:
        safe = re.sub(r"[^a-z0-9_]", "_", (run_name or "").lower())
        safe = re.sub(r"_+", "_", safe).strip("_")[:40] or "unnamed"
        suffix = f"{safe}_{(run_id or 'norun')[:8]}"
        table = f"{catalog}.{schema}.variant_annotation_pathogenic__{suffix}"

    try:
        cols_df = execute_select_query(f"DESCRIBE {table}")
        available_cols = cols_df["col_name"].tolist() if "col_name" in cols_df.columns else []
    except Exception:
        available_cols = []

    extra_cols = "category, condition, " if "category" in available_cols else ""
    query = (
        f"SELECT gene, {extra_cols}chromosome, start as position, ref, alt, zygosity, "
        f"array_join(clinical_significance, ', ') as clinical_significance, "
        f"array_join(disease_name, ', ') as disease_name "
        f"FROM {table} ORDER BY gene, position"
    )
    return execute_select_query(query)


# ─── Run details (for View dialogs) ────────────────────────────────────────


def get_run_details(run_id: str) -> dict[str, Any]:
    """MLflow params + tags for any disease-biology run. Used by the View
    dialogs to show input paths + status."""
    # Same set-then-construct dance the other DB helpers do — MlflowClient()
    # silently uses whatever tracking URI is active in the process; without
    # explicitly steering to databricks-uc it sometimes ends up pointing at
    # the local file:// store and `get_run` 404s.
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()
    run = client.get_run(run_id)
    return {
        "run_name": run.data.tags.get("mlflow.runName", ""),
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "job_status": run.data.tags.get("job_status", ""),
        "job_run_id": run.data.tags.get("job_run_id", ""),
        "params": dict(run.data.params),
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
    }

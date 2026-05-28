"""Single-cell MLflow run discovery + artifact download + job dispatch.
Ported from modules/core/app/utils/single_cell_analysis.py."""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from mlflow.tracking import MlflowClient

from genesis_workbench.models import set_mlflow_experiment
from genesis_workbench.workbench import UserInfo

logger = logging.getLogger(__name__)

SINGLECELL_MODES = ("scanpy", "rapids-singlecell")

# 3-step progress chip — mirrors `_SC_PROGRESS_MAP` from the Streamlit utils.
PROGRESS_MAP = {
    "started": "🟩⬜⬜",
    "processing": "🟩🟩⬜",
    "complete": "🟩🟩🟩",
    "finished": "🟩🟩🟩",
    "failed": "🟥",
    "unknown": "⬜⬜⬜",
}


def progress_chip(status: str) -> str:
    return PROGRESS_MAP.get(status.lower(), PROGRESS_MAP["unknown"])


@dataclass(frozen=True)
class SingleCellRun:
    run_id: str
    run_name: str
    experiment_name: str
    processing_mode: str
    start_time_ms: int | None
    status: str
    cells: int | None


def search_runs(user_email: str, processing_mode: str | None = None) -> list[SingleCellRun]:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if not experiment_list:
        return []
    experiments = {exp.experiment_id: exp.name for exp in experiment_list}

    filter_parts = [
        f"tags.created_by='{user_email}'",
        "tags.origin='genesis_workbench'",
    ]
    if processing_mode:
        filter_parts.append(f"tags.processing_mode='{processing_mode}'")
    else:
        quoted = ",".join(f"'{m}'" for m in SINGLECELL_MODES)
        filter_parts.append(f"tags.processing_mode IN ({quoted})")

    try:
        runs = mlflow.search_runs(
            experiment_ids=list(experiments.keys()),
            filter_string=" AND ".join(filter_parts),
            order_by=["start_time DESC"],
            max_results=100,
        )
    except Exception as e:
        logger.warning("search_runs filtered failed: %s", e)
        return []

    if runs.empty:
        return []

    out: list[SingleCellRun] = []
    for _, r in runs.iterrows():
        exp_id = r.get("experiment_id")
        exp_full = experiments.get(exp_id, "")
        exp_short = exp_full.split("/")[-1] if exp_full else ""
        start_ts = r.get("start_time")
        start_ms = int(start_ts.value // 1_000_000) if pd.notna(start_ts) else None
        try:
            cells = int(r.get("metrics.n_cells")) if pd.notna(r.get("metrics.n_cells")) else None
        except (TypeError, ValueError):
            cells = None
        out.append(
            SingleCellRun(
                run_id=str(r["run_id"]),
                run_name=str(r.get("tags.mlflow.runName", "")),
                experiment_name=exp_short,
                processing_mode=str(r.get("tags.processing_mode", "")),
                start_time_ms=start_ms,
                status=str(r.get("tags.job_status", "unknown")),
                cells=cells,
            )
        )
    return out


def download_markers_df(run_id: str) -> pd.DataFrame:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = client.download_artifacts(run_id, "markers_flat.parquet", dst_path=tmpdir)
        return pd.read_parquet(local_file)


def download_cluster_markers_mapping(run_id: str) -> pd.DataFrame:
    """Per-cluster Wilcoxon rank-sum markers from `top_markers_per_cluster.csv`.

    Columns are cluster IDs (as strings), rows are gene rankings (most
    discriminative first). Ported from the Streamlit utils — enrichment uses
    these instead of top-by-mean-expression so each cluster gets its own
    distinguishing gene set rather than the same housekeeping list."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = client.download_artifacts(
            run_id, "top_markers_per_cluster.csv", dst_path=tmpdir
        )
        return pd.read_csv(local_file)


# ─── annotation persistence ────────────────────────────────────────────────
#
# Streamlit used MlflowClient.log_artifact directly (not mlflow.start_run +
# log_artifact) because the Databricks Apps runtime sets MLFLOW_RUN_ID at
# process level, and start_run(run_id=...) inside that env raises a "Cannot
# start run … active run ID does not match" error. The client API ignores
# the env var and takes run_id as an explicit argument — same trick here.
# Storage format matches Streamlit's so existing saved annotations from the
# Streamlit app stay loadable on the same runs:
#   { "cluster_col": str, "model": "scimilarity"|"teddy", "results": <records> }

_ANNOTATION_ARTIFACTS = {
    "scimilarity": "scimilarity_annotation.json",
    "teddy": "teddy_annotation.json",
}


def save_annotation(
    run_id: str,
    model: str,
    cluster_col: str,
    results: list[dict],
) -> None:
    """Persist a cluster-annotation result list as a per-model JSON artifact
    on the source MLflow run. `results` is the per-cluster list of dicts
    (already JSON-serializable). Overwrites any existing artifact of the
    same name on the run."""
    import json

    artifact_name = _ANNOTATION_ARTIFACTS.get(model)
    if not artifact_name:
        raise ValueError(
            f"Unknown annotation model '{model}'. Expected one of: {list(_ANNOTATION_ARTIFACTS)}"
        )

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()
    payload = {"cluster_col": cluster_col, "model": model, "results": results}
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, artifact_name)
        with open(local, "w") as f:
            json.dump(payload, f, default=str)
        client.log_artifact(run_id=run_id, local_path=local)
    logger.info("Saved %s annotation (%d clusters) to MLflow run %s", model, len(results), run_id)


def load_annotation(run_id: str, model: str) -> dict | None:
    """Return the previously-saved annotation payload for ``model`` on this
    run, or None if absent. Treats "artifact missing" as a clean miss; only
    real failures (auth, server error, malformed JSON) raise."""
    import json

    from mlflow.exceptions import MlflowException

    artifact_name = _ANNOTATION_ARTIFACTS.get(model)
    if not artifact_name:
        raise ValueError(
            f"Unknown annotation model '{model}'. Expected one of: {list(_ANNOTATION_ARTIFACTS)}"
        )

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local = client.download_artifacts(run_id, artifact_name, dst_path=tmpdir)
            with open(local, "r") as f:
                return json.load(f)
    except (FileNotFoundError, MlflowException) as e:
        msg = str(e).lower()
        if isinstance(e, FileNotFoundError) or any(
            k in msg for k in ("does not exist", "not found", "resource_does_not_exist")
        ):
            return None
        raise


@dataclass(frozen=True)
class JobDispatchResult:
    job_id: int
    job_run_id: int
    mlflow_run_id: str
    experiment_id: str


def start_job(
    *,
    mode: str,
    user_info: UserInfo,
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
    compute_pseudotime: bool = False,
) -> JobDispatchResult:
    """Dispatch a scanpy or rapids-singlecell processing run.

    Pre-creates the MLflow run (status=started) so the Search Past Runs UI
    picks it up immediately, then kicks the job with mlflow_run_id passed
    through so the notebook attaches to the same run."""
    if mode == "scanpy":
        job_env_var = "RUN_SCANPY_JOB_ID"
    elif mode == "rapids-singlecell":
        job_env_var = "RUN_RAPIDSSINGLECELL_JOB_ID"
    else:
        raise ValueError(f"Unknown processing mode: {mode!r}")

    job_id = os.environ.get(job_env_var)
    if not job_id:
        raise RuntimeError(
            f"{mode} job not registered (env {job_env_var} unset). "
            "Deploy the corresponding submodule first."
        )

    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment, user_email=user_info.user_email
    )

    w = WorkspaceClient()
    with mlflow.start_run(
        run_name=mlflow_run_name, experiment_id=experiment.experiment_id
    ) as run:
        mlflow_run_id = run.info.run_id
        params = {
            "data_path": data_path,
            "mlflow_experiment": experiment.name,
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
            "compute_pseudotime": str(compute_pseudotime).lower(),
        }
        mlflow.log_params(params)

        job_run = w.jobs.run_now(
            job_id=int(job_id),
            job_parameters={
                "catalog": os.environ["CORE_CATALOG_NAME"],
                "schema": os.environ["CORE_SCHEMA_NAME"],
                "user_email": user_info.user_email,
                "mlflow_run_id": mlflow_run_id,
                **params,
            },
        )

        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", mode)
        mlflow.set_tag("processing_mode", mode)
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", str(job_run.run_id))
        mlflow.set_tag("job_status", "started")

    return JobDispatchResult(
        job_id=int(job_id),
        job_run_id=int(job_run.run_id),
        mlflow_run_id=mlflow_run_id,
        experiment_id=str(experiment.experiment_id),
    )

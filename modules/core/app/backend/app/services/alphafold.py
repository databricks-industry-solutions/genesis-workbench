"""AlphaFold2 async-job wrappers ported from
modules/core/app/utils/protein_structure.py. The reference-PDB alignment
path (include_pdb=True) is intentionally not ported — it requires biopython
and the React UI doesn't expose that toggle yet."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient

from genesis_workbench.models import set_mlflow_experiment
from genesis_workbench.workbench import UserInfo, execute_workflow

from app.services.workbench import get_job_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlphaFoldRun:
    run_id: str
    run_name: str
    experiment_name: str
    protein_sequence: str
    start_time_ms: int | None
    status: str
    # Workspace UI link to the dispatched AlphaFold job's run page.
    run_url: str = ""


def start_run_alphafold_job(
    protein_sequence: str,
    mlflow_experiment_name: str,
    mlflow_run_name: str,
    user_info: UserInfo,
) -> str:
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
        mlflow.log_param("protein_sequence", protein_sequence)

        job_run_id = execute_workflow(
            job_id=int(get_job_id("run_alphafold_job_id")),
            params={
                "catalog": os.environ["CORE_CATALOG_NAME"],
                "schema": os.environ["CORE_SCHEMA_NAME"],
                "run_id": mlflow_run_id,
                "protein_sequence": protein_sequence,
                "user_email": user_info.user_email,
            },
        )
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "alphafold")
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_run_id", job_run_id)
        mlflow.set_tag("job_status", "started")

    return str(job_run_id)


def _search_runs(user_email: str) -> tuple[dict[str, str], pd.DataFrame]:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if not experiment_list:
        return {}, pd.DataFrame()
    experiments = {exp.experiment_id: exp.name.split("/")[-1] for exp in experiment_list}
    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='alphafold' AND tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=list(experiments.keys()),
    )
    return experiments, runs


def _df_to_runs(experiments: dict[str, str], df: pd.DataFrame) -> list[AlphaFoldRun]:
    if df.empty:
        return []
    from app.services.databricks_links import job_run_url

    df = df.copy()
    df["experiment_name"] = df["experiment_id"].map(experiments)
    job_id = get_job_id("run_alphafold_job_id")
    out: list[AlphaFoldRun] = []
    for _, r in df.iterrows():
        job_run_id = str(r.get("tags.job_run_id", "") or "")
        out.append(
            AlphaFoldRun(
                run_id=str(r["run_id"]),
                run_name=str(r.get("tags.mlflow.runName", "")),
                experiment_name=str(r.get("experiment_name", "")),
                protein_sequence=str(r.get("params.protein_sequence", "")),
                start_time_ms=int(r["start_time"].value // 1_000_000)
                if pd.notna(r.get("start_time"))
                else None,
                status=str(r.get("tags.job_status", "unknown")),
                run_url=job_run_url(job_id, job_run_id),
            )
        )
    return out


def search_by_run_name(user_email: str, run_name_filter: str) -> list[AlphaFoldRun]:
    experiments, runs = _search_runs(user_email)
    if runs.empty:
        return []
    matched = runs[
        runs["tags.mlflow.runName"].str.contains(run_name_filter, case=False, na=False)
    ]
    return _df_to_runs(experiments, matched)


def search_by_experiment_name(
    user_email: str, experiment_name_filter: str
) -> list[AlphaFoldRun]:
    experiments, runs = _search_runs(user_email)
    if runs.empty:
        return []
    needle = experiment_name_filter.upper()
    matching_ids = {eid for eid, name in experiments.items() if needle in name.upper()}
    matched = runs[runs["experiment_id"].isin(matching_ids)]
    return _df_to_runs(experiments, matched)


def pull_alphafold_pdb(run_id: str) -> str:
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    logger.info("Fetching AlphaFold PDB for run %s", run_id)
    w = WorkspaceClient()
    response = w.files.download(
        f"/Volumes/{catalog}/{schema}/alphafold/results/{run_id}/{run_id}/ranked_0.pdb"
    )
    return str(response.contents.read(), encoding="utf-8")


def apply_pdb_header(pdb_str: str, name: str) -> str:
    return (
        f'HEADER    "{name}"                           00-JAN-00   0XXX\n'
        f'TITLE     "{name}"\n'
        f"COMPND    MOL_ID: 1;\n"
        f"COMPND   2 MOLECULE: {name};\n"
        f"COMPND   3 CHAIN: A;\n"
        + pdb_str
    )

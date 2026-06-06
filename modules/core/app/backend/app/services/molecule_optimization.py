"""Guided Molecule Optimization — dispatch + read helpers.

Small-molecule twin of enzyme_optimization.py. Dispatches the
`run_molecule_optimization_gwb` orchestrator job (GenMol→score→reseed loop),
pre-creating the MLflow run so Search Past Runs shows it in-flight; the
orchestrator logs the trajectory + top_k artifact + job_status to that run.
Status/search read MLflow by the `feature='molecule_optimization'` tag.
"""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import uuid
from typing import Any, Optional

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from genesis_workbench.models import set_mlflow_experiment
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

ORCHESTRATOR_JOB_NAME = "run_molecule_optimization_gwb"
TARGET_VOLUME_DIR = "genmol"  # the genmol module's UC cache volume
_job_id_cache: dict[str, int] = {}


def _write_pdb_to_volume(pdb_str: str, catalog: str, schema: str) -> str:
    """Upload the target PDB to the genmol UC volume; return its path. The
    Apps sandbox blocks open('/Volumes/...'), so go through the Files API
    (the orchestrator cluster reads it back with a plain open())."""
    run_uuid = uuid.uuid4().hex[:12]
    path = f"/Volumes/{catalog}/{schema}/{TARGET_VOLUME_DIR}/mol_opt_targets/{run_uuid}/target.pdb"
    WorkspaceClient().files.upload(
        file_path=path, contents=io.BytesIO(pdb_str.encode("utf-8")), overwrite=True
    )
    return path


def _resolve_orchestrator_job_id(w: Optional[WorkspaceClient] = None) -> int:
    cached = _job_id_cache.get(ORCHESTRATOR_JOB_NAME)
    if cached is not None:
        return cached
    env_id = os.environ.get("RUN_MOLECULE_OPTIMIZATION_JOB_ID")
    if env_id:
        _job_id_cache[ORCHESTRATOR_JOB_NAME] = int(env_id)
        return int(env_id)
    workspace = w or WorkspaceClient()
    matches = list(workspace.jobs.list(name=ORCHESTRATOR_JOB_NAME))
    if not matches:
        raise RuntimeError(
            f"Orchestrator job '{ORCHESTRATOR_JOB_NAME}' not found. Deploy the genmol "
            "submodule: `./deploy.sh small_molecule aws --only-submodule genmol/genmol_v1`"
        )
    _job_id_cache[ORCHESTRATOR_JOB_NAME] = int(matches[0].job_id)
    return _job_id_cache[ORCHESTRATOR_JOB_NAME]


def start_molecule_optimization_job(
    *,
    user_email: str,
    mlflow_experiment: str,
    mlflow_run_name: str,
    seed_smiles: list[str],
    num_samples: int,
    num_iterations: int,
    select_top: int,
    dock_top_k: int,
    weights: dict[str, float],
    temperature: float,
    randomness: float,
    target_pdb: str = "",
    dock_per_iter: int = 8,
    dock_samples: int = 3,
) -> dict:
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment, user_email=user_email, host=None, token=None
    )
    w = WorkspaceClient()
    job_id = _resolve_orchestrator_job_id(w)
    # If a target structure is supplied, stage it on the volume so the loop can
    # dock candidates in-reward (DiffDock). Empty path ⇒ QED+ADMET-only loop.
    target_pdb_path = ""
    if target_pdb and target_pdb.strip():
        target_pdb_path = _write_pdb_to_volume(
            target_pdb, os.environ["CORE_CATALOG_NAME"], os.environ["CORE_SCHEMA_NAME"]
        )

    with mlflow.start_run(
        run_name=mlflow_run_name, experiment_id=experiment.experiment_id
    ) as pre:
        run_id = pre.info.run_id
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "molecule_optimization")
        mlflow.set_tag("created_by", user_email)
        mlflow.set_tag("job_status", "submitted")
        mlflow.log_param("num_samples", num_samples)
        mlflow.log_param("num_iterations", num_iterations)
        try:
            job_run = w.jobs.run_now(
                job_id=job_id,
                job_parameters={
                    "catalog": os.environ["CORE_CATALOG_NAME"],
                    "schema": os.environ["CORE_SCHEMA_NAME"],
                    "sql_warehouse_id": os.environ.get("SQL_WAREHOUSE", ""),
                    "user_email": user_email,
                    "mlflow_experiment": mlflow_experiment,
                    "mlflow_run_name": mlflow_run_name,
                    "mlflow_run_id": run_id,
                    "seed_smiles_csv": ",".join(seed_smiles),
                    "num_samples": str(num_samples),
                    "num_iterations": str(num_iterations),
                    "select_top": str(select_top),
                    "dock_top_k": str(dock_top_k),
                    "weights_json": json.dumps({"qed": 1.0, "admet": 1.0, **(weights or {})}),
                    "temperature": str(temperature),
                    "randomness": str(randomness),
                    "target_pdb_path": target_pdb_path or "",
                    "dock_per_iter": str(dock_per_iter),
                    "dock_samples": str(dock_samples),
                },
            )
        except Exception as e:
            mlflow.set_tag("job_status", "failed")
            mlflow.set_tag("error", str(e)[:500])
            raise
        mlflow.set_tag("job_run_id", str(job_run.run_id))

    return {
        "job_id": job_id,
        "job_run_id": int(job_run.run_id),
        "mlflow_run_id": run_id,
        "experiment_id": str(experiment.experiment_id),
    }


def get_run_status(run_id: str) -> dict[str, Any]:
    client = MlflowClient()
    run = client.get_run(run_id)

    def hist(metric: str):
        try:
            return [
                {"step": m.step, "value": float(m.value)}
                for m in client.get_metric_history(run_id, metric)
            ]
        except Exception:
            return []

    return {
        "status": run.info.status,
        "job_status": run.data.tags.get("job_status", ""),
        "best_reward_history": hist("iter_best_reward"),
        "mean_reward_history": hist("iter_mean_reward"),
        "best_qed_history": hist("iter_best_qed"),
        "current_metrics": {k: float(v) for k, v in run.data.metrics.items()},
        "experiment_id": run.info.experiment_id,
        "run_name": run.data.tags.get("mlflow.runName", ""),
    }


def load_top_k(run_id: str) -> list[dict]:
    """The `top_k.json` artifact → list of {smiles, qed, tox, reward, dock_confidence?}."""
    client = MlflowClient()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run_id, "top_k.json", dst_path=tmp)
            with open(local) as f:
                data = json.load(f)
        return list(data.get("top_k", data) if isinstance(data, dict) else data)
    except Exception as e:
        logger.info("top_k not yet available for run %s: %s", run_id, e)
        return []


def _experiment_map() -> dict[str, str]:
    experiments = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench = 'yes'"
    )
    return {e.experiment_id: e.name for e in experiments}


def search_runs(user_email: str, by: str, text: str) -> list[dict]:
    exp_map = _experiment_map()
    if not exp_map:
        return []
    if by == "experiment_name":
        needle = text.upper()
        exp_map = {eid: n for eid, n in exp_map.items() if needle in n.upper()}
        if not exp_map:
            return []
    runs = mlflow.search_runs(
        filter_string=(
            "tags.feature='molecule_optimization' AND "
            f"tags.created_by='{user_email}' AND tags.origin='genesis_workbench'"
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

    def _g(r, col):
        return r[col] if col in r and pd.notna(r[col]) else None

    out = []
    for _, r in runs.iterrows():
        out.append({
            "run_id": str(r["run_id"]),
            "run_name": str(_g(r, "tags.mlflow.runName") or ""),
            "experiment_name": exp_map.get(str(r["experiment_id"]), ""),
            "job_status": str(_g(r, "tags.job_status") or ""),
            "num_iterations": (int(r["params.num_iterations"]) if "params.num_iterations" in r and pd.notna(r["params.num_iterations"]) else None),
            "iterations_completed": (int(r["metrics.iterations_completed"]) if "metrics.iterations_completed" in r and pd.notna(r["metrics.iterations_completed"]) else None),
            "start_time_ms": (int(r["start_time"].timestamp() * 1000) if "start_time" in r and pd.notna(r["start_time"]) else None),
        })
    return out

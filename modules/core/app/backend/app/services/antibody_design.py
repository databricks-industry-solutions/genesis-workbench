"""Antibody Design (VHH) — Databricks Job dispatcher + MLflow result loaders.

Mirrors services/enzyme_optimization.py: launching kicks off the
`run_antibody_design_gwb` orchestrator (RFD4-Proteina in-process on an H100 →
reward-weighted VHH loop) which runs for ~1-4 h. The app dispatches + searches;
results are reviewed via Search Past Runs + the result dialog.
"""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from genesis_workbench.models import set_mlflow_experiment
from genesis_workbench.workbench import UserInfo
from mlflow.tracking import MlflowClient

from app.services.databricks_links import mlflow_run_url

logger = logging.getLogger(__name__)

ORCHESTRATOR_JOB_NAME = "run_antibody_design_gwb"
ORCHESTRATOR_VOLUME_DIR_NAME = "antibody_design"

# Binding (boltz) + low-immunogenicity are weighted up — what matters most for a
# developable antibody. immuno is lower-is-better (handled in the orchestrator).
DEFAULT_AXIS_WEIGHTS: dict[str, float] = {
    "plddt":      1.3,
    "boltz":      2.0,
    "solubility": 1.0,
    "half_life":  1.0,
    "thermostab": 1.0,
    "immuno":     1.5,   # MHC-I (CD8) burden — MHCflurry
    "immuno_mhc2": 1.5,  # MHC-II (CD4 / anti-drug-antibody) burden — HLAIIPred (right signal for VHH)
    "liability":  1.0,   # rule-based sequence-liability scan (lower raw count = more inert)
}

_orchestrator_job_id_cache: dict[str, int] = {}


def _use_databricks_tracking() -> None:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")


def _resolve_orchestrator_job_id(w: Optional[WorkspaceClient] = None) -> int:
    cached = _orchestrator_job_id_cache.get(ORCHESTRATOR_JOB_NAME)
    if cached is not None:
        return cached
    env_id = os.environ.get("RUN_ANTIBODY_DESIGN_JOB_ID")
    if env_id:
        _orchestrator_job_id_cache[ORCHESTRATOR_JOB_NAME] = int(env_id)
        return int(env_id)
    workspace = w or WorkspaceClient()
    matches = list(workspace.jobs.list(name=ORCHESTRATOR_JOB_NAME))
    if not matches:
        raise RuntimeError(
            f"Orchestrator job '{ORCHESTRATOR_JOB_NAME}' not found. Deploy the "
            "antibody_design submodule first: `./deploy.sh large_molecule aws "
            "--only-submodule antibody_design/antibody_design_v1`"
        )
    _orchestrator_job_id_cache[ORCHESTRATOR_JOB_NAME] = int(matches[0].job_id)
    return _orchestrator_job_id_cache[ORCHESTRATOR_JOB_NAME]


def _write_antigen_pdb_to_volume(antigen_pdb_str: str, catalog: str, schema: str) -> str:
    """Upload the antigen PDB to a per-run UC volume dir via the Files API (the
    Apps sandbox blocks direct open('/Volumes/...'))."""
    run_uuid = uuid.uuid4().hex[:12]
    volume_dir = f"/Volumes/{catalog}/{schema}/{ORCHESTRATOR_VOLUME_DIR_NAME}/{run_uuid}"
    antigen_path = f"{volume_dir}/antigen.pdb"
    WorkspaceClient().files.upload(
        file_path=antigen_path,
        contents=io.BytesIO(antigen_pdb_str.encode("utf-8")),
        overwrite=True,
    )
    return antigen_path


@dataclass(frozen=True)
class JobDispatchResult:
    job_id: int
    job_run_id: int
    mlflow_run_id: str
    experiment_id: str


def start_antibody_design_job(
    *,
    antigen_pdb_str: str,
    epitope_residues: list[int],
    antigen_chain: str,
    vhh_length_min: int,
    vhh_length_max: int,
    num_samples: int,
    num_iterations: int,
    weights: dict[str, float],
    user_info: UserInfo,
    mlflow_experiment: str,
    mlflow_run_name: str,
    references: Optional[list[dict]] = None,
    half_life_margin: float = 0.05,
    resampling_temperature: float = 0.1,
    strategy: str = "resample",
    run_proteinmpnn: bool = True,
    cofold_antigen: bool = True,
    convergence_threshold: Optional[float] = 0.01,
    convergence_window: int = 2,
    target_reward: Optional[float] = None,
    best_k_target: Optional[int] = None,
    best_k_threshold: Optional[float] = None,
) -> JobDispatchResult:
    _use_databricks_tracking()
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    antigen_pdb_path = _write_antigen_pdb_to_volume(antigen_pdb_str, catalog, schema)

    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment, user_email=user_info.user_email, host=None, token=None
    )
    w = WorkspaceClient()
    job_id = _resolve_orchestrator_job_id(w=w)

    with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as pre_run:
        mlflow_run_id = pre_run.info.run_id
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "antibody_design")
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_status", "submitted")
        mlflow.log_param("format", "VHH")
        mlflow.log_param("vhh_length_min", vhh_length_min)
        mlflow.log_param("vhh_length_max", vhh_length_max)
        mlflow.log_param("num_samples", num_samples)
        mlflow.log_param("num_iterations", num_iterations)

        try:
            job_run = w.jobs.run_now(
                job_id=job_id,
                job_parameters={
                    "catalog": catalog,
                    "schema": schema,
                    "cache_dir": ORCHESTRATOR_VOLUME_DIR_NAME,
                    "sql_warehouse_id": os.environ.get("SQL_WAREHOUSE", ""),
                    "user_email": user_info.user_email,
                    "mlflow_experiment": mlflow_experiment,
                    "mlflow_run_name": mlflow_run_name,
                    "mlflow_run_id": mlflow_run_id,
                    "antigen_pdb_path": antigen_pdb_path,
                    "epitope_residues_csv": ",".join(str(r) for r in epitope_residues),
                    "antigen_chain": antigen_chain,
                    "vhh_length_min": str(vhh_length_min),
                    "vhh_length_max": str(vhh_length_max),
                    "num_samples": str(num_samples),
                    "num_iterations": str(num_iterations),
                    "run_proteinmpnn": str(run_proteinmpnn).lower(),
                    "resampling_temperature": str(resampling_temperature),
                    "strategy": strategy,
                    "cofold_antigen": str(cofold_antigen).lower(),
                    "references_json": json.dumps(references or []),
                    "half_life_margin": str(half_life_margin),
                    "weights_json": json.dumps({**DEFAULT_AXIS_WEIGHTS, **(weights or {})}),
                    "dev_user_prefix": os.environ.get("DEV_USER_PREFIX", "") or "",
                    "convergence_threshold": str(convergence_threshold)
                    if convergence_threshold is not None else "0.01",
                    "convergence_window": str(int(convergence_window)),
                    "target_reward": str(target_reward) if target_reward is not None else "",
                    "best_k_target": str(int(best_k_target)) if best_k_target is not None else "",
                    "best_k_threshold": str(best_k_threshold) if best_k_threshold is not None else "",
                },
            )
        except Exception as e:
            mlflow.set_tag("job_status", "failed")
            mlflow.set_tag("error", str(e)[:500])
            raise
        mlflow.set_tag("job_run_id", str(job_run.run_id))

    return JobDispatchResult(
        job_id=job_id,
        job_run_id=int(job_run.run_id),
        mlflow_run_id=mlflow_run_id,
        experiment_id=str(experiment.experiment_id),
    )


# ─── Result loaders ──────────────────────────────────────────────────────────


def get_run_status(run_id: str) -> dict[str, Any]:
    _use_databricks_tracking()
    client = MlflowClient()
    run = client.get_run(run_id)
    iter_max = client.get_metric_history(run_id, "iter_max_reward")
    iter_mean = client.get_metric_history(run_id, "iter_mean_reward")
    return {
        "status": run.info.status,
        "job_status": run.data.tags.get("job_status", ""),
        "iter_max_reward_history": [{"step": m.step, "value": float(m.value)} for m in iter_max],
        "iter_mean_reward_history": [{"step": m.step, "value": float(m.value)} for m in iter_mean],
        "current_metrics": {k: float(v) for k, v in run.data.metrics.items()},
        "experiment_id": run.info.experiment_id,
        "run_name": run.data.tags.get("mlflow.runName", ""),
    }


def load_trajectory(run_id: str) -> pd.DataFrame:
    client = MlflowClient()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run_id, "results/reward_trajectory.csv", dst_path=tmp)
            return pd.read_csv(local)
    except Exception as e:
        logger.info("trajectory not yet available for run %s: %s", run_id, e)
        return pd.DataFrame()


def load_top_k_pdbs(run_id: str) -> dict[str, str]:
    client = MlflowClient()
    out: dict[str, str] = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = client.download_artifacts(run_id, "results/topK_pdbs", dst_path=tmp)
            for fname in sorted(os.listdir(local_dir)):
                if fname.endswith(".pdb"):
                    with open(os.path.join(local_dir, fname)) as f:
                        out[fname[:-4]] = f.read()
    except Exception as e:
        logger.info("topK PDBs not yet available for run %s: %s", run_id, e)
    return out


# ─── Search past runs (returns the shared DBRunRow contract so the frontend's
#     RunSearchSection consumes it directly — same shape as the KERMT search) ───


_PROGRESS_MAP = {
    "submitted": "🟩⬜⬜⬜",
    "started": "🟩🟩⬜⬜",
    "complete": "🟩🟩🟩🟩",
    "failed": "🟥",
    "unknown": "⬜⬜⬜⬜",
}


def _progress(status: str) -> str:
    if not status:
        return _PROGRESS_MAP["unknown"]
    if status in _PROGRESS_MAP:
        return _PROGRESS_MAP[status]
    if status.startswith("iter_"):
        return "🟩🟩🟩⬜"
    return _PROGRESS_MAP["unknown"]


def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None
    except (ValueError, TypeError):
        return None


def _experiment_map() -> dict[str, str]:
    _use_databricks_tracking()
    experiments = mlflow.search_experiments(filter_string="tags.used_by_genesis_workbench='yes'")
    return {e.experiment_id: e.name.split("/")[-1] for e in experiments}


def search_runs(user_email: str, by: str, text: str) -> list[dict]:
    """Return DBRunRow dicts (run_id, run_name, experiment_name, status, progress,
    detail, start_time_ms, run_url). `run_url` is the MLflow run page (metrics/
    artifacts), per the batch-workflow pattern — NOT the job-run page."""
    exp_map = _experiment_map()
    if not exp_map:
        return []
    if by == "experiment_name":
        needle = text.upper()
        exp_map = {eid: name for eid, name in exp_map.items() if needle in name.upper()}
        if not exp_map:
            return []
    runs = mlflow.search_runs(
        filter_string=(
            "tags.feature='antibody_design' AND "
            f"tags.created_by='{user_email}' AND tags.origin='genesis_workbench'"
        ),
        experiment_ids=list(exp_map.keys()),
    )
    if runs.empty:
        return []
    if by == "run_name":
        runs = runs[runs["tags.mlflow.runName"].astype(str).str.contains(text, case=False, na=False)]
        if runs.empty:
            return []

    def _g(r, col):
        return r[col] if col in r and pd.notna(r[col]) else None

    out: list[dict] = []
    for _, r in runs.iterrows():
        status = str(_g(r, "tags.job_status") or "")
        lifecycle = str(_g(r, "status") or "")
        if lifecycle in ("FAILED", "KILLED") and status not in ("complete", "failed"):
            status = "failed"
        exp_id = str(r["experiment_id"])
        iter_max = _safe_float(_g(r, "metrics.iter_max_reward"))
        detail = f"max reward {iter_max:.3f}" if iter_max is not None else "VHH"
        out.append({
            "run_id": str(r["run_id"]),
            "run_name": str(_g(r, "tags.mlflow.runName") or ""),
            "experiment_name": exp_map.get(exp_id, ""),
            "status": status,
            "progress": _progress(status),
            "detail": detail,
            "start_time_ms": (int(r["start_time"].timestamp() * 1000)
                              if "start_time" in r and pd.notna(r["start_time"]) else None),
            "run_url": mlflow_run_url(exp_id, str(r["run_id"])),
        })
    return out

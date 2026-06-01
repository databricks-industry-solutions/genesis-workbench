"""Guided Enzyme Optimization — Databricks Job dispatcher + MLflow result
loaders. Ported from modules/core/app/utils/enzyme_optimization_tools.py.

This workflow is NOT in-process: launching kicks off an orchestrator job
that runs for ~30 min (Fast) to ~6 h (Accurate). The app dispatches and
polls — same shape as the AlphaFold2 panel."""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from genesis_workbench.models import set_mlflow_experiment
from genesis_workbench.workbench import UserInfo
from mlflow.tracking import MlflowClient

from app.services.databricks_links import job_run_url
from app.services.endpoints import get_endpoint_name

logger = logging.getLogger(__name__)

ORCHESTRATOR_JOB_NAME = "run_enzyme_optimization_gwb"
ORCHESTRATOR_JOB_NAME_ACCURATE = "run_enzyme_optimization_gwb_inprocess_ame"
ORCHESTRATOR_VOLUME_DIR_NAME = "enzyme_optimization"

DEFAULT_AXIS_WEIGHTS: dict[str, float] = {
    "motif_rmsd": 1.0,
    "plddt":      1.3,
    "boltz":      0.5,
    "solubility": 1.0,
    "half_life":  2.6,
    "thermostab": 1.0,
    "immuno":     1.5,
}

# T4 lysozyme — canonical stable enzyme used as the smoke-test sequence.
T4_LYSOZYME_SEQUENCE = (
    "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYK"
)
# Mat-1 -> N-end-rule destabilised variant (~30 min half-life). Used as
# the default destabilised reference in the form.
T4_LYSOZYME_NEND_DESTABILIZED = (
    T4_LYSOZYME_SEQUENCE[0] + "R" + T4_LYSOZYME_SEQUENCE[2:]
)

# Long-timeout client for predictor endpoints (GPU cold-starts on first call).
_PREDICTOR_TIMEOUT_SECONDS = 600
_predictor_client = WorkspaceClient(
    config=Config(http_timeout_seconds=_PREDICTOR_TIMEOUT_SECONDS)
)

_orchestrator_job_id_cache: dict[str, int] = {}

_DEFAULT_MHC_ALLELES = (
    "HLA-A*02:01,HLA-A*01:01,HLA-B*07:02,HLA-B*44:02,HLA-C*07:01,HLA-C*04:01"
)


# ─── Job dispatch ──────────────────────────────────────────────────────────


def _resolve_orchestrator_job_id(
    use_inprocess_ame: bool = False, w: Optional[WorkspaceClient] = None
) -> int:
    """Look up the orchestrator job by name; cache after first call."""
    job_name = (
        ORCHESTRATOR_JOB_NAME_ACCURATE if use_inprocess_ame else ORCHESTRATOR_JOB_NAME
    )
    cached = _orchestrator_job_id_cache.get(job_name)
    if cached is not None:
        return cached
    env_var = (
        "RUN_ENZYME_OPTIMIZATION_INPROCESS_AME_JOB_ID"
        if use_inprocess_ame
        else "RUN_ENZYME_OPTIMIZATION_JOB_ID"
    )
    env_id = os.environ.get(env_var)
    if env_id:
        _orchestrator_job_id_cache[job_name] = int(env_id)
        return _orchestrator_job_id_cache[job_name]
    workspace = w or WorkspaceClient()
    matches = list(workspace.jobs.list(name=job_name))
    if not matches:
        raise RuntimeError(
            f"Orchestrator job '{job_name}' not found. Deploy the "
            "enzyme_optimization submodule first: "
            "`./deploy.sh small_molecule aws --only-submodule "
            "enzyme_optimization/enzyme_optimization_v1`"
        )
    _orchestrator_job_id_cache[job_name] = int(matches[0].job_id)
    return _orchestrator_job_id_cache[job_name]


def _write_motif_pdb_to_volume(
    motif_pdb_str: str, catalog: str, schema: str
) -> str:
    """Upload the motif PDB to a per-run UC volume directory and return its
    absolute path. The Databricks Apps sandbox blocks direct
    `open("/Volumes/...")`, so we go through the SDK's Files API."""
    run_uuid = uuid.uuid4().hex[:12]
    volume_dir = (
        f"/Volumes/{catalog}/{schema}/{ORCHESTRATOR_VOLUME_DIR_NAME}/{run_uuid}"
    )
    motif_path = f"{volume_dir}/motif.pdb"
    w = WorkspaceClient()
    w.files.upload(
        file_path=motif_path,
        contents=io.BytesIO(motif_pdb_str.encode("utf-8")),
        overwrite=True,
    )
    return motif_path


@dataclass(frozen=True)
class JobDispatchResult:
    job_id: int
    job_run_id: int
    mlflow_run_id: str
    experiment_id: str


def start_enzyme_optimization_job(
    *,
    motif_pdb_str: str,
    motif_residues: list[int],
    target_chain: str,
    scaffold_length_min: int,
    scaffold_length_max: int,
    num_samples: int,
    num_iterations: int,
    weights: dict[str, float],
    user_info: UserInfo,
    mlflow_experiment: str,
    mlflow_run_name: str,
    substrate_smiles: str = "",
    references: Optional[list[dict]] = None,
    half_life_margin: float = 0.05,
    resampling_temperature: float = 0.1,
    strategy: str = "resample",
    run_proteinmpnn: bool = True,
    convergence_threshold: Optional[float] = 0.01,
    convergence_window: int = 2,
    target_reward: Optional[float] = None,
    best_k_target: Optional[int] = None,
    best_k_threshold: Optional[float] = None,
    use_inprocess_ame: bool = False,
) -> JobDispatchResult:
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    motif_pdb_path = _write_motif_pdb_to_volume(motif_pdb_str, catalog, schema)

    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment,
        user_email=user_info.user_email,
        host=None,
        token=None,
    )

    w = WorkspaceClient()
    job_id = _resolve_orchestrator_job_id(use_inprocess_ame=use_inprocess_ame, w=w)

    with mlflow.start_run(
        run_name=mlflow_run_name, experiment_id=experiment.experiment_id
    ) as pre_run:
        mlflow_run_id = pre_run.info.run_id
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "enzyme_optimization")
        mlflow.set_tag("created_by", user_info.user_email)
        # `job_status` is provisionally `submitted` so the search-past-runs
        # table shows the dispatch as in-flight. If `jobs.run_now` raises
        # below (e.g. CAN_MANAGE_RUN missing on the orchestrator job, UC
        # WRITE missing on the motif volume), the except handler flips it
        # to `failed` so the run doesn't sit at "Submitted" forever.
        mlflow.set_tag("job_status", "submitted")
        mlflow.log_param("generation_mode", "Accurate" if use_inprocess_ame else "Fast")
        mlflow.log_param("scaffold_length_min", scaffold_length_min)
        mlflow.log_param("scaffold_length_max", scaffold_length_max)
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
                    "motif_pdb_path": motif_pdb_path,
                    "motif_residues_csv": ",".join(str(r) for r in motif_residues),
                    "target_chain": target_chain,
                    "scaffold_length_min": str(scaffold_length_min),
                    "scaffold_length_max": str(scaffold_length_max),
                    "num_samples": str(num_samples),
                    "num_iterations": str(num_iterations),
                    "substrate_smiles": substrate_smiles or "",
                    "references_json": json.dumps(references or []),
                    "half_life_margin": str(half_life_margin),
                    "weights_json": json.dumps({**DEFAULT_AXIS_WEIGHTS, **(weights or {})}),
                    "resampling_temperature": str(resampling_temperature),
                    "strategy": strategy,
                    "run_proteinmpnn": str(run_proteinmpnn).lower(),
                    "dev_user_prefix": os.environ.get("DEV_USER_PREFIX", "") or "",
                    "convergence_threshold": str(convergence_threshold)
                    if convergence_threshold is not None else "0.01",
                    "convergence_window": str(int(convergence_window)),
                    "target_reward": str(target_reward) if target_reward is not None else "",
                    "best_k_target": str(int(best_k_target)) if best_k_target is not None else "",
                    "best_k_threshold": str(best_k_threshold) if best_k_threshold is not None else "",
                    "use_inprocess_ame": "true" if use_inprocess_ame else "false",
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


# ─── Smoke test (developability predictors on a single sequence) ──────────


_AXIS_DISPLAY_NAMES = {
    "solubility":  "NetSolP Solubility",
    "half_life":   "PLTNUM Half-Life Stability",
    "thermostab":  "DeepSTABp Tm",
    "immuno":      "MHCflurry Immunogenicity",
}


def predict_enzyme_properties(sequence: str) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for axis, display_name in _AXIS_DISPLAY_NAMES.items():
        try:
            if axis == "thermostab":
                records = [{"sequence": sequence, "growth_temp": 37.0, "mt_mode": "Cell"}]
            elif axis == "immuno":
                records = [{"sequence": sequence, "alleles": _DEFAULT_MHC_ALLELES}]
            else:
                records = [{"sequence": sequence}]
            name = get_endpoint_name(display_name)
            resp = _predictor_client.serving_endpoints.query(
                name=name, dataframe_records=records
            )
            preds = pd.DataFrame(resp.predictions)
            if axis == "solubility":
                out[axis] = float(preds["predicted_solubility"].iloc[0])
            elif axis == "half_life":
                out[axis] = float(preds["predicted_stability"].iloc[0])
            elif axis == "thermostab":
                out[axis] = float(preds["predicted_tm_celsius"].iloc[0])
            elif axis == "immuno":
                out[axis] = float(preds["predicted_immuno_burden"].iloc[0])
        except Exception as e:
            logger.warning("predict_enzyme_properties: %s failed: %s", axis, e)
            out[axis] = None
    return out


# ─── Result loaders ────────────────────────────────────────────────────────


def get_run_status(run_id: str) -> dict[str, Any]:
    """Run status + iter_max_reward / iter_mean_reward history + the
    current metrics dict. Empty histories while the orchestrator is still
    warming up; populated as soon as the loop logs its first iteration."""
    client = MlflowClient()
    run = client.get_run(run_id)
    iter_max = client.get_metric_history(run_id, "iter_max_reward")
    iter_mean = client.get_metric_history(run_id, "iter_mean_reward")
    job_status = run.data.tags.get("job_status", "")
    return {
        "status": run.info.status,
        "job_status": job_status,
        "iter_max_reward_history": [
            {"step": m.step, "value": float(m.value)} for m in iter_max
        ],
        "iter_mean_reward_history": [
            {"step": m.step, "value": float(m.value)} for m in iter_mean
        ],
        "current_metrics": {k: float(v) for k, v in run.data.metrics.items()},
        "experiment_id": run.info.experiment_id,
        "run_name": run.data.tags.get("mlflow.runName", ""),
    }


def load_optimization_trajectory(run_id: str) -> pd.DataFrame:
    """`results/reward_trajectory.csv` from the run's artifacts; empty if
    not logged yet."""
    client = MlflowClient()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(
                run_id, "results/reward_trajectory.csv", dst_path=tmp
            )
            return pd.read_csv(local)
    except Exception as e:
        logger.info("trajectory not yet available for run %s: %s", run_id, e)
        return pd.DataFrame()


def load_top_k_pdbs(run_id: str) -> dict[str, str]:
    """`results/topK_pdbs/*.pdb` from the run's artifacts → `{cand_id:
    pdb_string}`. Empty if not logged yet."""
    client = MlflowClient()
    out: dict[str, str] = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = client.download_artifacts(
                run_id, "results/topK_pdbs", dst_path=tmp
            )
            for fname in sorted(os.listdir(local_dir)):
                if not fname.endswith(".pdb"):
                    continue
                with open(os.path.join(local_dir, fname)) as f:
                    out[fname[:-4]] = f.read()
    except Exception as e:
        logger.info("topK PDBs not yet available for run %s: %s", run_id, e)
    return out


# ─── Search past runs ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class EnzymeRun:
    run_id: str
    run_name: str
    experiment_name: str
    generation_mode: str
    iter_max_reward: float | None
    iterations_completed: int | None
    start_time_ms: int | None
    job_status: str
    progress: str
    # Workspace UI link to the dispatched orchestrator-job run.
    run_url: str = ""


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


def _safe_int(v) -> int | None:
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None  # NaN guard
    except (ValueError, TypeError):
        return None


def _start_time_ms(v) -> int | None:
    try:
        if pd.isna(v):
            return None
        if hasattr(v, "value"):  # pandas Timestamp
            return int(v.value // 1_000_000)
        return int(v)
    except (ValueError, TypeError, AttributeError):
        return None


def _format_runs(runs: pd.DataFrame, exp_map: dict[str, str]) -> list[EnzymeRun]:
    if runs.empty:
        return []
    runs = runs.copy()
    runs["experiment_name"] = runs["experiment_id"].map(exp_map)
    out: list[EnzymeRun] = []
    for _, r in runs.iterrows():
        job_status = str(r.get("tags.job_status", "") or "")
        gen_mode = str(r.get("params.generation_mode", "") or "")
        job_run_id = str(r.get("tags.job_run_id", "") or "")
        try:
            job_id = _resolve_orchestrator_job_id(
                use_inprocess_ame=(gen_mode == "Accurate")
            )
        except Exception:
            job_id = 0
        out.append(
            EnzymeRun(
                run_id=str(r.get("run_id", "")),
                run_name=str(r.get("tags.mlflow.runName", "") or ""),
                experiment_name=str(r.get("experiment_name", "") or ""),
                generation_mode=gen_mode,
                iter_max_reward=_safe_float(r.get("metrics.iter_max_reward")),
                iterations_completed=_safe_int(r.get("metrics.iterations_completed")),
                start_time_ms=_start_time_ms(r.get("start_time")),
                job_status=job_status,
                progress=_progress(job_status),
                run_url=job_run_url(job_id, job_run_id) if job_id else "",
            )
        )
    return out


def _experiment_map() -> dict[str, str]:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    experiments = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    return {e.experiment_id: e.name.split("/")[-1] for e in experiments}


def search_runs(
    user_email: str, by: str, text: str
) -> list[EnzymeRun]:
    """`by` is 'run_name' or 'experiment_name'. Case-insensitive contains."""
    exp_map = _experiment_map()
    if not exp_map:
        return []

    if by == "experiment_name":
        needle = text.upper()
        exp_map = {
            eid: name
            for eid, name in exp_map.items()
            if needle in name.upper()
        }
        if not exp_map:
            return []

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='enzyme_optimization' AND "
            f"tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=list(exp_map.keys()),
    )
    if runs.empty:
        return []

    if by == "run_name":
        runs = runs[
            runs["tags.mlflow.runName"]
            .astype(str)
            .str.contains(text, case=False, na=False)
        ]
        if runs.empty:
            return []

    return _format_runs(runs, exp_map)

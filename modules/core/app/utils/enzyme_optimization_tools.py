"""App-side dispatcher and result loaders for the Guided Enzyme Optimization
workflow.

Mirrors the shape of `start_scanpy_job` in `single_cell_analysis.py`:
  * `start_enzyme_optimization_job` writes the motif PDB to a UC volume,
    looks up the orchestrator job by name, and dispatches it via
    `w.jobs.run_now(...)`. Returns (job_id, run_id).
  * `predict_enzyme_properties` is a single-call dispatcher across the four
    new developability predictor endpoints (NetSolP, PLTNUM, DeepSTABp,
    MHCflurry) — used for the smoke-test button in the Streamlit page.
  * `load_optimization_*` helpers read the orchestrator's MLflow artifacts
    so the results view can render the reward trajectory and top-K PDBs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from mlflow.tracking import MlflowClient

from genesis_workbench.models import set_mlflow_experiment
from genesis_workbench.workbench import UserInfo

from .streamlit_helper import get_endpoint_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ORCHESTRATOR_JOB_NAME = "run_enzyme_optimization_gwb"
ORCHESTRATOR_JOB_NAME_ACCURATE = "run_enzyme_optimization_gwb_inprocess_ame"
ORCHESTRATOR_VOLUME_DIR_NAME = "enzyme_optimization"

DEFAULT_AXIS_WEIGHTS: Dict[str, float] = {
    "motif_rmsd": 1.0,
    "plddt":      1.0,
    "boltz":      0.5,
    "solubility": 1.0,
    "half_life":  1.0,
    "thermostab": 1.0,
    "immuno":     1.0,
}

# Long-timeout client for predictor endpoints (some are GPU cold-starts).
_PREDICTOR_TIMEOUT_SECONDS = 600
_predictor_client = WorkspaceClient(
    config=Config(http_timeout_seconds=_PREDICTOR_TIMEOUT_SECONDS)
)

# Cached job-id lookup so we don't hit the Jobs API on every form submission.
# Keyed by job name so Fast and Accurate jobs cache independently.
_orchestrator_job_id_cache: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# Job dispatch
# ---------------------------------------------------------------------------

def _resolve_orchestrator_job_id(use_inprocess_ame: bool = False,
                                 workspace_client: Optional[WorkspaceClient] = None) -> int:
    """Find the orchestrator job by name. Cached after first lookup.

    The Streamlit form's Generation-mode toggle picks between two jobs:
    - Fast (use_inprocess_ame=False): `run_enzyme_optimization_gwb` (CPU)
    - Accurate (use_inprocess_ame=True):
      `run_enzyme_optimization_gwb_inprocess_ame` (A10 GPU)

    Falls back to env vars `RUN_ENZYME_OPTIMIZATION_JOB_ID` /
    `RUN_ENZYME_OPTIMIZATION_INPROCESS_AME_JOB_ID` if the deploy flow ever
    wires them up.
    """
    job_name = ORCHESTRATOR_JOB_NAME_ACCURATE if use_inprocess_ame else ORCHESTRATOR_JOB_NAME
    if job_name in _orchestrator_job_id_cache:
        return _orchestrator_job_id_cache[job_name]

    env_var = ("RUN_ENZYME_OPTIMIZATION_INPROCESS_AME_JOB_ID"
               if use_inprocess_ame else "RUN_ENZYME_OPTIMIZATION_JOB_ID")
    env_id = os.environ.get(env_var)
    if env_id:
        _orchestrator_job_id_cache[job_name] = int(env_id)
        return _orchestrator_job_id_cache[job_name]

    w = workspace_client or WorkspaceClient()
    matches = list(w.jobs.list(name=job_name))
    if not matches:
        raise RuntimeError(
            f"Orchestrator job '{job_name}' not found. "
            "Deploy the enzyme_optimization submodule first: "
            "./deploy.sh small_molecule aws --only-submodule "
            "enzyme_optimization/enzyme_optimization_v1"
        )
    _orchestrator_job_id_cache[job_name] = int(matches[0].job_id)
    return _orchestrator_job_id_cache[job_name]


def _write_motif_pdb_to_volume(motif_pdb_str: str, catalog: str, schema: str) -> str:
    """Write the motif PDB into the orchestrator's UC volume under a
    per-run directory. Returns the absolute volume path that the job will read.
    """
    run_uuid = uuid.uuid4().hex[:12]
    volume_dir = f"/Volumes/{catalog}/{schema}/{ORCHESTRATOR_VOLUME_DIR_NAME}/{run_uuid}"
    os.makedirs(volume_dir, exist_ok=True)
    motif_path = os.path.join(volume_dir, "motif.pdb")
    with open(motif_path, "w") as f:
        f.write(motif_pdb_str)
    return motif_path


def start_enzyme_optimization_job(
    motif_pdb_str: str,
    motif_residues: List[int],
    target_chain: str,
    scaffold_length_min: int,
    scaffold_length_max: int,
    num_samples: int,
    num_iterations: int,
    weights: Dict[str, float],
    user_info: UserInfo,
    mlflow_experiment: str,
    mlflow_run_name: str,
    substrate_smiles: str = "",
    references: Optional[List[Dict[str, Any]]] = None,
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
) -> Tuple[int, str]:
    """Dispatch the orchestrator job and return (job_id, run_id).

    The motif PDB is written to a UC volume; everything else is passed via
    job parameters. The orchestrator notebook creates its own MLflow run
    inside `mlflow_experiment` named `mlflow_run_name`.

    `use_inprocess_ame` toggles the dispatcher between the two job specs:
    False → CPU job + endpoint-based AME (Fast); True → A10 GPU job + in-process
    AME with FK steering (Accurate). The toggle is also forwarded as a job
    parameter so the notebook fails loud on a misdispatch.
    """
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]

    motif_pdb_path = _write_motif_pdb_to_volume(motif_pdb_str, catalog, schema)

    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment,
        user_email=user_info.user_email,
    )

    w = WorkspaceClient()
    job_id = _resolve_orchestrator_job_id(use_inprocess_ame=use_inprocess_ame,
                                          workspace_client=w)

    job_run = w.jobs.run_now(
        job_id=job_id,
        job_parameters={
            "catalog": catalog,
            "schema": schema,
            "cache_dir": ORCHESTRATOR_VOLUME_DIR_NAME,
            "sql_warehouse_id": os.environ["SQL_WAREHOUSE"],
            "user_email": user_info.user_email,

            "mlflow_experiment": experiment.name,
            "mlflow_run_name": mlflow_run_name,
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

            # Stopping criteria. Empty strings disable the opt-in modes; the
            # convergence mode defaults to ON via the job-spec default values.
            "convergence_threshold": str(convergence_threshold)
                if convergence_threshold is not None else "0.01",
            "convergence_window": str(int(convergence_window)),
            "target_reward": str(target_reward) if target_reward is not None else "",
            "best_k_target": str(int(best_k_target)) if best_k_target is not None else "",
            "best_k_threshold": str(best_k_threshold) if best_k_threshold is not None else "",

            # Generation-mode toggle. The cluster-bound default is also set,
            # but we forward at the param level so a misdispatch fails loud.
            "use_inprocess_ame": "true" if use_inprocess_ame else "false",
        },
    )
    return job_id, job_run.run_id


# ---------------------------------------------------------------------------
# Single-endpoint smoke test (used by the "Test on T4 lysozyme" button)
# ---------------------------------------------------------------------------

# T4 lysozyme — a canonical, well-behaved enzyme for smoke testing.
T4_LYSOZYME_SEQUENCE = (
    "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYK"
)

_AXIS_DISPLAY_NAMES = {
    "solubility":  "NetSolP Solubility",
    "half_life":   "PLTNUM Half-Life Stability",
    "thermostab":  "DeepSTABp Tm",
    "immuno":      "MHCflurry Immunogenicity",
}


def _query_predictor(display_name: str, payload: Any) -> Any:
    name = get_endpoint_name(display_name)
    return _predictor_client.serving_endpoints.query(name=name, inputs=payload)


def predict_enzyme_properties(sequence: str) -> Dict[str, Optional[float]]:
    """Single-sequence smoke test across all four developability predictors.

    Returns a dict with keys solubility / half_life / thermostab / immuno.
    A failed endpoint returns None for that axis (so the UI can still render
    a partial row).
    """
    out: Dict[str, Optional[float]] = {}
    df = pd.DataFrame({"sequence": [sequence]}).to_dict(orient="split")

    for axis, display_name in _AXIS_DISPLAY_NAMES.items():
        try:
            resp = _query_predictor(display_name, df)
            preds = pd.DataFrame(resp.predictions)
            if axis == "solubility":
                out[axis] = float(preds["predicted_solubility"].iloc[0])
            elif axis == "half_life":
                out[axis] = float(preds["predicted_stability"].iloc[0])
            elif axis == "thermostab":
                # DeepSTABp expects extra columns; supply defaults.
                payload = pd.DataFrame({
                    "sequence": [sequence],
                    "growth_temp": [37.0],
                    "mt_mode": ["Cell"],
                }).to_dict(orient="split")
                resp = _query_predictor(display_name, payload)
                preds = pd.DataFrame(resp.predictions)
                out[axis] = float(preds["predicted_tm_celsius"].iloc[0])
            elif axis == "immuno":
                out[axis] = float(preds["predicted_immuno_burden"].iloc[0])
        except Exception as e:
            logger.warning(f"predict_enzyme_properties: {axis} endpoint failed: {e}")
            out[axis] = None
    return out


# ---------------------------------------------------------------------------
# Result loaders (for the Streamlit results view)
# ---------------------------------------------------------------------------

def load_optimization_trajectory(run_id: str) -> pd.DataFrame:
    """Read `results/reward_trajectory.csv` from the run's artifacts.
    Returns an empty DataFrame if the run hasn't logged it yet.
    """
    client = MlflowClient()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run_id, "results/reward_trajectory.csv", dst_path=tmp)
            return pd.read_csv(local)
    except Exception as e:
        logger.info(f"trajectory not yet available for run {run_id}: {e}")
        return pd.DataFrame()


def load_top_k_pdbs(run_id: str) -> Dict[str, str]:
    """Returns dict {candidate_id: pdb_string} of the top-K artifacts."""
    client = MlflowClient()
    out: Dict[str, str] = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = client.download_artifacts(run_id, "results/topK_pdbs", dst_path=tmp)
            for fname in sorted(os.listdir(local_dir)):
                if not fname.endswith(".pdb"):
                    continue
                with open(os.path.join(local_dir, fname)) as f:
                    out[fname[:-4]] = f.read()
    except Exception as e:
        logger.info(f"topK PDBs not yet available for run {run_id}: {e}")
    return out


def get_run_status(run_id: str) -> Dict[str, Any]:
    """Returns {status, iter_max_reward_history, iter_mean_reward_history}.
    The history lists are length=N once the run is finished, shorter while
    the loop is still iterating."""
    client = MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    iter_max = client.get_metric_history(run_id, "iter_max_reward")
    iter_mean = client.get_metric_history(run_id, "iter_mean_reward")
    return {
        "status": run.info.status,
        "iter_max_reward_history": [(m.step, m.value) for m in iter_max],
        "iter_mean_reward_history": [(m.step, m.value) for m in iter_mean],
        "current_metrics": dict(metrics),
        "experiment_id": run.info.experiment_id,
    }


def wait_for_completion(run_id: str, poll_interval_sec: float = 10.0,
                        timeout_sec: float = 86400.0) -> str:
    """Block until the orchestrator's MLflow run reaches a terminal state.
    Returns the final status. Used by integration tests; the Streamlit UI
    polls `get_run_status` instead."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        status = get_run_status(run_id)["status"]
        if status in ("FINISHED", "FAILED", "KILLED"):
            return status
        time.sleep(poll_interval_sec)
    return "TIMEOUT"

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
import mlflow
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

    Databricks Apps run in a sandboxed container that does NOT have POSIX
    access to ``/Volumes/...`` — ``open(volume_path, "w")`` raises
    ``PermissionError: [Errno 13] Permission denied: '/Volumes'``. The
    Databricks SDK's Files API is the supported channel: it uploads the
    bytes via the workspace's UC volume backend using the app SP's auth
    context, and auto-creates intermediate directories.
    """
    import io

    run_uuid = uuid.uuid4().hex[:12]
    volume_dir = f"/Volumes/{catalog}/{schema}/{ORCHESTRATOR_VOLUME_DIR_NAME}/{run_uuid}"
    motif_path = f"{volume_dir}/motif.pdb"

    w = WorkspaceClient()
    w.files.upload(
        file_path=motif_path,
        contents=io.BytesIO(motif_pdb_str.encode("utf-8")),
        overwrite=True,
    )
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
) -> Tuple[int, int]:
    """Dispatch the orchestrator job and return (job_id, job_run_id).

    The motif PDB is written to a UC volume; everything else is passed via
    job parameters. The orchestrator notebook creates its own MLflow run
    inside `mlflow_experiment` named `mlflow_run_name`. Past results are
    discoverable via `search_enzyme_optimization_runs_by_*` once the run
    has logged at least one iteration.

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

    # Pre-create the MLflow run from the app so the row shows up in
    # "Search Past Runs" the moment the user clicks Launch — same pattern as
    # disease_biology.start_parabricks_alignment / start_gwas_analysis.
    # The orchestrator re-attaches via `mlflow.start_run(run_id=...)` once
    # its cluster is ready.
    with mlflow.start_run(
        run_name=mlflow_run_name,
        experiment_id=experiment.experiment_id,
    ) as pre_run:
        mlflow_run_id = pre_run.info.run_id

        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "enzyme_optimization")
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_status", "submitted")

        mlflow.log_param("generation_mode", "Accurate" if use_inprocess_ame else "Fast")
        mlflow.log_param("scaffold_length_min", scaffold_length_min)
        mlflow.log_param("scaffold_length_max", scaffold_length_max)
        mlflow.log_param("num_samples", num_samples)
        mlflow.log_param("num_iterations", num_iterations)

        job_run = w.jobs.run_now(
            job_id=job_id,
            job_parameters={
                "catalog": catalog,
                "schema": schema,
                "cache_dir": ORCHESTRATOR_VOLUME_DIR_NAME,
                "sql_warehouse_id": os.environ["SQL_WAREHOUSE"],
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
        mlflow.set_tag("job_run_id", str(job_run.run_id))

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


def _query_predictor(display_name: str, records: list) -> Any:
    """Send a list-of-records (named columns) payload to a serving endpoint.

    `dataframe_records` is the only payload shape the four developability
    endpoints accept reliably — `inputs=` with `df.to_dict(orient="split")`
    gets rejected by MLflow's schema enforcement as "extra inputs:
    ['index', 'columns', 'data']" because it's interpreted as raw tensor
    input rather than a DataFrame.
    """
    name = get_endpoint_name(display_name)
    return _predictor_client.serving_endpoints.query(
        name=name, dataframe_records=records,
    )


# 6-allele Sette-style HLA panel — matches the orchestrator default in
# `enzyme_optimization_v1/notebooks/utils.py:call_mhcflurry`.
_DEFAULT_MHC_ALLELES = (
    "HLA-A*02:01,HLA-A*01:01,HLA-B*07:02,HLA-B*44:02,HLA-C*07:01,HLA-C*04:01"
)


def predict_enzyme_properties(sequence: str) -> Dict[str, Optional[float]]:
    """Single-sequence smoke test across all four developability predictors.

    Returns a dict with keys solubility / half_life / thermostab / immuno.
    A failed endpoint returns None for that axis (so the UI can still render
    a partial row).
    """
    out: Dict[str, Optional[float]] = {}

    for axis, display_name in _AXIS_DISPLAY_NAMES.items():
        try:
            # Build the per-axis payload. Each endpoint declares its own
            # required columns in its MLflow signature; sending the same
            # naked `[{"sequence": s}]` to all of them gets rejected by
            # MHCflurry (needs `alleles`) and DeepSTABp (needs growth_temp +
            # mt_mode).
            if axis == "thermostab":
                records = [{
                    "sequence": sequence,
                    "growth_temp": 37.0,
                    "mt_mode": "Cell",
                }]
            elif axis == "immuno":
                records = [{
                    "sequence": sequence,
                    "alleles": _DEFAULT_MHC_ALLELES,
                }]
            else:  # solubility, half_life
                records = [{"sequence": sequence}]

            resp = _query_predictor(display_name, records)
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


# ---------------------------------------------------------------------------
# Past-run search (mirrors the AlphaFold + Disease Biology pattern)
# ---------------------------------------------------------------------------
#
# Filter convention (matches `protein_structure.search_alphafold_runs_*`):
#   * experiments with `tags.used_by_genesis_workbench='yes'`
#   * runs with `tags.feature='enzyme_optimization' AND
#               tags.origin='genesis_workbench' AND
#               tags.created_by=<email>`
# Surface columns (run_id is hidden in the UI's data_editor):
#   run_id, run_name, experiment_name, generation_mode (Fast/Accurate),
#   iter_max_reward, iterations_completed, start_time,
#   job_status (orchestrator-set progressive stage:
#               started → iter_<N>_generating → iter_<N>_redesigning →
#               iter_<N>_scoring → iter_<N>_complete → complete).

_SEARCH_COLUMNS = [
    "run_id",
    "tags.mlflow.runName",
    "experiment_name",
    "params.generation_mode",
    "metrics.iter_max_reward",
    "metrics.iterations_completed",
    "start_time",
    "tags.job_status",
]


# 4-stage progress mapping. Per-iteration sub-stages (`iter_<N>_*`) all collapse
# to the same dot pattern because N is variable (1-30) and we want a fixed-width
# visual. Mirrors `disease_biology._PROGRESS_MAP`.
_ENZYME_PROGRESS_MAP = {
    "submitted": "🟩⬜⬜⬜",
    "started":   "🟩🟩⬜⬜",
    "complete":  "🟩🟩🟩🟩",
    "failed":    "🟥",
    "unknown":   "⬜⬜⬜⬜",
}


def _enzyme_progress(status: str) -> str:
    if not status:
        return _ENZYME_PROGRESS_MAP["unknown"]
    if status in _ENZYME_PROGRESS_MAP:
        return _ENZYME_PROGRESS_MAP[status]
    if status.startswith("iter_"):
        return "🟩🟩🟩⬜"
    return _ENZYME_PROGRESS_MAP["unknown"]


def _add_progress_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "job_status" not in df.columns:
        return df
    df = df.copy()
    df["progress"] = df["job_status"].map(_enzyme_progress)
    cols = list(df.columns)
    cols.remove("progress")
    idx = cols.index("job_status") + 1
    cols.insert(idx, "progress")
    return df[cols]


def _format_search_runs(runs: pd.DataFrame, exp_map: Dict[str, str]) -> pd.DataFrame:
    """Common formatter — projects the columns we surface and renames them
    to short forms (drop `tags.` / `params.` / `metrics.` prefixes).

    The orchestrator's `stop_reason` tag is intentionally NOT surfaced here;
    users who care about why a run exited early can read it from the MLflow
    run's tags via the View dialog's "View in MLflow" link.
    """
    runs = runs.copy()
    runs["experiment_name"] = runs["experiment_id"].map(exp_map)
    cols = [c for c in _SEARCH_COLUMNS if c in runs.columns]
    out = runs[cols].copy()
    out.columns = [c.split(".")[-1] for c in out.columns]
    out = out.rename(columns={"runName": "run_name"})
    return _add_progress_column(out)


def search_enzyme_optimization_runs_by_run_name(user_email: str, run_name: str) -> pd.DataFrame:
    """Returns runs whose `tags.mlflow.runName` contains `run_name` (case-
    insensitive) and were tagged by the orchestrator as
    `feature=enzyme_optimization`. Empty DataFrame if no match.
    """
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    experiment_list = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if not experiment_list:
        return pd.DataFrame()
    exp_map = {e.experiment_id: e.name.split("/")[-1] for e in experiment_list}

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='enzyme_optimization' AND "
            f"tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=list(exp_map.keys()),
    )
    if runs.empty:
        return pd.DataFrame()
    runs = runs[runs["tags.mlflow.runName"].str.contains(run_name, case=False, na=False)]
    if runs.empty:
        return pd.DataFrame()
    return _format_search_runs(runs, exp_map)


def search_enzyme_optimization_runs_by_experiment_name(user_email: str, experiment_name: str) -> pd.DataFrame:
    """Returns runs in any experiment whose name (last path segment) contains
    `experiment_name` case-insensitively, restricted to this user's
    enzyme_optimization runs. Empty DataFrame if no match.
    """
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    all_experiments = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if not all_experiments:
        return pd.DataFrame()

    needle = experiment_name.upper()
    exp_map = {
        e.experiment_id: e.name.split("/")[-1]
        for e in all_experiments
        if needle in e.name.split("/")[-1].upper()
    }
    if not exp_map:
        return pd.DataFrame()

    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='enzyme_optimization' AND "
            f"tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=list(exp_map.keys()),
    )
    if runs.empty:
        return pd.DataFrame()
    return _format_search_runs(runs, exp_map)

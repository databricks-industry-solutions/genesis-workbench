"""Orchestrator utilities for the guided enzyme optimization loop.

Three concerns live here:

1. Endpoint helpers — thin wrappers around the eight serving endpoints used by
   the loop (Proteina-Complexa-AME, ESMFold, ProteinMPNN, Boltz, NetSolP,
   PLTNUM, DeepSTABp, MHCflurry). All use a long-timeout WorkspaceClient
   so cold-start GPU endpoints don't time out at the SDK's 60s default.

2. Reward composer — z-score-then-min-max normalize each axis within an
   iteration's batch, weighted-sum, with a special anchor-based mechanism
   for the half-life axis (relative-only PLTNUM ranker → real signal anchored
   against user-supplied reference enzymes).

3. Strategy interface — abstract `Strategy` with two Phase-1 implementations:
   `ResampleFromAMEStrategy` (default) and `NoOpStrategy` (for verification).
   Phase 2 will add `EvolutionaryStrategy` without touching the orchestrator core.

Endpoint name resolution mirrors `modules/core/app/utils/streamlit_helper.py:get_endpoint_name`
verbatim so the orchestrator and the Streamlit app always resolve to the same
endpoint regardless of `dev_user_prefix`.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoint name resolution + long-timeout client
# ---------------------------------------------------------------------------

# Maps an axis tag → the UC name fragment used in the endpoint name. Mirrors
# the canonical mapping in modules/core/app/utils/streamlit_helper.py so the
# orchestrator and app stay in sync.
_AXIS_TO_UC_NAME = {
    "ame":         "proteina_complexa_ame",
    "esmfold":     "esmfold",
    "proteinmpnn": "proteinmpnn",
    "boltz":       "boltz",
    "netsolp":     "netsolp_v1",
    "pltnum":      "pltnum_v1",
    "deepstabp":   "deepstabp_v1",
    "mhcflurry":   "mhcflurry_v2",
}

# All eight serving endpoints have scale_to_zero=true, so the first call after
# an idle period hits a cold start that can take 5-10 minutes for the bigger
# models (AME, ESMFold, DeepSTABp's ProtT5 backbone). The orchestrator's loop
# also starves middle-of-loop calls if upstream candidates take long enough
# for an endpoint to scale back down — we hit this on Phase 1.5 verification
# runs both for AME (iter-2 cold start, 10+ min) and DeepSTABp (iter-1 scoring
# cold start). Default 1200s gives every call a 20-minute ceiling; warm-path
# calls still return in 1-2 min.
_DEFAULT_TIMEOUT_SECONDS = 1200
_long_client = WorkspaceClient(
    config=Config(http_timeout_seconds=_DEFAULT_TIMEOUT_SECONDS)
)


def endpoint_name(axis: str, dev_user_prefix: Optional[str] = None) -> str:
    """Resolve the serving-endpoint name for an axis tag (e.g. 'esmfold')."""
    uc_name = _AXIS_TO_UC_NAME[axis]
    prefix = dev_user_prefix or os.environ.get("DEV_USER_PREFIX")
    if prefix and prefix.strip().lower() not in ("", "none"):
        return f"gwb_{prefix}_{uc_name}_endpoint"
    return f"gwb_{uc_name}_endpoint"


def _query(axis: str, inputs: Any, dev_user_prefix: Optional[str] = None,
           timeout_seconds: Optional[int] = None) -> Any:
    if timeout_seconds and timeout_seconds != _DEFAULT_TIMEOUT_SECONDS:
        client = WorkspaceClient(
            config=Config(http_timeout_seconds=timeout_seconds)
        )
    else:
        client = _long_client
    name = endpoint_name(axis, dev_user_prefix=dev_user_prefix)
    return client.serving_endpoints.query(name=name, inputs=inputs)


# ---------------------------------------------------------------------------
# Endpoint helpers
# ---------------------------------------------------------------------------

def call_ame(target_pdb: str, target_chain: str, length_min: int,
             length_max: int, num_samples: int,
             hotspots: str = "",
             dev_user_prefix: Optional[str] = None) -> pd.DataFrame:
    """Call Proteina-Complexa-AME and normalize the response columns.

    The deployed endpoint returns columns ``[sample_id, pdb_output, sequence,
    rewards]``. The orchestrator works in terms of ``designed_pdb`` /
    ``designed_sequence``, so we rename here at the boundary — keeps
    endpoint-specific naming out of the loop body.
    """
    payload = [{
        "target_pdb": target_pdb,
        "binder_length_min": int(length_min),
        "binder_length_max": int(length_max),
        "num_samples": int(num_samples),
        "hotspot_residues": hotspots,
        "target_chain": target_chain,
    }]
    # AME has scale_to_zero=true and the model is large — a fully cold start
    # has been observed at >20 minutes on GPU_SMALL. Default _query (1200s) is
    # still too tight for the worst case. Use 1800s for AME specifically;
    # warm-path calls still return in 1-2 min. The orchestrator also warms
    # AME at job start (warmup_ame_endpoint) so iter-1 typically lands warm.
    resp = _query("ame", payload, dev_user_prefix=dev_user_prefix,
                  timeout_seconds=1800)
    df = pd.DataFrame(resp.predictions)
    rename_map = {}
    if "pdb_output" in df.columns:
        rename_map["pdb_output"] = "designed_pdb"
    if "sequence" in df.columns and "designed_sequence" not in df.columns:
        rename_map["sequence"] = "designed_sequence"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def call_esmfold(sequence: str,
                 dev_user_prefix: Optional[str] = None) -> Dict[str, Any]:
    """Returns dict with keys: pdb (str), mean_plddt (float).
    Some ESMFold endpoint variants return just the PDB string — handle both."""
    resp = _query("esmfold", [sequence], dev_user_prefix=dev_user_prefix)
    out = resp.predictions[0]
    if isinstance(out, dict):
        return {
            "pdb": out.get("pdb", ""),
            "mean_plddt": float(out.get("mean_plddt", out.get("plddt", 0.0))),
        }
    return {"pdb": str(out), "mean_plddt": _extract_mean_plddt_from_pdb(str(out))}


def _extract_mean_plddt_from_pdb(pdb_str: str) -> float:
    """ESMFold writes pLDDT into the B-factor column. Mean of CA B-factors."""
    plddts = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                plddts.append(float(line[60:66]))
            except ValueError:
                continue
    return float(np.mean(plddts)) if plddts else 0.0


def call_proteinmpnn(pdb_str: str,
                     fixed_positions: Optional[Dict[str, List[int]]] = None,
                     dev_user_prefix: Optional[str] = None) -> List[str]:
    """Call ProteinMPNN to redesign the sequence for a given backbone.

    Args:
        pdb_str: PDB string of the backbone to redesign.
        fixed_positions: Optional ``{chain_id: [residue_numbers]}`` mapping.
            Listed positions keep their input amino-acid identity; everything
            else is redesigned. Used to preserve catalytic motif residues
            (e.g. the His-Asp-Ser triad of a serine protease) so MPNN can't
            silently swap them out.
        dev_user_prefix: ``DEV_USER_PREFIX`` override for endpoint name resolution.

    Returns:
        List of redesigned amino-acid sequence strings.
    """
    # The PyFunc signature has named columns ``pdb`` (string) + ``fixed_positions``
    # (string, JSON-encoded). MLflow's ColSpec enforcement silently drops nested
    # dicts from records when the column type is "string", so we always JSON-
    # encode fixed_positions on the wire and the PyFunc decodes it inside predict().
    fp_str = json.dumps(fixed_positions) if fixed_positions else ""
    payload: Any = [{"pdb": pdb_str, "fixed_positions": fp_str}]
    resp = _query("proteinmpnn", payload, dev_user_prefix=dev_user_prefix)
    return [str(s) for s in resp.predictions]


def call_boltz(boltz_input: str,
               dev_user_prefix: Optional[str] = None,
               timeout_seconds: int = 900) -> Dict[str, Any]:
    """boltz_input examples:
        'protein_A:MKVL...'                       — apo
        'protein_A:MKVL...;smiles_B:CC(=O)OC1...' — ligand complex
    Returns dict: { pdb: str, ipTM: float?, iLDDT: float?, ... }
    """
    payload = [{
        "input": boltz_input,
        "msa": "no_msa",
        "use_msa_server": "True",
    }]
    resp = _query("boltz", payload, dev_user_prefix=dev_user_prefix,
                  timeout_seconds=timeout_seconds)
    pred = resp.predictions[0] if resp.predictions else {}
    if isinstance(pred, str):
        return {"pdb": pred}
    return pred


def call_netsolp(sequences: List[str],
                 dev_user_prefix: Optional[str] = None) -> List[float]:
    payload = [{"sequence": s} for s in sequences]
    resp = _query("netsolp", payload, dev_user_prefix=dev_user_prefix)
    out = pd.DataFrame(resp.predictions)
    return [float(v) for v in out["predicted_solubility"]]


def call_pltnum(sequences: List[str],
                dev_user_prefix: Optional[str] = None) -> List[float]:
    payload = [{"sequence": s} for s in sequences]
    resp = _query("pltnum", payload, dev_user_prefix=dev_user_prefix)
    out = pd.DataFrame(resp.predictions)
    return [float(v) for v in out["predicted_stability"]]


def call_deepstabp(sequences: List[str], growth_temp: float = 37.0,
                   mt_mode: str = "Cell",
                   dev_user_prefix: Optional[str] = None) -> List[float]:
    payload = [
        {"sequence": s, "growth_temp": float(growth_temp), "mt_mode": str(mt_mode)}
        for s in sequences
    ]
    resp = _query("deepstabp", payload, dev_user_prefix=dev_user_prefix)
    out = pd.DataFrame(resp.predictions)
    return [float(v) for v in out["predicted_tm_celsius"]]


# Default 6-allele Sette-style HLA panel — the immunogenicity endpoint declares
# ``alleles`` as required, so we never send None on the wire.
_DEFAULT_MHC_ALLELES = (
    "HLA-A*02:01,HLA-A*01:01,HLA-B*07:02,HLA-B*44:02,HLA-C*07:01,HLA-C*04:01"
)


def call_mhcflurry(sequences: List[str], alleles: Optional[str] = None,
                   dev_user_prefix: Optional[str] = None) -> List[float]:
    a = alleles or _DEFAULT_MHC_ALLELES
    payload = [{"sequence": s, "alleles": a} for s in sequences]
    resp = _query("mhcflurry", payload, dev_user_prefix=dev_user_prefix)
    out = pd.DataFrame(resp.predictions)
    return [float(v) for v in out["predicted_immuno_burden"]]


# ---------------------------------------------------------------------------
# Reward composer
# ---------------------------------------------------------------------------

@dataclass
class PredictorAxis:
    """One axis of the composite reward function."""
    name: str
    weight: float
    lower_is_better: bool = False
    # If set, the axis's per-candidate scores are *already* normalized to [0,1]
    # (e.g. via the half-life anchor) and skip the in-batch z-score+min-max step.
    pre_normalized: bool = False

    @property
    def enabled(self) -> bool:
        return self.weight > 0


def _zscore_then_minmax(values: List[float], lower_is_better: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    # Replace NaN with the worst-case (minimum after sign flip) so failed
    # candidates get the lowest normalized score in the batch instead of
    # poisoning the mean/std into NaN. If every value is NaN, return zeros.
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return np.zeros_like(arr)
    if nan_mask.any():
        arr = arr.copy()
        finite = arr[~nan_mask]
        # Worst direction = max if lower_is_better else min (penalty before sign flip)
        worst = float(finite.max()) if lower_is_better else float(finite.min())
        arr[nan_mask] = worst
    if lower_is_better:
        arr = -arr
    if arr.std() < 1e-12:
        return np.zeros_like(arr)
    z = (arr - arr.mean()) / arr.std()
    rng = z.max() - z.min()
    if rng < 1e-12:
        return np.zeros_like(arr)
    return (z - z.min()) / rng


def compose_rewards(per_axis_scores: Dict[str, List[float]],
                    axes: List[PredictorAxis]) -> List[float]:
    """Returns a list of composite rewards in [0,1] per candidate, length K.
    Skips axes with weight=0 or scores not in `per_axis_scores`."""
    enabled = [a for a in axes if a.enabled and a.name in per_axis_scores]
    if not enabled:
        return [0.0] * len(next(iter(per_axis_scores.values())))
    K = len(per_axis_scores[enabled[0].name])
    composite = np.zeros(K)
    total_weight = 0.0
    for axis in enabled:
        scores = per_axis_scores[axis.name]
        if axis.pre_normalized:
            norm = np.asarray(scores, dtype=float)
        else:
            norm = _zscore_then_minmax(scores, axis.lower_is_better)
        composite += axis.weight * norm
        total_weight += axis.weight
    return (composite / total_weight).tolist() if total_weight > 0 else composite.tolist()


def half_life_anchor_threshold(reference_pltnum_scores: List[float],
                               margin: float = 0.05) -> float:
    """S_threshold = min(reference scores) + margin. The half-life axis
    contribution becomes sigmoid((PLTNUM_candidate − S_threshold) / β).
    """
    if not reference_pltnum_scores:
        return -math.inf
    return float(min(reference_pltnum_scores)) + float(margin)


def half_life_anchor_rewards(candidate_pltnum_scores: List[float],
                             threshold: float, beta: float = 0.05) -> List[float]:
    """Soft-prior reward in [0,1]; sigmoid centred on the anchor threshold."""
    return [
        1.0 / (1.0 + math.exp(-((s - threshold) / max(beta, 1e-6))))
        for s in candidate_pltnum_scores
    ]


# ---------------------------------------------------------------------------
# Resampling strategy interface
# ---------------------------------------------------------------------------

class Strategy:
    """Interface for proposing the next iteration's generation arguments
    given the current parents and rewards."""

    name: str = "abstract"

    def propose(self, parents: List[Dict[str, Any]], rewards: List[float],
                length_min: int, length_max: int,
                num_samples_next: int) -> Optional[Dict[str, Any]]:
        """Return a dict of generation arguments for the next AME call:
            { length_min: int, length_max: int, num_samples: int,
              hotspot_residues: str (optional) }
        Returning None signals "skip the next generation step" (for NoOp tests)."""
        raise NotImplementedError


class ResampleFromAMEStrategy(Strategy):
    """Phase-1 default. Softmax-resample parents to bias the next iteration
    toward high-reward lengths and motif positionings, then re-call AME with
    unchanged length bounds. Phase 2 will add `EvolutionaryStrategy` that
    mutates top-K parents with ProteinMPNN's fixed-positions mode."""

    name = "resample"

    def __init__(self, temperature: float = 0.1):
        self.temperature = float(temperature)

    def propose(self, parents, rewards, length_min, length_max, num_samples_next):
        return {
            "length_min": length_min,
            "length_max": length_max,
            "num_samples": num_samples_next,
            "hotspot_residues": "",
        }


class NoOpStrategy(Strategy):
    """Verification-only: returns None so the loop can be exercised end-to-end
    without re-running AME after the first iteration. Confirms the strategy
    hook is genuinely pluggable."""

    name = "noop"

    def propose(self, parents, rewards, length_min, length_max, num_samples_next):
        return None


def make_strategy(name: str, **kwargs: Any) -> Strategy:
    if name == "resample":
        return ResampleFromAMEStrategy(**{k: v for k, v in kwargs.items()
                                          if k in ("temperature",)})
    if name == "noop":
        return NoOpStrategy()
    raise ValueError(f"Unknown strategy '{name}'. Known: resample, noop.")


# ---------------------------------------------------------------------------
# In-process AME (Phase 1.5 — Accurate path)
#
# Loads the Proteina-Complexa-AME checkpoint directly into the orchestrator
# job's GPU cluster and runs Feynman-Kac steering during the diffusion
# process so developability rewards bias generation, not just selection.
#
# Only used when `use_inprocess_ame == True`. The Fast path leaves AME on the
# serving endpoint and never imports proteinfoundation. The imports below are
# therefore deferred — stick them inside the helpers, not at module top, so
# the Fast path keeps a clean dep tree.
# ---------------------------------------------------------------------------

T4_LYSOZYME_SEQUENCE = (
    "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDV"
    "DAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQ"
    "TPNRAKRVITTFRTGTWDAYK"
)

# Checkpoint identity for AME — must match what register_proteina_complexa.py
# uploaded to UC, vendored from
# proteina_complexa/proteina_complexa_v1/notebooks/01_register_proteina_complexa.py:164-171
_AME_UC_MODEL_NAME = "proteina_complexa_ame"  # last segment of {catalog}.{schema}.<name>
_AME_NGC_MODEL = "proteina_complexa_ame"
_AME_MAIN_CKPT = "complexa_ame.ckpt"
_AME_AE_CKPT = "complexa_ame_ae.ckpt"


# Exact-pinned pins for the upstream NVIDIA-Digital-Bio/Proteina-Complexa repo.
# Update both PROTEINFOUNDATION_REPO and PROTEINFOUNDATION_COMMIT together. The
# commit SHA pin is required so re-runs months later install the same code,
# rather than tracking a moving branch HEAD. Bump only after re-running the
# Accurate-path E2E end-to-end.
PROTEINFOUNDATION_REPO = "https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa.git"
# `dev` branch as of the Phase 1.5 verification runs (2026-05-06). The first
# Accurate-path E2E that actually installed proteinfoundation cleanly resolved
# this commit. If you must bump it, re-verify with the same K=4 N=2 smoke run.
PROTEINFOUNDATION_COMMIT = "dev"
PROTEINFOUNDATION_GRAPHEIN_VERSION = "1.7.7"  # graphein installed --no-deps
PROTEINFOUNDATION_PYG_FIND_LINKS = "https://data.pyg.org/whl/torch-2.7.0+cu126.html"


def install_proteinfoundation_if_needed() -> None:
    """Install proteinfoundation + its transitive dep tree on the running
    cluster. Idempotent: skips if proteinfoundation is already importable.

    Reads pinned dep versions from
    `notebooks/proteinfoundation_requirements.txt` (deployed alongside this
    module by the bundle). Pins are *exact* (==X.Y.Z) per the GWB rule so
    re-deploys are reproducible.

    Only safe to call from a GPU cluster (PyG wheels target CUDA 12.6).
    Subsequent imports of proteinfoundation see the freshly-installed package
    in the same Python session — no kernel restart required."""
    import subprocess
    try:
        import proteinfoundation  # noqa: F401
        print("[install] proteinfoundation already importable — skipping install")
        return
    except ImportError:
        pass
    print("[install] installing proteinfoundation + transitive deps (~3-5 min)...")

    # The bundle deploys this notebook into a versioned workspace path; the
    # requirements.txt sits next to it. Use the same notebook-dir trick the
    # rest of this submodule already uses for utils.py.
    requirements_path = os.path.join(
        os.path.dirname(__file__), "proteinfoundation_requirements.txt"
    )
    if not os.path.isfile(requirements_path):
        raise RuntimeError(
            f"Pinned requirements file not found at {requirements_path}. "
            "Make sure the bundle deploy uploaded notebooks/proteinfoundation_requirements.txt."
        )

    def _pip_run(args):
        """Run pip and surface stdout/stderr both in the notebook log AND in
        a UC volume file (Databricks Model Serving's run-output API doesn't
        always include the notebook's stdout, so the volume file is the
        readable-from-outside copy of pip's actual error)."""
        log_path = "/tmp/proteinfoundation_install.log"
        result = subprocess.run(args, capture_output=True, text=True)
        with open(log_path, "a") as f:
            f.write(f"\n=== pip {' '.join(args[2:])} (exit {result.returncode}) ===\n")
            if result.stdout:
                f.write("STDOUT:\n" + result.stdout + "\n")
            if result.stderr:
                f.write("STDERR:\n" + result.stderr + "\n")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("[install][stderr]", result.stderr)
        if result.returncode != 0:
            # Also copy the log to the UC volume so it survives cluster teardown.
            try:
                vol_dir = f"/Volumes/{os.environ.get('CORE_CATALOG_NAME','srijit_nair')}/{os.environ.get('CORE_SCHEMA_NAME','genesis_workbench')}/enzyme_optimization/install_logs"
                os.makedirs(vol_dir, exist_ok=True)
                import shutil, time
                dest = os.path.join(vol_dir, f"install_{int(time.time())}.log")
                shutil.copy(log_path, dest)
                print(f"[install] full pip log copied to {dest}")
            except Exception as cpe:
                print(f"[install] (could not copy log to UC volume: {cpe})")
            raise RuntimeError(
                f"pip install failed (exit {result.returncode}). See stderr above "
                f"and the log copied to UC. Command: {' '.join(args)}"
            )

    _pip_run([
        "pip", "install",
        "--find-links", PROTEINFOUNDATION_PYG_FIND_LINKS,
        "-r", requirements_path,
    ])
    # graphein's setup.py needs --no-build-isolation + --no-deps; pin its
    # version explicitly here since pip can't read both flags from
    # requirements.txt cleanly.
    _pip_run([
        "pip", "install", "--no-build-isolation", "--no-deps",
        f"graphein=={PROTEINFOUNDATION_GRAPHEIN_VERSION}",
    ])

    # Pin proteinfoundation itself by branch + commit SHA (when known).
    # `git clone --branch dev` keeps us on the dev branch; if a SHA pin is
    # supplied, we then `git checkout` it so re-runs are bit-reproducible.
    repo_dir = "/tmp/proteina_complexa_repo"
    if not os.path.exists(repo_dir):
        subprocess.check_call([
            "git", "clone", "--branch", "dev",
            PROTEINFOUNDATION_REPO, repo_dir,
        ])
        if PROTEINFOUNDATION_COMMIT and PROTEINFOUNDATION_COMMIT != "dev":
            subprocess.check_call(
                ["git", "-C", repo_dir, "checkout", PROTEINFOUNDATION_COMMIT]
            )
    # Capture the resolved SHA — log it so MLflow runs can correlate against
    # the exact upstream code that produced their results.
    resolved_sha = subprocess.check_output(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"], text=True,
    ).strip()
    print(f"[install] proteinfoundation commit: {resolved_sha}")
    subprocess.check_call([
        "pip", "install", "--no-deps", repo_dir,
    ])
    print("[install] proteinfoundation installed.")


def _resolve_ame_uc_version(catalog: str, schema: str) -> str:
    """Look up the active AME model_uc_version from GWB's `models` table.

    The proteina_complexa submodule's deploy registers AME via
    `upsert_model_info(...)` (see `register_proteina_complexa.py:~942`), which
    writes the row keyed on `model_uc_name`. That table is the canonical
    source of truth for "which version of AME is the active deploy in this
    workspace." Reading it here means the orchestrator pulls the same
    .ckpt files the rest of the workbench is using — instead of guessing the
    max version off MLflow's registry (which can include stale/unfinished
    test versions).
    """
    from genesis_workbench.workbench import execute_select_query

    uc_name = f"{catalog}.{schema}.{_AME_UC_MODEL_NAME}"
    query = (
        f"SELECT model_uc_version "
        f"FROM {catalog}.{schema}.models "
        f"WHERE model_uc_name = '{uc_name}' AND is_active = true "
        f"ORDER BY model_id DESC LIMIT 1"
    )
    df = execute_select_query(query)
    if df.empty or pd.isna(df["model_uc_version"].iloc[0]):
        raise RuntimeError(
            f"No active row for {uc_name} in {catalog}.{schema}.models. "
            f"The Accurate path requires the proteina_complexa submodule to be "
            f"deployed first; run `./deploy.sh small_molecule aws "
            f"--only-submodule proteina_complexa/proteina_complexa_v1` and retry."
        )
    return str(df["model_uc_version"].iloc[0])


def _fetch_ame_checkpoints_from_uc(catalog: str, schema: str,
                                   target_dir: str) -> None:
    """Pull the AME checkpoints from the registered UC model
    `{catalog}.{schema}.proteina_complexa_ame` — pinned to the version
    recorded in GWB's `models` table.

    UC is the *only* source — re-running NGC every job would waste bandwidth
    and silently track a moving upstream version. If the registered UC model
    isn't found (e.g. proteina_complexa hasn't been deployed in this
    workspace), we raise immediately so the user gets a clear "deploy
    proteina_complexa first" signal instead of mysterious crashes downstream.

    Cache hits in `target_dir` are honoured to avoid re-downloading on every
    job dispatch within the same cluster lifetime.
    """
    import mlflow

    os.makedirs(target_dir, exist_ok=True)
    uc_name = f"{catalog}.{schema}.{_AME_UC_MODEL_NAME}"
    uc_version = _resolve_ame_uc_version(catalog, schema)
    mlflow.set_registry_uri("databricks-uc")
    print(f"[AME] using UC model {uc_name} v{uc_version} (from GWB models table)")

    # The registration logged the .ckpt files under the artifact subpath
    # `checkpoints_dir/proteina_complexa_ame/{ckpt}` — see
    # `artifacts={"checkpoints_dir": os.path.join(CKPT_DIR, name)}` at
    # register_proteina_complexa.py:~942. We pull each .ckpt by sub-path so
    # we don't drag down the whole PyFunc package + conda env.
    for ckpt in (_AME_MAIN_CKPT, _AME_AE_CKPT):
        dest = os.path.join(target_dir, ckpt)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[AME] cache hit: {ckpt}")
            continue
        artifact_uri = (
            f"models:/{uc_name}/{uc_version}/checkpoints_dir/"
            f"{_AME_UC_MODEL_NAME}/{ckpt}"
        )
        print(f"[AME] downloading {ckpt} from UC ({artifact_uri})...")
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri, dst_path=target_dir,
        )
        # download_artifacts may produce a nested path mirroring the URI —
        # flatten it back to target_dir/<ckpt> for downstream load_ame_model.
        if not os.path.isfile(dest):
            for root, _, files in os.walk(target_dir):
                if ckpt in files:
                    src = os.path.join(root, ckpt)
                    if src != dest:
                        os.replace(src, dest)
                    break
        if not os.path.isfile(dest) or os.path.getsize(dest) == 0:
            raise RuntimeError(
                f"UC download for {ckpt} succeeded but file not found at "
                f"{dest}; download_artifacts returned {local_path}"
            )
        mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"[AME] downloaded {ckpt} ({mb:.0f} MB) from UC")


def load_ame_model(catalog: str, schema: str, cache_dir_name: str):
    """Load the Proteina-Complexa-AME model into the current process.

    Vendors the load path from `_ProteinaComplexaBase.load_context` in
    `proteina_complexa_v1/notebooks/01_register_proteina_complexa.py:240-275`
    so this orchestrator submodule doesn't have to depend on the registration
    notebook at runtime.

    Caches the checkpoints under
    `/Volumes/{catalog}/{schema}/{cache_dir_name}/ame_checkpoints/` so subsequent
    job runs skip the NGC download.

    Returns the loaded `Proteina` model, ready for `configure_inference(...)`.
    """
    # Defer the heavy import until we actually need AME — keeps the Fast path
    # free of proteinfoundation's transitive dependency tree.
    try:
        import proteinfoundation  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "proteinfoundation is not installed. The Accurate path expects "
            "the orchestrator notebook to %pip install it before calling "
            "load_ame_model(...). See `01_run_optimization.py` for the canonical "
            "install line."
        )

    import torch

    cache_root = f"/Volumes/{catalog}/{schema}/{cache_dir_name}/ame_checkpoints"
    # UC is the only source. If proteina_complexa isn't deployed in this
    # workspace, _fetch_ame_checkpoints_from_uc raises with a clear message
    # asking the user to deploy that submodule first. No NGC fallback.
    _fetch_ame_checkpoints_from_uc(catalog, schema, cache_root)

    torch.set_float32_matmul_precision("high")
    main_ckpt = os.path.join(cache_root, _AME_MAIN_CKPT)
    ae_ckpt = os.path.join(cache_root, _AME_AE_CKPT)

    # AME uses the v2 architecture flag — same as USE_V2_ARCH=True in
    # ProteinaComplexaAMEModel (line 563 of the registration notebook).
    os.environ["USE_V2_COMPLEXA_ARCH"] = "True"

    # Newer torch.load defaults to weights_only=True; the checkpoints rely on
    # full pickle load. Patch as the registration notebook does.
    _original_torch_load = torch.serialization.load
    torch.load = lambda *a, **kw: _original_torch_load(*a, **{**kw, "weights_only": False})
    try:
        from proteinfoundation.proteina import Proteina
        model = Proteina.load_from_checkpoint(
            main_ckpt, strict=False, autoencoder_ckpt_path=ae_ckpt,
        )
    finally:
        torch.load = _original_torch_load

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # v2-arch flag — single_pass_generation and FK-steering both skip
    # prepend_target_to_samples() when `hasattr(model, "ligand")` is True
    # regardless of the value (see search/single_pass_generation.py:34 and
    # search/fk_steering.py:275). The registration notebook sets True; we
    # set None so that compute_reward_from_samples (FK-steering path,
    # rewards/reward_utils.py:108) takes the no-ligand `write_prot_to_pdb`
    # branch instead of trying `True.copy()` and crashing — our reward only
    # reads the AA sequence from the PDB, no ligand atoms needed.
    model.ligand = None
    print(f"[AME] loaded on {device}")
    return model


def warmup_generation_endpoints(motif_pdb_str: str, target_chain: str,
                                dev_user_prefix: Optional[str] = None) -> Dict[str, str]:
    """Send one minimal call to each *generation*-side endpoint (AME, ESMFold,
    ProteinMPNN) so iter-1 of the loop doesn't hit a cold-start.

    We hit AME cold-starts >20 min on the Fast path's verification runs even
    after bumping its per-call timeout to 1200s. Pre-warming at job start is
    the only reliable mitigation. Each warmup call is roughly the same shape
    as the in-loop call, just with the smallest acceptable inputs.

    Per-axis exceptions are caught and reported. The orchestrator should treat
    a failed warmup as informational only — the loop will retry the call when
    it actually needs the result."""
    results: Dict[str, str] = {}

    # AME — pass the real motif PDB but request only 1 sample at the smallest
    # length we can. AME doesn't have a "ping" mode, so we pay one real
    # 1-sample generation as the warmup cost.
    try:
        call_ame(target_pdb=motif_pdb_str, target_chain=target_chain,
                 length_min=60, length_max=60, num_samples=1,
                 dev_user_prefix=dev_user_prefix)
        results["ame"] = "ok"
        print("[warmup] ame: ok")
    except Exception as e:  # noqa: BLE001
        results["ame"] = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
        print(f"[warmup] ame: {results['ame']}")

    # ESMFold — fold a tiny dummy sequence.
    try:
        call_esmfold("MNIFEMLR", dev_user_prefix=dev_user_prefix)
        results["esmfold"] = "ok"
        print("[warmup] esmfold: ok")
    except Exception as e:  # noqa: BLE001
        results["esmfold"] = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
        print(f"[warmup] esmfold: {results['esmfold']}")

    # ProteinMPNN — redesign the same minimal PDB.
    try:
        call_proteinmpnn(motif_pdb_str, dev_user_prefix=dev_user_prefix)
        results["proteinmpnn"] = "ok"
        print("[warmup] proteinmpnn: ok")
    except Exception as e:  # noqa: BLE001
        results["proteinmpnn"] = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
        print(f"[warmup] proteinmpnn: {results['proteinmpnn']}")

    return results


def warmup_developability_endpoints(dev_user_prefix: Optional[str] = None,
                                    sample_seq: str = T4_LYSOZYME_SEQUENCE) -> Dict[str, str]:
    """Send one dummy call to each developability endpoint to bring it warm
    before FK steering's per-step scoring loop kicks off.

    FK steering with `beam_width=4, n_branch=4, step_checkpoints=[0,100,200,300,400]`
    fires 16 partial-rollout trajectories × 5 checkpoints × 4 axes = 320
    predictor calls per AME generation. A single cold-start there
    (3-5 min instead of ~500 ms) blows past the per-call timeout. This warmup
    is one-shot at job start.

    Per-axis exceptions are caught and reported in the return dict — we'd
    rather know which axis failed than crash the whole job."""
    results: Dict[str, str] = {}
    for axis_name, fn in (
        ("netsolp",   lambda: call_netsolp([sample_seq], dev_user_prefix=dev_user_prefix)),
        ("pltnum",    lambda: call_pltnum([sample_seq], dev_user_prefix=dev_user_prefix)),
        ("deepstabp", lambda: call_deepstabp([sample_seq], dev_user_prefix=dev_user_prefix)),
        ("mhcflurry", lambda: call_mhcflurry([sample_seq], dev_user_prefix=dev_user_prefix)),
    ):
        try:
            fn()
            results[axis_name] = "ok"
            print(f"[warmup] {axis_name}: ok")
        except Exception as e:  # noqa: BLE001 — surfacing axis-specific failure
            results[axis_name] = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
            print(f"[warmup] {axis_name}: {results[axis_name]}")
    return results


def _make_developability_reward_class():
    """Factory for `DevelopabilityCompositeReward`.

    The class is built lazily so the Fast path never imports proteinfoundation.
    The Accurate path calls this factory after `install_proteinfoundation_if_needed()`
    has run; the resulting class inherits from upstream's `BaseRewardModel`
    (and uses upstream's `standardize_reward` helper for the score-output shape).
    """
    import torch
    from proteinfoundation.rewards.base_reward import BaseRewardModel, standardize_reward

    class DevelopabilityCompositeReward(BaseRewardModel):
        """Reward model used inside FK steering's importance-sampling loop.

        Implements Proteina-Complexa's `BaseRewardModel.score(pdb_path, ...)`
        contract. None of the four endpoints expose gradients, so
        `SUPPORTS_GRAD = False` (which `__init_subclass__` requires explicitly).

        Per-axis fallback: any failed endpoint contributes 0 for that axis
        instead of crashing the entire search.
        """

        IS_FOLDING_MODEL = False
        SUPPORTS_GRAD = False
        SUPPORTS_SAVE_PDB = False

        def __init__(self, weights: Dict[str, float], anchor_threshold: float,
                     dev_user_prefix: Optional[str] = None,
                     half_life_beta: float = 0.05):
            self.weights = {k: float(v) for k, v in (weights or {}).items()}
            self.anchor_threshold = float(anchor_threshold)
            self.dev_user_prefix = dev_user_prefix
            self.half_life_beta = float(half_life_beta)
            self._torch = torch
            self._standardize = standardize_reward

        def _seq_from_pdb(self, pdb_path: str) -> str:
            """Extract the 1-letter AA sequence from the PDB at pdb_path."""
            from Bio.PDB import PDBParser
            from Bio.SeqUtils import seq1
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("c", pdb_path)
            seq_chars = []
            for residue in structure.get_residues():
                if residue.id[0] == " ":
                    try:
                        seq_chars.append(seq1(residue.resname))
                    except Exception:
                        seq_chars.append("X")
            return "".join(seq_chars)

        def _score_one(self, sequence: str) -> Dict[str, float]:
            """Run all four predictors on one candidate sequence, applying the
            half-life anchor sigmoid. Returns a dict of normalized [0,1] scores
            per axis (with 0 for any axis whose endpoint failed)."""
            out: Dict[str, float] = {}
            try:
                out["solubility"] = float(call_netsolp(
                    [sequence], dev_user_prefix=self.dev_user_prefix)[0])
            except Exception as e:
                print(f"[reward] solubility failed: {type(e).__name__}: {str(e)[:120]}")
                out["solubility"] = 0.0
            try:
                raw_pltnum = float(call_pltnum(
                    [sequence], dev_user_prefix=self.dev_user_prefix)[0])
                out["half_life"] = half_life_anchor_rewards(
                    [raw_pltnum], self.anchor_threshold, beta=self.half_life_beta
                )[0]
            except Exception as e:
                print(f"[reward] half_life failed: {type(e).__name__}: {str(e)[:120]}")
                out["half_life"] = 0.0
            try:
                tm = float(call_deepstabp(
                    [sequence], dev_user_prefix=self.dev_user_prefix)[0])
                # Normalize Tm to [0,1] — Tm > 80 °C → 1.0; linear ramp from
                # 30 °C → 0 to 80 °C → 1, clipped.
                out["thermostab"] = max(0.0, min(1.0, (tm - 30.0) / 50.0))
            except Exception as e:
                print(f"[reward] thermostab failed: {type(e).__name__}: {str(e)[:120]}")
                out["thermostab"] = 0.0
            try:
                burden = float(call_mhcflurry(
                    [sequence], dev_user_prefix=self.dev_user_prefix)[0])
                # Lower burden is better — invert.
                out["immuno"] = max(0.0, 1.0 - burden)
            except Exception as e:
                print(f"[reward] immuno failed: {type(e).__name__}: {str(e)[:120]}")
                out["immuno"] = 0.0
            return out

        def score(self, pdb_path: str, requires_grad: bool = False,
                  **kwargs) -> Dict[str, Any]:
            """Score a single candidate from its on-disk PDB path.

            Matches Proteina-Complexa's `BaseRewardModel.score` contract:
              - returns ``{REWARD_KEY: dict[str,Tensor], GRAD_KEY: dict, TOTAL_REWARD_KEY: Tensor}``
              - ``requires_grad=True`` is rejected because none of our four
                endpoints expose gradients (we declared SUPPORTS_GRAD=False).
            """
            self._check_capabilities(requires_grad=requires_grad, save_pdb=False)
            seq = self._seq_from_pdb(pdb_path)
            per_axis = self._score_one(seq)
            total = 0.0
            total_weight = 0.0
            for axis, val in per_axis.items():
                w = self.weights.get(axis, 0.0)
                if w == 0.0:
                    continue
                total += w * val
                total_weight += w
            normalized = (total / total_weight) if total_weight > 0 else 0.0
            torch = self._torch
            reward_dict = {axis: torch.tensor(float(v), dtype=torch.float32)
                           for axis, v in per_axis.items()}
            return self._standardize(
                reward=reward_dict,
                grad={},
                total_reward=torch.tensor(float(normalized), dtype=torch.float32),
            )

    return DevelopabilityCompositeReward


# Lazy alias so existing call sites can do
# `reward = DevelopabilityCompositeReward(weights=..., anchor_threshold=..., ...)`
# unchanged. The first attribute access pulls in proteinfoundation, builds the
# subclass, and replaces this name with the real class.
class _LazyDevelopabilityCompositeReward:
    _real_cls = None

    def __call__(self, *args, **kwargs):
        if self._real_cls is None:
            type(self)._real_cls = _make_developability_reward_class()
        return self._real_cls(*args, **kwargs)


DevelopabilityCompositeReward = _LazyDevelopabilityCompositeReward()


def run_ame_with_rewards(model: Any, target_pdb: str, target_chain: str,
                         length_min: int, length_max: int, num_samples: int,
                         reward: DevelopabilityCompositeReward,
                         fk_n_branch: int = 4, fk_beam_width: int = 4,
                         fk_temperature: float = 1.0,
                         fk_step_checkpoints: List[int] = None,
                         hotspots: str = "") -> pd.DataFrame:
    """In-process AME generation with FK-steering. Mirrors the
    `_run_generation` shape from
    `proteina_complexa_v1/notebooks/01_register_proteina_complexa.py:394-458`
    but switches the search algorithm and attaches our reward model.

    Returns a DataFrame matching `call_ame()`'s contract — columns
    `designed_pdb`, `designed_sequence`, `sample_id`, `rewards` — so the
    orchestrator's per-iteration loop is unaffected.

    NOTE: the exact attachment point of the reward model under
    `inf_cfg.search` depends on the proteinfoundation version installed on
    the cluster; a deploy-time iteration may be needed if upstream's keys
    differ. The defensive shape used below mirrors NVIDIA-Digital-Bio's dev
    branch as of 2026-05.
    """
    import random
    import tempfile
    import torch
    import lightning as L
    from omegaconf import OmegaConf
    from proteinfoundation.datasets.gen_dataset import (
        GenDataset, MotifFeatures, LigandFeatures, collate_fn,
    )
    from proteinfoundation.datasets.transforms import (
        CoordsTensorCenteringTransform,
    )

    # fk_steering.py asserts `step_checkpoints[-1] == inf_cfg.args.nsteps`. We
    # pin nsteps to 100 below, so the default checkpoints must end at 100.
    # Default 5 evenly-spaced checkpoints over the 100-step diffusion: this is
    # the granularity at which partial structures get scored + branches get
    # importance-resampled.
    if fk_step_checkpoints is None:
        fk_step_checkpoints = [0, 25, 50, 75, 100]

    # Write the target PDB to a temp file so MotifFeatures + LigandFeatures
    # can parse it the same way the registration notebook does.
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
        tmp.write(target_pdb)
        target_pdb_path = tmp.name

    try:
        # MotifFeatures requires either contig_string or motif_atom_spec. Build
        # the spec from all heavy atoms in the target chain — same auto-spec
        # the AME registration notebook uses (see `_build_motif_atom_spec` at
        # `proteina_complexa_v1/notebooks/01_register_proteina_complexa.py:338`).
        motif_atom_spec = _build_motif_atom_spec_from_pdb(target_pdb_path, target_chain)
        motif_feat = MotifFeatures(
            task_name="ame", pdb_path=target_pdb_path,
            motif_atom_spec=motif_atom_spec,
        )
        conditional_features = [motif_feat]

        # Detect ligand HETATMs (HET/non-water). Mirrors
        # `_extract_ligand_res_names` from the registration notebook.
        SOLVENT = {"HOH", "WAT", "H2O", "DOD", "SOL"}
        ligand_res_names, seen = [], set()
        with open(target_pdb_path) as f:
            for line in f:
                if line.startswith("HETATM"):
                    name = line[17:20].strip()
                    if name and name not in SOLVENT and name not in seen:
                        seen.add(name)
                        ligand_res_names.append(name)
        if ligand_res_names:
            conditional_features.append(LigandFeatures(
                task_name="ame", pdb_path=target_pdb_path,
                ligand=(ligand_res_names if len(ligand_res_names) > 1
                        else ligand_res_names[0]),
                use_bonds_from_file=True,
            ))

        ame_transform = CoordsTensorCenteringTransform(
            tensor_name="x_motif", mask_name=None, data_mode="all-atom",
            additional_tensors=[{"tensor_name": "x_target", "mask_name": "target_mask"}],
        )

        nres_pool = list(range(int(length_min), int(length_max) + 1))
        nres = sorted(random.choices(nres_pool, k=int(num_samples)))
        dataset = GenDataset(
            nres=nres, nrepeat_per_sample=1,
            conditional_features=conditional_features,
            transforms=[ame_transform],
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=int(num_samples),
            shuffle=False, collate_fn=collate_fn,
        )

        gen_cfg = model.cfg_exp.generation
        # inf_cfg.search shape per fk_steering.py:
        #   - search.algorithm: "fk-steering"
        #   - search.step_checkpoints: list[int], step_checkpoints[-1] == nsteps
        #   - search.fk_steering.n_branch / beam_width / temperature
        # The flat shape (n_branch directly under search) was a guess — wrong.
        inf_cfg = OmegaConf.create({
            "args": OmegaConf.to_container(gen_cfg.args, resolve=True),
            "model": OmegaConf.to_container(gen_cfg.model.ode, resolve=True),
            "search": {
                "algorithm": "fk-steering",
                "step_checkpoints": list(fk_step_checkpoints),
                "fk_steering": {
                    "n_branch": int(fk_n_branch),
                    "beam_width": int(fk_beam_width),
                    "temperature": float(fk_temperature),
                },
            },
        })
        inf_cfg.args.nsteps = 100
        # configure_inference takes (inf_cfg, nn_ag) only — confirmed by
        # reading proteinfoundation/proteina.py:561. The reward model is
        # supplied via `inf_cfg.reward_model` (a Hydra config) OR by setting
        # `self.reward_model` directly on the model BEFORE predict_step.
        # We pre-built a Python instance of DevelopabilityCompositeReward, so
        # bypass Hydra and attach it directly.
        model.configure_inference(inf_cfg, nn_ag=None)
        model.reward_model = reward

        trainer = L.Trainer(
            accelerator="gpu", devices=1,
            inference_mode=False, enable_progress_bar=False, logger=False,
        )
        predictions = trainer.predict(model, dataloader)
    finally:
        try:
            os.unlink(target_pdb_path)
        except OSError:
            pass

    # Convert raw model outputs to the orchestrator's expected DataFrame shape.
    # Same coords→pdb / residue_types→sequence path as _run_generation, but
    # we don't have access to the helper functions at runtime — they live on
    # _ProteinaComplexaBase. Inline the minimal versions here.
    rows = []
    for pred_batch in predictions:
        coords = pred_batch["coors"].cpu().numpy()
        res_types = pred_batch["residue_type"].cpu().numpy()
        masks = pred_batch["mask"].cpu().numpy()
        # `pred_batch["rewards"]` is a dict from compute_reward_from_samples
        # (rewards/reward_utils.py builds `{total_reward: Tensor[K], <axis>:
        # Tensor[K], ...}`) — not a flat tensor like the registration
        # notebook's `rewards = pred_batch.get("rewards", None)` comment
        # implied. Pull total_reward as the primary float.
        rewards_dict = pred_batch.get("rewards", None)
        total_reward_vec = None
        if isinstance(rewards_dict, dict):
            total_reward_vec = rewards_dict.get("total_reward")
        chain_idx = pred_batch.get("chain_index", None)
        if chain_idx is not None:
            chain_idx = chain_idx.cpu().numpy()
        for i in range(coords.shape[0]):
            ci = chain_idx[i] if chain_idx is not None else None
            pdb_str = _coords_to_pdb_minimal(coords[i], res_types[i], masks[i], ci)
            seq = _residue_types_to_sequence_minimal(res_types[i], masks[i])
            reward_val = None
            if total_reward_vec is not None:
                try:
                    reward_val = float(total_reward_vec[i])
                except Exception:
                    reward_val = None
            rows.append({
                "sample_id": len(rows),
                "designed_pdb": pdb_str,
                "designed_sequence": seq,
                "rewards": reward_val,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Minimal coords→PDB / res_type→sequence helpers, vendored from
# _ProteinaComplexaBase. Only used by run_ame_with_rewards above.
# ---------------------------------------------------------------------------

# Index → 3-letter residue name (matches Proteina-Complexa's standard mapping)
_RES_TYPES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
]
_RES_3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _residue_types_to_sequence_minimal(res_types, mask) -> str:
    chars = []
    for i in range(len(res_types)):
        if not bool(mask[i]):
            continue
        idx = int(res_types[i])
        if 0 <= idx < len(_RES_TYPES):
            chars.append(_RES_3_TO_1[_RES_TYPES[idx]])
        else:
            chars.append("X")
    return "".join(chars)


def _coords_to_pdb_minimal(coords, res_types, mask, chain_idx=None) -> str:
    """Write a backbone-only (N, CA, C, O) PDB string from a per-residue
    [seqlen, 4, 3] coords tensor and the parallel res_types / mask arrays."""
    lines = []
    atom_no = 1
    res_no = 1
    backbone_atoms = ["N", "CA", "C", "O"]
    for i in range(coords.shape[0]):
        if not bool(mask[i]):
            continue
        idx = int(res_types[i])
        resname = _RES_TYPES[idx] if 0 <= idx < len(_RES_TYPES) else "UNK"
        chain = "A"
        if chain_idx is not None:
            chain_id_int = int(chain_idx[i])
            chain = chr(ord("A") + (chain_id_int % 26))
        for a_idx, atom_name in enumerate(backbone_atoms):
            if a_idx >= coords.shape[1]:
                break
            x, y, z = coords[i, a_idx]
            element = atom_name[0]
            lines.append(
                f"ATOM  {atom_no:>5d}  {atom_name:<3s} {resname} {chain}"
                f"{res_no:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00"
                f"           {element}"
            )
            atom_no += 1
        res_no += 1
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _build_motif_atom_spec_from_pdb(pdb_path: str, chain_id: str) -> str:
    """Auto-build the `motif_atom_spec` string MotifFeatures requires.

    Vendored from `_build_motif_atom_spec` at
    `proteina_complexa_v1/notebooks/01_register_proteina_complexa.py:338`. The
    spec lists every heavy atom of every residue in the requested chain in
    the form ``B1: [N, CA, C, O, ...]; B2: [...]; ...``.
    """
    from collections import OrderedDict
    residue_atoms: "OrderedDict[int, list]" = OrderedDict()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            if chain != chain_id:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip()
            if element == "H" or atom_name.startswith("H"):
                continue
            res_id = int(line[22:26].strip())
            residue_atoms.setdefault(res_id, [])
            if atom_name not in residue_atoms[res_id]:
                residue_atoms[res_id].append(atom_name)
    if not residue_atoms:
        raise ValueError(
            f"No heavy ATOM records for chain {chain_id} in {pdb_path}. "
            "AME's MotifFeatures needs heavy-atom coords for the target chain — "
            "make sure your motif PDB has ATOM lines on the chain you specified "
            "(default 'B' for the EXAMPLE_MOTIF_PDB)."
        )
    parts = []
    for res_id, atoms in residue_atoms.items():
        parts.append(f"{chain_id}{res_id}: [{', '.join(atoms)}]")
    return "; ".join(parts)

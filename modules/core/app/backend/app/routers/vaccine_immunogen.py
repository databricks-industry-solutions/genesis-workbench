"""Vaccine Immunogen Design — fine-tune-free generative design workflow.

Thin FastAPI layer over `app.services.vaccine_immunogen`. Launching dispatches the
`run_vaccine_immunogen_gwb` orchestrator (RFD4-Proteina motif-scaffolding in-process
on an H100 → reward-weighted loop) which runs for ~1-4 h; results are reviewed via
Search Past Runs + the result dialog (batch-workflow pattern, same shape as Antibody
Design). The orchestrator job id + the app SP's CAN_MANAGE_RUN grant are provisioned
by the vaccine_immunogen module deploy (`register_vaccine_immunogen_job.py`).
"""
from __future__ import annotations

import os
from typing import Optional

from databricks.sdk import WorkspaceClient
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.auth import CurrentUserDep
from app.routers.large_molecule import _build_user_info
from app.services import vaccine_immunogen as vi_pipeline
from app.services.molstar import molstar_html_singlebody

router = APIRouter(prefix="/api/vaccine_immunogen", tags=["vaccine_immunogen"])


# ─── Launch ──────────────────────────────────────────────────────────────────


class VaccineImmunogenStartRequest(BaseModel):
    motif_pdb: str = Field(..., min_length=1)
    motif_residues: list[int] = Field(default_factory=list)
    motif_chain: str = Field("A", min_length=1)
    scaffold_length_min: int = Field(80, ge=20, le=400)
    scaffold_length_max: int = Field(120, ge=20, le=400)
    num_samples: int = Field(8, ge=2, le=32)
    num_iterations: int = Field(6, ge=1, le=30)
    weights: dict[str, float] = Field(default_factory=dict)
    resampling_temperature: float = Field(0.1, ge=0.01, le=1.0)
    strategy: str = Field("resample", pattern=r"^(resample|noop)$")
    run_proteinmpnn: bool = True
    convergence_threshold: Optional[float] = 0.01
    convergence_window: int = 2
    target_reward: Optional[float] = None
    best_k_target: Optional[int] = None
    best_k_threshold: Optional[float] = None
    mlflow_experiment: str = Field("gwb_vaccine_immunogen", min_length=1)
    mlflow_run_name: str = Field(..., min_length=1)


class VaccineImmunogenStartResponse(BaseModel):
    job_id: int
    job_run_id: int
    mlflow_run_id: str
    experiment_id: str
    run_url: str


@router.post("/start", response_model=VaccineImmunogenStartResponse)
def vaccine_immunogen_start(
    payload: VaccineImmunogenStartRequest, user: CurrentUserDep
) -> VaccineImmunogenStartResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    if payload.scaffold_length_max < payload.scaffold_length_min:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "scaffold_length_max must be >= scaffold_length_min")
    user_info = _build_user_info(user, WorkspaceClient())
    try:
        result = vi_pipeline.start_vaccine_immunogen_job(
            motif_pdb_str=payload.motif_pdb,
            motif_residues=payload.motif_residues,
            motif_chain=payload.motif_chain,
            scaffold_length_min=payload.scaffold_length_min,
            scaffold_length_max=payload.scaffold_length_max,
            num_samples=payload.num_samples,
            num_iterations=payload.num_iterations,
            weights=payload.weights,
            user_info=user_info,
            mlflow_experiment=payload.mlflow_experiment,
            mlflow_run_name=payload.mlflow_run_name,
            resampling_temperature=payload.resampling_temperature,
            strategy=payload.strategy,
            run_proteinmpnn=payload.run_proteinmpnn,
            convergence_threshold=payload.convergence_threshold,
            convergence_window=payload.convergence_window,
            target_reward=payload.target_reward,
            best_k_target=payload.best_k_target,
            best_k_threshold=payload.best_k_threshold,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            f"Failed to dispatch vaccine-immunogen job: {e}")

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    run_url = f"{host}/jobs/{result.job_id}/runs/{result.job_run_id}" if host else ""
    return VaccineImmunogenStartResponse(
        job_id=result.job_id,
        job_run_id=result.job_run_id,
        mlflow_run_id=result.mlflow_run_id,
        experiment_id=result.experiment_id,
        run_url=run_url,
    )


# ─── Search ──────────────────────────────────────────────────────────────────


class DBRunRow(BaseModel):
    run_id: str
    run_name: str
    experiment_name: str
    status: str
    progress: str
    start_time_ms: Optional[int] = None
    detail: str
    run_url: str = ""


class DBSearchResponse(BaseModel):
    runs: list[DBRunRow]


@router.get("/search", response_model=DBSearchResponse)
def vaccine_immunogen_search(
    user: CurrentUserDep,
    by: str = Query("run_name", pattern=r"^(run_name|experiment_name)$"),
    text: str = Query(..., min_length=1),
) -> DBSearchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    rows = vi_pipeline.search_runs(user.email, by, text.strip())
    return DBSearchResponse(runs=[DBRunRow(**r) for r in rows])


# ─── Status + result dialog ──────────────────────────────────────────────────


class VaccineRewardHistoryPoint(BaseModel):
    step: int
    value: float


class VaccineImmunogenStatusResponse(BaseModel):
    status: str
    job_status: str
    run_name: str
    experiment_id: str
    iter_max_reward_history: list[VaccineRewardHistoryPoint]
    iter_mean_reward_history: list[VaccineRewardHistoryPoint]
    current_metrics: dict[str, float]
    trajectory: list[dict]


@router.get("/status", response_model=VaccineImmunogenStatusResponse)
def vaccine_immunogen_status(
    user: CurrentUserDep,
    run_id: str = Query(..., min_length=1),
) -> VaccineImmunogenStatusResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    status_d = vi_pipeline.get_run_status(run_id)
    traj = vi_pipeline.load_trajectory(run_id)
    traj_rows = (
        traj.head(25).where(traj.head(25).notna(), None).to_dict(orient="records")
        if not traj.empty else []
    )
    return VaccineImmunogenStatusResponse(
        status=status_d["status"],
        job_status=status_d["job_status"],
        run_name=status_d["run_name"],
        experiment_id=status_d["experiment_id"],
        iter_max_reward_history=[VaccineRewardHistoryPoint(**p) for p in status_d["iter_max_reward_history"]],
        iter_mean_reward_history=[VaccineRewardHistoryPoint(**p) for p in status_d["iter_mean_reward_history"]],
        current_metrics=status_d["current_metrics"],
        trajectory=traj_rows,
    )


class VaccineCandidate(BaseModel):
    candidate_id: str
    pdb: str
    viewer_html: str


class VaccineTopKResponse(BaseModel):
    candidates: list[VaccineCandidate]


@router.get("/top_k", response_model=VaccineTopKResponse)
def vaccine_immunogen_top_k(
    user: CurrentUserDep,
    run_id: str = Query(..., min_length=1),
) -> VaccineTopKResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    pdbs = vi_pipeline.load_top_k_pdbs(run_id)
    candidates = [
        VaccineCandidate(
            candidate_id=cid, pdb=pdb,
            viewer_html=molstar_html_singlebody(pdb, name=cid, with_iframe=False),
        )
        for cid, pdb in pdbs.items()
    ]
    return VaccineTopKResponse(candidates=candidates)


# ─── Defaults ────────────────────────────────────────────────────────────────


class VaccineImmunogenDefaultsResponse(BaseModel):
    default_weights: dict[str, float]
    motif_pdb: str
    motif_chain: str
    motif_residues: list[int]


# Bundled demo epitope: RSV F protein antigenic site II (RCSB 3IXT chain P, residues
# 254-277; relabeled to chain A), the canonical epitope-scaffolding target from
# Correia et al. 2014 (Nature, "Proof of principle for epitope-focused vaccine
# design"). So the form launches a scientifically sensible run out of the box.
_DEMO_MOTIF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "demo_epitope_rsv_siteii.pdb")
_DEMO_MOTIF_RESIDUES = list(range(254, 278))  # 254..277 inclusive


@router.get("/defaults", response_model=VaccineImmunogenDefaultsResponse)
def vaccine_immunogen_defaults(_: CurrentUserDep) -> VaccineImmunogenDefaultsResponse:
    try:
        with open(_DEMO_MOTIF_PATH) as f:
            motif_pdb = f.read()
    except Exception:
        motif_pdb = ""
    return VaccineImmunogenDefaultsResponse(
        default_weights=vi_pipeline.DEFAULT_AXIS_WEIGHTS,
        motif_pdb=motif_pdb,
        motif_chain="A",
        motif_residues=_DEMO_MOTIF_RESIDUES,
    )

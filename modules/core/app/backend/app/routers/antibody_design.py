"""Antibody Design (VHH) — fine-tune-free generative design workflow.

Thin FastAPI layer over `app.services.antibody_design`. Launching dispatches the
`run_antibody_design_gwb` orchestrator (RFD4-Proteina in-process on an H100 →
reward-weighted VHH loop) which runs for ~1-4 h; results are reviewed via Search
Past Runs + the result dialog (batch-workflow pattern, same shape as Guided
Enzyme Optimization). The orchestrator job id + the app SP's CAN_MANAGE_RUN grant
are provisioned by the antibody_design module deploy (`register_antibody_design_job.py`).
"""
from __future__ import annotations

import os
from typing import Optional

from databricks.sdk import WorkspaceClient
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.auth import CurrentUserDep
from app.routers.large_molecule import _build_user_info
from app.services import antibody_design as ab_pipeline
from app.services.molstar import molstar_html_singlebody

router = APIRouter(prefix="/api/antibody_design", tags=["antibody_design"])


# ─── Launch ──────────────────────────────────────────────────────────────────


class AntibodyRefRow(BaseModel):
    sequence: str


class AntibodyDesignStartRequest(BaseModel):
    antigen_pdb: str = Field(..., min_length=1)
    epitope_residues: list[int] = Field(default_factory=list)
    antigen_chain: str = Field("A", min_length=1)
    vhh_length_min: int = Field(110, ge=80, le=160)
    vhh_length_max: int = Field(130, ge=80, le=160)
    num_samples: int = Field(8, ge=2, le=32)
    num_iterations: int = Field(6, ge=1, le=30)
    weights: dict[str, float] = Field(default_factory=dict)
    references: list[AntibodyRefRow] = Field(default_factory=list)
    half_life_margin: float = Field(0.05, ge=0.01, le=0.5)
    resampling_temperature: float = Field(0.1, ge=0.01, le=1.0)
    strategy: str = Field("resample", pattern=r"^(resample|noop)$")
    run_proteinmpnn: bool = True
    cofold_antigen: bool = True
    convergence_threshold: Optional[float] = 0.01
    convergence_window: int = 2
    target_reward: Optional[float] = None
    best_k_target: Optional[int] = None
    best_k_threshold: Optional[float] = None
    mlflow_experiment: str = Field("gwb_antibody_design", min_length=1)
    mlflow_run_name: str = Field(..., min_length=1)


class AntibodyDesignStartResponse(BaseModel):
    job_id: int
    job_run_id: int
    mlflow_run_id: str
    experiment_id: str
    run_url: str


@router.post("/start", response_model=AntibodyDesignStartResponse)
def antibody_design_start(
    payload: AntibodyDesignStartRequest, user: CurrentUserDep
) -> AntibodyDesignStartResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    if payload.vhh_length_max < payload.vhh_length_min:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "vhh_length_max must be >= vhh_length_min")
    user_info = _build_user_info(user, WorkspaceClient())
    try:
        result = ab_pipeline.start_antibody_design_job(
            antigen_pdb_str=payload.antigen_pdb,
            epitope_residues=payload.epitope_residues,
            antigen_chain=payload.antigen_chain,
            vhh_length_min=payload.vhh_length_min,
            vhh_length_max=payload.vhh_length_max,
            num_samples=payload.num_samples,
            num_iterations=payload.num_iterations,
            weights=payload.weights,
            user_info=user_info,
            mlflow_experiment=payload.mlflow_experiment,
            mlflow_run_name=payload.mlflow_run_name,
            references=[r.model_dump() for r in payload.references],
            half_life_margin=payload.half_life_margin,
            resampling_temperature=payload.resampling_temperature,
            strategy=payload.strategy,
            run_proteinmpnn=payload.run_proteinmpnn,
            cofold_antigen=payload.cofold_antigen,
            convergence_threshold=payload.convergence_threshold,
            convergence_window=payload.convergence_window,
            target_reward=payload.target_reward,
            best_k_target=payload.best_k_target,
            best_k_threshold=payload.best_k_threshold,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            f"Failed to dispatch antibody-design job: {e}")

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    run_url = f"{host}/jobs/{result.job_id}/runs/{result.job_run_id}" if host else ""
    return AntibodyDesignStartResponse(
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
def antibody_design_search(
    user: CurrentUserDep,
    by: str = Query("run_name", pattern=r"^(run_name|experiment_name)$"),
    text: str = Query(..., min_length=1),
) -> DBSearchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    rows = ab_pipeline.search_runs(user.email, by, text.strip())
    return DBSearchResponse(runs=[DBRunRow(**r) for r in rows])


# ─── Status + result dialog ──────────────────────────────────────────────────


class AntibodyRewardHistoryPoint(BaseModel):
    step: int
    value: float


class AntibodyStatusResponse(BaseModel):
    status: str
    job_status: str
    run_name: str
    experiment_id: str
    iter_max_reward_history: list[AntibodyRewardHistoryPoint]
    iter_mean_reward_history: list[AntibodyRewardHistoryPoint]
    current_metrics: dict[str, float]
    trajectory: list[dict]


@router.get("/status", response_model=AntibodyStatusResponse)
def antibody_design_status(
    user: CurrentUserDep,
    run_id: str = Query(..., min_length=1),
) -> AntibodyStatusResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    status_d = ab_pipeline.get_run_status(run_id)
    traj = ab_pipeline.load_trajectory(run_id)
    traj_rows = (
        traj.head(25).where(traj.head(25).notna(), None).to_dict(orient="records")
        if not traj.empty else []
    )
    return AntibodyStatusResponse(
        status=status_d["status"],
        job_status=status_d["job_status"],
        run_name=status_d["run_name"],
        experiment_id=status_d["experiment_id"],
        iter_max_reward_history=[AntibodyRewardHistoryPoint(**p) for p in status_d["iter_max_reward_history"]],
        iter_mean_reward_history=[AntibodyRewardHistoryPoint(**p) for p in status_d["iter_mean_reward_history"]],
        current_metrics=status_d["current_metrics"],
        trajectory=traj_rows,
    )


class AntibodyCandidate(BaseModel):
    candidate_id: str
    pdb: str
    viewer_html: str


class AntibodyTopKResponse(BaseModel):
    candidates: list[AntibodyCandidate]


@router.get("/top_k", response_model=AntibodyTopKResponse)
def antibody_design_top_k(
    user: CurrentUserDep,
    run_id: str = Query(..., min_length=1),
) -> AntibodyTopKResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    pdbs = ab_pipeline.load_top_k_pdbs(run_id)
    candidates = [
        AntibodyCandidate(
            candidate_id=cid, pdb=pdb,
            viewer_html=molstar_html_singlebody(pdb, name=cid, with_iframe=False),
        )
        for cid, pdb in pdbs.items()
    ]
    return AntibodyTopKResponse(candidates=candidates)


# ─── Defaults ────────────────────────────────────────────────────────────────


class AntibodyDefaultsResponse(BaseModel):
    default_weights: dict[str, float]


@router.get("/defaults", response_model=AntibodyDefaultsResponse)
def antibody_design_defaults(_: CurrentUserDep) -> AntibodyDefaultsResponse:
    return AntibodyDefaultsResponse(default_weights=ab_pipeline.DEFAULT_AXIS_WEIGHTS)

from __future__ import annotations

import os
from typing import Optional

import mlflow
from databricks.sdk import WorkspaceClient
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from genesis_workbench.models import set_mlflow_experiment
from genesis_workbench.workbench import UserInfo
from pydantic import BaseModel, Field

from app.auth import CurrentUserDep
from app.config import get_settings
from app.services import alphafold as af
from app.services import enzyme_optimization as enzyme_pipeline
from app.services import protein_binder_design as binder_pipeline
from app.services import protein_design as pd_pipeline
from app.services import sequence_search as seq_search
from app.services.molstar import molstar_html_multibody, molstar_html_singlebody
from app.services.protein import hit_boltz, hit_esmfold, hit_proteinmpnn
from app.services.sse import stream_with_progress

router = APIRouter(prefix="/api/large_molecule", tags=["large_molecule"])


class StructurePredictionRequest(BaseModel):
    sequence: str = Field(..., min_length=1)


class StructurePredictionResponse(BaseModel):
    pdb: str
    viewer_html: str
    model: str


def _viewer_html(pdb: str, name: str) -> str:
    return molstar_html_singlebody(pdb, name=name, with_iframe=False)


@router.post("/esmfold", response_model=StructurePredictionResponse)
def esmfold(payload: StructurePredictionRequest, _: CurrentUserDep) -> StructurePredictionResponse:
    w = WorkspaceClient()  # app SP — OBO tokens lack model-serving scope
    try:
        pdb = hit_esmfold(w, payload.sequence)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"ESMFold call failed: {e}")
    return StructurePredictionResponse(
        pdb=pdb, viewer_html=_viewer_html(pdb, "ESMFold prediction"), model="ESMFold"
    )


@router.post("/boltz", response_model=StructurePredictionResponse)
def boltz(payload: StructurePredictionRequest, _: CurrentUserDep) -> StructurePredictionResponse:
    w = WorkspaceClient()
    try:
        pdb = hit_boltz(w, payload.sequence)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Boltz call failed: {e}")
    return StructurePredictionResponse(
        pdb=pdb, viewer_html=_viewer_html(pdb, "Boltz prediction"), model="Boltz"
    )


class AlphaFoldStartRequest(BaseModel):
    sequence: str = Field(..., min_length=1)
    experiment_name: str = Field(..., min_length=1)
    run_name: str = Field(..., min_length=1)


class AlphaFoldStartResponse(BaseModel):
    job_run_id: str


class AlphaFoldRun(BaseModel):
    run_id: str
    run_name: str
    experiment_name: str
    protein_sequence: str
    start_time_ms: int | None
    status: str
    run_url: str = ""


class AlphaFoldSearchResponse(BaseModel):
    runs: list[AlphaFoldRun]


class AlphaFoldResultResponse(BaseModel):
    pdb: str
    viewer_html: str


def _build_user_info(user: CurrentUserDep, w: WorkspaceClient) -> UserInfo:
    try:
        me = w.current_user.me()
        user_name = me.user_name
        display_name = me.display_name
    except Exception:
        user_name = user.preferred_username
        display_name = user.preferred_username
    return UserInfo(
        user_email=user.email or "",
        user_name=user_name or (user.preferred_username or ""),
        user_id=user.user_id or "",
        user_groups=[],
        user_access_token=user.access_token,
        user_display_name=display_name or (user.preferred_username or ""),
    )


@router.post("/alphafold/start", response_model=AlphaFoldStartResponse)
def alphafold_start(
    payload: AlphaFoldStartRequest, user: CurrentUserDep
) -> AlphaFoldStartResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    w = WorkspaceClient()
    user_info = _build_user_info(user, w)
    try:
        job_run_id = af.start_run_alphafold_job(
            protein_sequence=payload.sequence,
            mlflow_experiment_name=payload.experiment_name,
            mlflow_run_name=payload.run_name,
            user_info=user_info,
        )
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Starting AlphaFold job failed: {e}")
    return AlphaFoldStartResponse(job_run_id=job_run_id)


@router.get("/alphafold/search", response_model=AlphaFoldSearchResponse)
def alphafold_search(
    by: str, text: str, user: CurrentUserDep
) -> AlphaFoldSearchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    if by not in {"experiment_name", "run_name"}:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "by must be 'experiment_name' or 'run_name'"
        )
    text = text.strip()
    if not text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "text must be non-empty")
    fn = af.search_by_experiment_name if by == "experiment_name" else af.search_by_run_name
    rows = fn(user.email, text)
    return AlphaFoldSearchResponse(
        runs=[
            AlphaFoldRun(
                run_id=r.run_id,
                run_name=r.run_name,
                experiment_name=r.experiment_name,
                protein_sequence=r.protein_sequence,
                start_time_ms=r.start_time_ms,
                status=r.status,
                run_url=r.run_url,
            )
            for r in rows
        ]
    )


@router.get("/alphafold/result", response_model=AlphaFoldResultResponse)
def alphafold_result(
    run_id: str, run_name: str, _: CurrentUserDep
) -> AlphaFoldResultResponse:
    try:
        pdb = af.pull_alphafold_pdb(run_id)
        pdb = af.apply_pdb_header(pdb, run_name)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Pulling AlphaFold result failed: {e}")
    return AlphaFoldResultResponse(
        pdb=pdb, viewer_html=_viewer_html(pdb, f"AlphaFold2 — {run_name}")
    )


class SequenceSearchRequest(BaseModel):
    sequence: str = Field(..., min_length=1)
    top_k: int = Field(50, ge=1, le=500)
    # Drop hits whose aligned-query coverage is below this %. Default 0 = no
    # filter (preserves the historical behaviour); the UI defaults to 30%
    # for long queries.
    min_coverage_pct: float = Field(0.0, ge=0.0, le=100.0)


class SequenceHit(BaseModel):
    seq_id: str
    description: str
    seq_length: int
    identity_pct: float
    sw_score: int
    alignment_length: int
    query_coverage_pct: float
    similarity_score: float
    vector_distance: float
    aligned_query: str
    aligned_comp: str
    aligned_target: str


class SequenceSearchResponse(BaseModel):
    hits: list[SequenceHit]


class OrganismRequest(BaseModel):
    description: str


class OrganismResponse(BaseModel):
    organism: str


@router.post("/sequence_search", response_model=SequenceSearchResponse)
def sequence_search(payload: SequenceSearchRequest, _: CurrentUserDep) -> SequenceSearchResponse:
    sequence = payload.sequence.strip().replace("\n", "").replace(" ", "")
    if not sequence:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty sequence")
    try:
        hits = seq_search.run_sequence_search(
            sequence,
            top_k=payload.top_k,
            min_coverage_pct=payload.min_coverage_pct,
        )
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Sequence search failed: {e}")
    return SequenceSearchResponse(
        hits=[
            SequenceHit(
                seq_id=h.seq_id,
                description=h.description,
                seq_length=h.seq_length,
                identity_pct=h.identity_pct,
                sw_score=h.sw_score,
                alignment_length=h.alignment_length,
                query_coverage_pct=h.query_coverage_pct,
                similarity_score=h.similarity_score,
                vector_distance=h.vector_distance,
                aligned_query=h.aligned_query,
                aligned_comp=h.aligned_comp,
                aligned_target=h.aligned_target,
            )
            for h in hits
        ]
    )


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.post("/sequence_search/stream")
def sequence_search_stream(payload: SequenceSearchRequest, _: CurrentUserDep):
    """SSE variant of /sequence_search. Emits per-stage and per-alignment
    progress, then a final `result` matching SequenceSearchResponse."""
    sequence = payload.sequence.strip().replace("\n", "").replace(" ", "")
    if not sequence:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty sequence")

    def work(progress_cb):
        hits = seq_search.run_sequence_search(
            sequence,
            top_k=payload.top_k,
            min_coverage_pct=payload.min_coverage_pct,
            progress_callback=progress_cb,
        )
        return {
            "hits": [
                {
                    "seq_id": h.seq_id,
                    "description": h.description,
                    "seq_length": h.seq_length,
                    "identity_pct": h.identity_pct,
                    "sw_score": h.sw_score,
                    "alignment_length": h.alignment_length,
                    "query_coverage_pct": h.query_coverage_pct,
                    "similarity_score": h.similarity_score,
                    "vector_distance": h.vector_distance,
                    "aligned_query": h.aligned_query,
                    "aligned_comp": h.aligned_comp,
                    "aligned_target": h.aligned_target,
                }
                for h in hits
            ]
        }

    return StreamingResponse(
        stream_with_progress(work),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/sequence_search/organism", response_model=OrganismResponse)
def sequence_search_organism(
    payload: OrganismRequest, _: CurrentUserDep
) -> OrganismResponse:
    s = get_settings()
    if not s.llm_endpoint_name:
        return OrganismResponse(organism="Unknown")
    organism = seq_search.extract_organism(payload.description, s.llm_endpoint_name)
    return OrganismResponse(organism=organism)


class InverseFoldingRequest(BaseModel):
    pdb: str = Field(..., min_length=1)


class InverseFoldingResponse(BaseModel):
    sequences: list[str]


@router.post("/inverse_folding", response_model=InverseFoldingResponse)
def inverse_folding(
    payload: InverseFoldingRequest, _: CurrentUserDep
) -> InverseFoldingResponse:
    if "ATOM" not in payload.pdb:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "PDB must contain at least one ATOM record",
        )
    w = WorkspaceClient()  # app SP — OBO tokens lack model-serving scope
    try:
        sequences = hit_proteinmpnn(w, payload.pdb)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"ProteinMPNN call failed: {e}")
    return InverseFoldingResponse(sequences=sequences)


class ProteinDesignRequest(BaseModel):
    sequence: str = Field(..., min_length=1)
    experiment_name: str = Field(..., min_length=1)
    run_name: str = Field(..., min_length=1)
    n_rfdiffusion_hits: int = Field(1, ge=1, le=4)


class ProteinDesignResponse(BaseModel):
    viewer_html: str
    experiment_id: str
    run_id: str
    n_designs: int


@router.post("/protein_design", response_model=ProteinDesignResponse)
def protein_design(
    payload: ProteinDesignRequest, user: CurrentUserDep
) -> ProteinDesignResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    if "[" not in payload.sequence or "]" not in payload.sequence:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Sequence must contain a region to redesign, marked by square brackets [ ]",
        )
    user_info = _build_user_info(user, WorkspaceClient())
    try:
        result = pd_pipeline.make_designs(
            sequence=payload.sequence,
            mlflow_experiment_name=payload.experiment_name,
            mlflow_run_name=payload.run_name,
            user_info=user_info,
            n_rfdiffusion_hits=payload.n_rfdiffusion_hits,
        )
        aligned = pd_pipeline.align_designed_pdbs(
            {"initial": result["initial"], "designed": result["designed"]}
        )
        viewer_html = molstar_html_multibody(aligned, with_iframe=False)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Protein design pipeline failed: {e}")

    return ProteinDesignResponse(
        viewer_html=viewer_html,
        experiment_id=str(result["experiment_id"]),
        run_id=str(result["run_id"]),
        n_designs=len(result["designed"]),
    )


@router.post("/protein_design/stream")
def protein_design_stream(payload: ProteinDesignRequest, user: CurrentUserDep):
    """SSE variant of /protein_design. Real progress through ESMFold,
    RFDiffusion×N, ProteinMPNN×N, ESMFold-each-design, alignment."""
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    if "[" not in payload.sequence or "]" not in payload.sequence:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Sequence must contain a region to redesign, marked by square brackets [ ]",
        )
    user_info = _build_user_info(user, WorkspaceClient())

    def work(progress_cb):
        result = pd_pipeline.make_designs(
            sequence=payload.sequence,
            mlflow_experiment_name=payload.experiment_name,
            mlflow_run_name=payload.run_name,
            user_info=user_info,
            n_rfdiffusion_hits=payload.n_rfdiffusion_hits,
            progress_callback=progress_cb,
        )
        progress_cb(96, "Aligning designed structures to initial fold")
        aligned = pd_pipeline.align_designed_pdbs(
            {"initial": result["initial"], "designed": result["designed"]}
        )
        viewer_html = molstar_html_multibody(aligned, with_iframe=False)
        progress_cb(100, f"Rendered {len(result['designed'])} aligned designs")
        return {
            "viewer_html": viewer_html,
            "experiment_id": str(result["experiment_id"]),
            "run_id": str(result["run_id"]),
            "n_designs": len(result["designed"]),
        }

    return StreamingResponse(
        stream_with_progress(work),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ─── Protein Binder Design ────────────────────────────────────────────────


class BinderDesignRequest(BaseModel):
    # Exactly one of target_pdb / target_sequence must be set. The sequence
    # path runs an extra ESMFold step server-side to fold the target before
    # passing the PDB to Proteina-Complexa.
    target_pdb: Optional[str] = None
    target_sequence: Optional[str] = None
    target_chain: str = Field("A", min_length=1, max_length=4)
    hotspot_residues: str = ""
    binder_length_min: int = Field(50, ge=20, le=200)
    binder_length_max: int = Field(80, ge=20, le=300)
    num_samples: int = Field(2, ge=1, le=10)
    validate_esmfold: bool = True
    mlflow_experiment: str = Field("gwb_binder_design", min_length=1)
    mlflow_run_name: str = Field(..., min_length=1)


class BinderDesign(BaseModel):
    sample_id: str
    sequence: str
    rewards: float
    esmfold_validated: bool
    # Two pre-built Mol* HTMLs so the view-mode toggle is instant on the
    # client. The binder-only one always exists if `binder_pdb` was found;
    # `viewer_html_with_target` overlays the target on top.
    viewer_html_binder_only: Optional[str] = None
    viewer_html_with_target: Optional[str] = None


class BinderDesignResponse(BaseModel):
    designs: list[BinderDesign]
    target_pdb: str
    target_only_viewer_html: str
    experiment_id: str
    run_id: str
    warnings: list[str]


@router.post("/binder_design/stream")
def binder_design_stream(payload: BinderDesignRequest, user: CurrentUserDep):
    """SSE variant. Stages (matched by the frontend `RealtimeProgress`):
        0 →15  Fold target sequence (skipped when target_pdb is supplied)
       15 →50  Proteina-Complexa binder generation
       50 →95  ESMFold validation per design (skipped if disabled)
       95 →100 Build Mol* viewers + log to MLflow"""
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    if not payload.target_pdb and not payload.target_sequence:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Provide either target_pdb or target_sequence",
        )
    user_info = _build_user_info(user, WorkspaceClient())

    def work(progress_cb):
        progress_cb(2, "Setting up MLflow experiment")
        experiment = set_mlflow_experiment(
            experiment_tag=payload.mlflow_experiment,
            user_email=user_info.user_email,
            host=None,
            token=None,
        )

        with mlflow.start_run(
            run_name=payload.mlflow_run_name, experiment_id=experiment.experiment_id
        ) as run:
            mlflow_run_id = run.info.run_id
            mlflow.log_params({
                "target_chain": payload.target_chain,
                "hotspot_residues": payload.hotspot_residues,
                "binder_len_min": payload.binder_length_min,
                "binder_len_max": payload.binder_length_max,
                "num_samples": payload.num_samples,
                "validate_esmfold": payload.validate_esmfold,
                "input_mode": "PDB" if payload.target_pdb else "sequence",
            })
            if payload.target_sequence:
                mlflow.log_param("input_sequence", payload.target_sequence.strip())

            result = binder_pipeline.run_binder_design(
                target_pdb=payload.target_pdb,
                target_sequence=payload.target_sequence,
                target_chain=payload.target_chain,
                hotspot_residues=payload.hotspot_residues,
                binder_length_min=payload.binder_length_min,
                binder_length_max=payload.binder_length_max,
                num_samples=payload.num_samples,
                validate_esmfold=payload.validate_esmfold,
                progress_callback=progress_cb,
            )

            target_pdb = result["target_pdb"]
            designs_raw = result["designs"]
            warnings = result["warnings"]

            # MLflow artifact for full audit trail. Skip the raw PDB blobs
            # (multi-KB) — sample_id + sequence + rewards + validated flag
            # is what an analyst actually wants saved.
            mlflow.log_dict(
                {
                    "designs": [
                        {
                            "sample_id": d["sample_id"],
                            "sequence": d["sequence"],
                            "rewards": d["rewards"],
                            "esmfold_validated": d["esmfold_validated"],
                        }
                        for d in designs_raw
                    ],
                },
                "proteina_complexa_results.json",
            )

            progress_cb(97, "Building Mol* viewers")
            target_only_html = molstar_html_multibody(
                [target_pdb], names=["target"], with_iframe=False
            )
            designs: list[dict] = []
            for d in designs_raw:
                binder_pdb = d.pop("binder_pdb", None)
                binder_only = None
                with_target = None
                if binder_pdb:
                    binder_only = molstar_html_multibody(
                        [binder_pdb], names=["binder"], with_iframe=False
                    )
                    with_target = molstar_html_multibody(
                        [target_pdb, binder_pdb],
                        names=["target", "binder"],
                        with_iframe=False,
                    )
                designs.append({
                    **d,
                    "viewer_html_binder_only": binder_only,
                    "viewer_html_with_target": with_target,
                })

            progress_cb(100, f"Done — {len(designs)} design(s) ready")
            return {
                "designs": designs,
                "target_pdb": target_pdb,
                "target_only_viewer_html": target_only_html,
                "experiment_id": str(experiment.experiment_id),
                "run_id": mlflow_run_id,
                "warnings": warnings,
            }

    return StreamingResponse(
        stream_with_progress(work),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ─── Guided Enzyme Optimization (async-job pattern) ──────────────────────


class EnzymeRefRow(BaseModel):
    sequence: str
    half_life_hours: float
    cell_system: str = "HEK293"


class EnzymeOptimizationStartRequest(BaseModel):
    motif_pdb: str = Field(..., min_length=10)
    motif_residues: list[int] = Field(default_factory=list)
    target_chain: str = "B"
    scaffold_length_min: int = Field(80, ge=20, le=400)
    scaffold_length_max: int = Field(120, ge=20, le=400)
    num_samples: int = Field(8, ge=2, le=32)
    num_iterations: int = Field(10, ge=1, le=30)
    weights: dict[str, float] = Field(default_factory=dict)
    substrate_smiles: str = ""
    references: list[EnzymeRefRow] = Field(default_factory=list)
    half_life_margin: float = Field(0.05, ge=0.01, le=0.5)
    resampling_temperature: float = Field(0.1, ge=0.01, le=1.0)
    strategy: str = Field("resample", pattern=r"^(resample|noop)$")
    run_proteinmpnn: bool = True
    # Stopping criteria — None / negative sentinel disables.
    convergence_threshold: Optional[float] = 0.01
    convergence_window: int = 2
    target_reward: Optional[float] = None
    best_k_target: Optional[int] = None
    best_k_threshold: Optional[float] = None
    use_inprocess_ame: bool = False
    mlflow_experiment: str = Field("gwb_enzyme_optimization", min_length=1)
    mlflow_run_name: str = Field(..., min_length=1)


class EnzymeOptimizationStartResponse(BaseModel):
    job_id: int
    job_run_id: int
    mlflow_run_id: str
    experiment_id: str
    run_url: str


@router.post("/enzyme_optimization/start", response_model=EnzymeOptimizationStartResponse)
def enzyme_optimization_start(
    payload: EnzymeOptimizationStartRequest, user: CurrentUserDep
) -> EnzymeOptimizationStartResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    if payload.scaffold_length_max < payload.scaffold_length_min:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "scaffold_length_max must be >= scaffold_length_min",
        )
    user_info = _build_user_info(user, WorkspaceClient())
    try:
        result = enzyme_pipeline.start_enzyme_optimization_job(
            motif_pdb_str=payload.motif_pdb,
            motif_residues=payload.motif_residues,
            target_chain=payload.target_chain,
            scaffold_length_min=payload.scaffold_length_min,
            scaffold_length_max=payload.scaffold_length_max,
            num_samples=payload.num_samples,
            num_iterations=payload.num_iterations,
            weights=payload.weights,
            user_info=user_info,
            mlflow_experiment=payload.mlflow_experiment,
            mlflow_run_name=payload.mlflow_run_name,
            substrate_smiles=payload.substrate_smiles,
            references=[r.model_dump() for r in payload.references],
            half_life_margin=payload.half_life_margin,
            resampling_temperature=payload.resampling_temperature,
            strategy=payload.strategy,
            run_proteinmpnn=payload.run_proteinmpnn,
            convergence_threshold=payload.convergence_threshold,
            convergence_window=payload.convergence_window,
            target_reward=payload.target_reward,
            best_k_target=payload.best_k_target,
            best_k_threshold=payload.best_k_threshold,
            use_inprocess_ame=payload.use_inprocess_ame,
        )
    except Exception as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Failed to dispatch enzyme-optimization job: {e}",
        )

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    run_url = f"{host}/jobs/{result.job_id}/runs/{result.job_run_id}" if host else ""
    return EnzymeOptimizationStartResponse(
        job_id=result.job_id,
        job_run_id=result.job_run_id,
        mlflow_run_id=result.mlflow_run_id,
        experiment_id=result.experiment_id,
        run_url=run_url,
    )


class EnzymeRunRow(BaseModel):
    run_id: str
    run_name: str
    experiment_name: str
    generation_mode: str
    iter_max_reward: Optional[float] = None
    iterations_completed: Optional[int] = None
    start_time_ms: Optional[int] = None
    job_status: str
    progress: str
    run_url: str = ""


class EnzymeSearchResponse(BaseModel):
    runs: list[EnzymeRunRow]


@router.get("/enzyme_optimization/search", response_model=EnzymeSearchResponse)
def enzyme_optimization_search(
    user: CurrentUserDep,
    by: str = Query("run_name", pattern=r"^(run_name|experiment_name)$"),
    text: str = Query(..., min_length=1),
) -> EnzymeSearchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    runs = enzyme_pipeline.search_runs(user.email, by, text.strip())
    return EnzymeSearchResponse(
        runs=[
            EnzymeRunRow(
                run_id=r.run_id,
                run_name=r.run_name,
                experiment_name=r.experiment_name,
                generation_mode=r.generation_mode,
                iter_max_reward=r.iter_max_reward,
                iterations_completed=r.iterations_completed,
                start_time_ms=r.start_time_ms,
                job_status=r.job_status,
                progress=r.progress,
                run_url=r.run_url,
            )
            for r in runs
        ]
    )


class EnzymeRewardHistoryPoint(BaseModel):
    step: int
    value: float


class EnzymeStatusResponse(BaseModel):
    status: str
    job_status: str
    run_name: str
    experiment_id: str
    iter_max_reward_history: list[EnzymeRewardHistoryPoint]
    iter_mean_reward_history: list[EnzymeRewardHistoryPoint]
    current_metrics: dict[str, float]
    # Top-25 rows of the reward trajectory. Columns vary; emit as flat dicts.
    trajectory: list[dict]


@router.get("/enzyme_optimization/status", response_model=EnzymeStatusResponse)
def enzyme_optimization_status(
    user: CurrentUserDep,
    run_id: str = Query(..., min_length=1),
) -> EnzymeStatusResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    status_d = enzyme_pipeline.get_run_status(run_id)
    traj = enzyme_pipeline.load_optimization_trajectory(run_id)
    # Keep payload small; the dialog shows ~25 rows.
    traj_rows = (
        traj.head(25).where(traj.head(25).notna(), None).to_dict(orient="records")
        if not traj.empty
        else []
    )
    return EnzymeStatusResponse(
        status=status_d["status"],
        job_status=status_d["job_status"],
        run_name=status_d["run_name"],
        experiment_id=status_d["experiment_id"],
        iter_max_reward_history=[
            EnzymeRewardHistoryPoint(**p) for p in status_d["iter_max_reward_history"]
        ],
        iter_mean_reward_history=[
            EnzymeRewardHistoryPoint(**p) for p in status_d["iter_mean_reward_history"]
        ],
        current_metrics=status_d["current_metrics"],
        trajectory=traj_rows,
    )


class EnzymeCandidate(BaseModel):
    candidate_id: str
    pdb: str
    viewer_html: str


class EnzymeTopKResponse(BaseModel):
    candidates: list[EnzymeCandidate]


@router.get("/enzyme_optimization/top_k", response_model=EnzymeTopKResponse)
def enzyme_optimization_top_k(
    user: CurrentUserDep,
    run_id: str = Query(..., min_length=1),
) -> EnzymeTopKResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    pdbs = enzyme_pipeline.load_top_k_pdbs(run_id)
    candidates = [
        EnzymeCandidate(
            candidate_id=cid,
            pdb=pdb,
            viewer_html=molstar_html_singlebody(pdb, name=cid, with_iframe=False),
        )
        for cid, pdb in pdbs.items()
    ]
    return EnzymeTopKResponse(candidates=candidates)


class EnzymeSmokeTestResponse(BaseModel):
    sequence: str
    solubility: Optional[float] = None
    half_life: Optional[float] = None
    thermostab: Optional[float] = None
    immuno: Optional[float] = None


@router.post("/enzyme_optimization/smoke_test", response_model=EnzymeSmokeTestResponse)
def enzyme_optimization_smoke_test(user: CurrentUserDep) -> EnzymeSmokeTestResponse:
    """One round-trip to each developability predictor on T4 lysozyme — used
    to verify the four endpoints are healthy before launching a full run."""
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    scores = enzyme_pipeline.predict_enzyme_properties(enzyme_pipeline.T4_LYSOZYME_SEQUENCE)
    return EnzymeSmokeTestResponse(
        sequence=enzyme_pipeline.T4_LYSOZYME_SEQUENCE,
        solubility=scores.get("solubility"),
        half_life=scores.get("half_life"),
        thermostab=scores.get("thermostab"),
        immuno=scores.get("immuno"),
    )


class EnzymeDefaultsResponse(BaseModel):
    motif_pdb: str
    default_weights: dict[str, float]
    default_references: list[EnzymeRefRow]


@router.get("/enzyme_optimization/defaults", response_model=EnzymeDefaultsResponse)
def enzyme_optimization_defaults(_: CurrentUserDep) -> EnzymeDefaultsResponse:
    """Form defaults: example motif PDB + the standard axis weight map +
    the two T4-lysozyme reference enzymes (stable + N-end destabilised)."""
    # Reuse the motif scaffolding example so we don't duplicate the string.
    from app.services.motif_scaffolding import hit_proteina_complexa_ame  # noqa: F401
    # The example motif PDB lives in the frontend's MotifScaffoldingTab.tsx;
    # serve a server-side copy so the enzyme tab doesn't have to inline it.
    example_motif = _EXAMPLE_MOTIF_PDB
    return EnzymeDefaultsResponse(
        motif_pdb=example_motif,
        default_weights=enzyme_pipeline.DEFAULT_AXIS_WEIGHTS,
        default_references=[
            EnzymeRefRow(
                sequence=enzyme_pipeline.T4_LYSOZYME_SEQUENCE,
                half_life_hours=24.0,
                cell_system="NIH3T3",
            ),
            EnzymeRefRow(
                sequence=enzyme_pipeline.T4_LYSOZYME_NEND_DESTABILIZED,
                half_life_hours=0.5,
                cell_system="NIH3T3",
            ),
        ],
    )


_EXAMPLE_MOTIF_PDB = """ATOM      1  N   HIS B   1       5.123   8.456   2.345  1.00 15.00           N
ATOM      2  CA  HIS B   1       5.891   7.234   2.789  1.00 15.00           C
ATOM      3  C   HIS B   1       7.321   7.567   3.123  1.00 15.00           C
ATOM      4  O   HIS B   1       7.654   8.678   3.567  1.00 15.00           O
ATOM      5  CB  HIS B   1       5.456   6.123   3.678  1.00 15.00           C
ATOM      6  CG  HIS B   1       4.012   5.789   3.456  1.00 15.00           C
ATOM      7  ND1 HIS B   1       3.123   6.567   4.123  1.00 15.00           N
ATOM      8  CE1 HIS B   1       1.890   6.012   3.890  1.00 15.00           C
ATOM      9  NE2 HIS B   1       1.987   4.890   3.123  1.00 15.00           N
ATOM     10  CD2 HIS B   1       3.234   4.678   2.890  1.00 15.00           C
ATOM     11  N   ASP B   2       8.123   6.567   2.890  1.00 15.00           N
ATOM     12  CA  ASP B   2       9.543   6.789   3.234  1.00 15.00           C
ATOM     13  C   ASP B   2      10.234   5.567   3.890  1.00 15.00           C
ATOM     14  O   ASP B   2       9.678   4.456   4.012  1.00 15.00           O
ATOM     15  CB  ASP B   2      10.123   7.890   2.345  1.00 15.00           C
ATOM     16  CG  ASP B   2      11.567   8.123   2.678  1.00 15.00           C
ATOM     17  OD1 ASP B   2      12.234   7.234   3.123  1.00 15.00           O
ATOM     18  OD2 ASP B   2      11.890   9.234   2.345  1.00 15.00           O
ATOM     19  N   SER B   3      11.456   5.678   4.234  1.00 15.00           N
ATOM     20  CA  SER B   3      12.234   4.567   4.890  1.00 15.00           C
ATOM     21  C   SER B   3      13.678   4.890   5.234  1.00 15.00           C
ATOM     22  O   SER B   3      14.123   5.987   5.012  1.00 15.00           O
ATOM     23  CB  SER B   3      11.890   3.234   4.234  1.00 15.00           C
ATOM     24  OG  SER B   3      12.567   2.123   4.678  1.00 15.00           O
HETATM   25  C1  LIG B   1       6.500   3.200   5.100  1.00  5.00           C
HETATM   26  O1  LIG B   1       7.200   2.100   5.500  1.00  5.00           O
HETATM   27  N1  LIG B   1       5.300   3.500   5.800  1.00  5.00           N
END
"""

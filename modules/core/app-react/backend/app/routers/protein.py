from databricks.sdk import WorkspaceClient
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from genesis_workbench.workbench import UserInfo
from pydantic import BaseModel, Field

from app.auth import CurrentUserDep
from app.config import get_settings
from app.services import alphafold as af
from app.services import protein_design as pd_pipeline
from app.services import sequence_search as seq_search
from app.services.molstar import molstar_html_multibody, molstar_html_singlebody
from app.services.protein import hit_boltz, hit_esmfold, hit_proteinmpnn
from app.services.sse import stream_with_progress

router = APIRouter(prefix="/api/protein_studies", tags=["protein_studies"])


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


class SequenceHit(BaseModel):
    seq_id: str
    description: str
    seq_length: int
    identity_pct: float
    sw_score: int
    alignment_length: int
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
        hits = seq_search.run_sequence_search(sequence, top_k=payload.top_k)
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
            sequence, top_k=payload.top_k, progress_callback=progress_cb
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

from collections import Counter

from databricks.sdk import WorkspaceClient
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.auth import CurrentUserDep
from app.config import get_settings
from app.routers.large_molecule import _build_user_info
from app.services import scgpt as scgpt_service
from app.services import scimilarity as scim
from app.services import single_cell_runs as runs_service
from app.services import teddy as teddy_service
from app.services.sse import stream_with_progress

router = APIRouter(prefix="/api/single_cell", tags=["single_cell"])


class SingleCellRun(BaseModel):
    run_id: str
    run_name: str
    experiment_name: str
    processing_mode: str
    start_time_ms: int | None
    status: str
    progress: str
    cells: int | None


class RunsResponse(BaseModel):
    runs: list[SingleCellRun]


@router.get("/runs", response_model=RunsResponse)
def list_runs(user: CurrentUserDep) -> RunsResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    items = runs_service.search_runs(user.email)
    return RunsResponse(
        runs=[
            SingleCellRun(
                run_id=r.run_id,
                run_name=r.run_name,
                experiment_name=r.experiment_name,
                processing_mode=r.processing_mode,
                start_time_ms=r.start_time_ms,
                status=r.status,
                progress=runs_service.progress_chip(r.status),
                cells=r.cells,
            )
            for r in items
        ]
    )


class StartProcessingRequest(BaseModel):
    mode: str = Field(..., pattern=r"^(scanpy|rapids-singlecell)$")
    data_path: str = Field(..., min_length=1)
    mlflow_experiment: str = Field(..., min_length=1)
    mlflow_run_name: str = Field(..., min_length=1)
    gene_name_column: str = ""
    species: str = "hsapiens"
    min_genes: int = Field(200, ge=0)
    min_cells: int = Field(3, ge=0)
    pct_counts_mt: float = Field(5.0, ge=0.0, le=100.0)
    n_genes_by_counts: int = Field(2500, ge=0)
    target_sum: int = Field(10000, ge=0)
    n_top_genes: int = Field(2000, ge=0)
    n_pcs: int = Field(50, ge=0)
    cluster_resolution: float = Field(0.15, ge=0.0, le=2.0)
    compute_pseudotime: bool = False


class StartProcessingResponse(BaseModel):
    job_id: int
    job_run_id: int
    mlflow_run_id: str
    experiment_id: str
    run_url: str


@router.post("/start", response_model=StartProcessingResponse)
def start_processing(
    payload: StartProcessingRequest, user: CurrentUserDep
) -> StartProcessingResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    if not payload.data_path.startswith("/Volumes"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "data_path must be a Unity Catalog Volume path (start with /Volumes/...)",
        )
    if not payload.data_path.endswith(".h5ad"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "data_path must point to an .h5ad file",
        )

    user_info = _build_user_info(user, WorkspaceClient())
    try:
        dispatch = runs_service.start_job(
            mode=payload.mode,
            user_info=user_info,
            data_path=payload.data_path,
            mlflow_experiment=payload.mlflow_experiment,
            mlflow_run_name=payload.mlflow_run_name,
            gene_name_column=payload.gene_name_column,
            species=payload.species,
            min_genes=payload.min_genes,
            min_cells=payload.min_cells,
            pct_counts_mt=payload.pct_counts_mt,
            n_genes_by_counts=payload.n_genes_by_counts,
            target_sum=payload.target_sum,
            n_top_genes=payload.n_top_genes,
            n_pcs=payload.n_pcs,
            cluster_resolution=payload.cluster_resolution,
            compute_pseudotime=payload.compute_pseudotime,
        )
    except Exception as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Failed to start {payload.mode} job: {e}",
        )

    import os as _os

    host = _os.environ.get("DATABRICKS_HOSTNAME", "")
    if host and not host.startswith("https://"):
        host = f"https://{host}"
    run_url = f"{host}/jobs/{dispatch.job_id}/runs/{dispatch.job_run_id}"

    return StartProcessingResponse(
        job_id=dispatch.job_id,
        job_run_id=dispatch.job_run_id,
        mlflow_run_id=dispatch.mlflow_run_id,
        experiment_id=dispatch.experiment_id,
        run_url=run_url,
    )


class RunSummaryRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    max_umap_points: int = Field(10_000, ge=100, le=50_000)


class KeyMetric(BaseModel):
    label: str
    value: str


class RunSummaryUmapPoint(BaseModel):
    umap_0: float
    umap_1: float
    cluster: str


class RunSummaryResponse(BaseModel):
    cells_total: int | None
    cells_subsample: int
    clusters_count: int
    markers_count: int
    has_umap: bool
    has_pseudotime: bool
    key_metrics: list[KeyMetric]
    umap_points: list[RunSummaryUmapPoint]
    cluster_col: str
    clusters: list[str]
    expr_genes: list[str]
    obs_categorical: list[str]
    obs_numerical: list[str]
    all_columns: list[str]
    mlflow_run_url: str | None


# Curated set shown first in run-header tables
# so the React card surfaces the same "useful subset" of MLflow metrics.
KEY_METRICS_ORDER: list[tuple[str, str, str]] = [
    ("total_cells_starting", "Total cells (input)", "{:,.0f}"),
    ("total_cells_before_subsample", "Cells before subsample", "{:,.0f}"),
    ("filter_simple_retention", "Filter retention", "{:.1f}%"),
    ("filter_mtgenes_retention", "MT-gene retention", "{:.1f}%"),
    ("gene_mapping_rate", "Gene-mapping rate", "{:.1f}%"),
    ("total_time", "Total time (s)", "{:.1f}s"),
]


@router.post("/run-summary", response_model=RunSummaryResponse)
def run_summary(payload: RunSummaryRequest, _: CurrentUserDep) -> RunSummaryResponse:
    try:
        markers_df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Could not download markers_flat.parquet: {e}",
        )

    cluster_col = _detect_cluster_col(markers_df)
    expr_cols = [c for c in markers_df.columns if c.startswith("expr_")]
    has_umap = "UMAP_0" in markers_df.columns and "UMAP_1" in markers_df.columns

    # Pull a curated subset of MLflow metrics for the summary header.
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()
    metrics: dict[str, float] = {}
    try:
        run_info = client.get_run(payload.run_id)
        metrics = dict(run_info.data.metrics or {})
    except Exception:
        metrics = {}

    key_metrics: list[KeyMetric] = []
    for key, label, fmt in KEY_METRICS_ORDER:
        if key in metrics:
            try:
                key_metrics.append(KeyMetric(label=label, value=fmt.format(metrics[key])))
            except Exception:
                key_metrics.append(KeyMetric(label=label, value=str(metrics[key])))

    cells_total = (
        int(metrics["total_cells_before_subsample"])
        if "total_cells_before_subsample" in metrics
        else (int(metrics["total_cells_starting"]) if "total_cells_starting" in metrics else None)
    )

    # Sample UMAP points to keep the payload bounded. NaN/Inf coords would
    # produce bare NaN/Infinity tokens in the JSON response and crash
    # JSON.parse on the client — filter them out here.
    import math

    points: list[RunSummaryUmapPoint] = []
    if has_umap and len(markers_df) > 0:
        sample = markers_df
        if len(markers_df) > payload.max_umap_points:
            sample = markers_df.sample(n=payload.max_umap_points, random_state=42)
        for _, row in sample.iterrows():
            try:
                x = float(row["UMAP_0"])
                y = float(row["UMAP_1"])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            points.append(
                RunSummaryUmapPoint(
                    umap_0=x,
                    umap_1=y,
                    cluster=str(row[cluster_col]),
                )
            )

    import os as _os
    import pandas as pd

    expr_genes = [c.replace("expr_", "") for c in expr_cols]
    obs_categorical = [
        c
        for c in markers_df.columns
        if not c.startswith("expr_")
        and not c.startswith("UMAP_")
        and not c.startswith("PC_")
        and (markers_df[c].dtype == "object" or pd.api.types.is_categorical_dtype(markers_df[c]))
    ]
    obs_numerical = [
        c
        for c in markers_df.columns
        if not c.startswith("expr_")
        and not c.startswith("UMAP_")
        and not c.startswith("PC_")
        and pd.api.types.is_numeric_dtype(markers_df[c])
    ]
    clusters_sorted = sorted(
        [str(c) for c in markers_df[cluster_col].unique()],
        key=lambda x: int(x) if x.isdigit() else x,
    )

    host = _os.environ.get("DATABRICKS_HOSTNAME", "")
    if host and not host.startswith("https://"):
        host = f"https://{host}"
    mlflow_run_url = f"{host}/ml/experiments/runs/{payload.run_id}" if host else None

    return RunSummaryResponse(
        cells_total=cells_total,
        cells_subsample=int(len(markers_df)),
        clusters_count=int(markers_df[cluster_col].nunique()),
        markers_count=len(expr_cols),
        has_umap=has_umap,
        has_pseudotime="dpt_pseudotime" in markers_df.columns,
        key_metrics=key_metrics,
        umap_points=points,
        cluster_col=cluster_col,
        clusters=clusters_sorted,
        expr_genes=expr_genes,
        obs_categorical=obs_categorical,
        obs_numerical=obs_numerical,
        all_columns=[c for c in markers_df.columns],
        mlflow_run_url=mlflow_run_url,
    )


class AnnotateRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    cells_per_cluster: int = Field(10, ge=3, le=50)
    k_neighbors: int = Field(20, ge=5, le=200)


class ClusterAnnotation(BaseModel):
    cluster: str
    predicted_cell_type: str
    confidence_pct: float
    top_predictions: str


class UmapPoint(BaseModel):
    umap_0: float
    umap_1: float
    cluster: str
    predicted_cell_type: str


class AnnotateResponse(BaseModel):
    annotations: list[ClusterAnnotation]
    umap_points: list[UmapPoint]


@router.post("/annotate", response_model=AnnotateResponse)
def annotate(payload: AnnotateRequest, _: CurrentUserDep) -> AnnotateResponse:
    try:
        markers_df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Could not download markers_flat.parquet for run {payload.run_id}: {e}",
        )

    cluster_col = next(
        (c for c in ["cluster", "leiden", "louvain"] if c in markers_df.columns),
        None,
    )
    if not cluster_col:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No cluster column found in markers_flat.parquet (expected one of cluster/leiden/louvain)",
        )

    try:
        results_df, cluster_to_type = scim.annotate_clusters(
            markers_df,
            cluster_col=cluster_col,
            cells_per_cluster=payload.cells_per_cluster,
            k_neighbors=payload.k_neighbors,
        )
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Annotation pipeline failed: {e}")

    annotations = [
        ClusterAnnotation(
            cluster=row["cluster"],
            predicted_cell_type=row["predicted_cell_type"],
            confidence_pct=float(row["confidence_pct"]),
            top_predictions=row["top_predictions"],
        )
        for _, row in results_df.iterrows()
    ]

    umap_points: list[UmapPoint] = []
    if "UMAP_0" in markers_df.columns and "UMAP_1" in markers_df.columns:
        for _, row in markers_df.iterrows():
            cl = str(row[cluster_col])
            umap_points.append(
                UmapPoint(
                    umap_0=float(row["UMAP_0"]),
                    umap_1=float(row["UMAP_1"]),
                    cluster=cl,
                    predicted_cell_type=cluster_to_type.get(cl, "Unknown"),
                )
            )

    return AnnotateResponse(annotations=annotations, umap_points=umap_points)


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.post("/annotate/stream")
def annotate_stream(payload: AnnotateRequest, _: CurrentUserDep):
    """SSE variant of /annotate. Emits `progress`, then a final `result`
    matching AnnotateResponse, or an `error` event on failure."""
    try:
        markers_df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Could not download markers_flat.parquet for run {payload.run_id}: {e}",
        )

    cluster_col = next(
        (c for c in ["cluster", "leiden", "louvain"] if c in markers_df.columns),
        None,
    )
    if not cluster_col:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No cluster column found in markers_flat.parquet (expected one of cluster/leiden/louvain)",
        )

    def work(progress_cb):
        results_df, cluster_to_type = scim.annotate_clusters(
            markers_df,
            cluster_col=cluster_col,
            cells_per_cluster=payload.cells_per_cluster,
            k_neighbors=payload.k_neighbors,
            progress_callback=progress_cb,
        )
        annotations = [
            {
                "cluster": row["cluster"],
                "predicted_cell_type": row["predicted_cell_type"],
                "confidence_pct": float(row["confidence_pct"]),
                "top_predictions": row["top_predictions"],
            }
            for _, row in results_df.iterrows()
        ]
        umap_points: list[dict] = []
        if "UMAP_0" in markers_df.columns and "UMAP_1" in markers_df.columns:
            for _, row in markers_df.iterrows():
                cl = str(row[cluster_col])
                umap_points.append({
                    "umap_0": float(row["UMAP_0"]),
                    "umap_1": float(row["UMAP_1"]),
                    "cluster": cl,
                    "predicted_cell_type": cluster_to_type.get(cl, "Unknown"),
                })
        # Persist to MLflow so the panel can restore the table on next Load
        # Results, and so downstream tools/notebooks can read the same JSON.
        try:
            runs_service.save_annotation(
                run_id=payload.run_id,
                model="scimilarity",
                cluster_col=cluster_col,
                results=annotations,
            )
        except Exception as e:
            # Don't fail the user-visible annotation just because the save
            # leg hit a permissions or MLflow-server issue — log and move on.
            import logging
            logging.getLogger(__name__).warning(
                "SCimilarity annotation save to MLflow run %s failed: %s",
                payload.run_id, e,
            )
        return {"annotations": annotations, "umap_points": umap_points}

    return StreamingResponse(
        stream_with_progress(work),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ─── TEDDY annotation ──────────────────────────────────────────────────────


class TeddyAnnotateRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    cells_per_cluster: int = Field(20, ge=3, le=50)
    k_neighbors: int = Field(50, ge=5, le=200)
    bias_correct: bool = True


class TeddyClusterAnnotation(BaseModel):
    cluster: str
    n_cells: int
    predicted_cell_type: str
    cell_type_confidence_pct: float
    cell_type_top3: str
    predicted_disease: str
    disease_confidence_pct: float
    disease_top3: str


class TeddyAnnotateResponse(BaseModel):
    annotations: list[TeddyClusterAnnotation]
    cluster_to_cell_type: dict[str, str]
    cluster_to_disease: dict[str, str]


@router.post("/annotate-teddy", response_model=TeddyAnnotateResponse)
def annotate_teddy(payload: TeddyAnnotateRequest, _: CurrentUserDep) -> TeddyAnnotateResponse:
    try:
        markers_df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Could not download markers_flat.parquet: {e}",
        )

    cluster_col = next(
        (c for c in ["cluster", "leiden", "louvain"] if c in markers_df.columns),
        None,
    )
    if not cluster_col:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No cluster column found in markers_flat.parquet",
        )

    try:
        rows, ct_map, ds_map = teddy_service.annotate_clusters(
            markers_df,
            cluster_col=cluster_col,
            cells_per_cluster=payload.cells_per_cluster,
            k_neighbors=payload.k_neighbors,
            bias_correct=payload.bias_correct,
        )
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"TEDDY annotation failed: {e}")

    return TeddyAnnotateResponse(
        annotations=[
            TeddyClusterAnnotation(
                cluster=r.cluster,
                n_cells=r.n_cells,
                predicted_cell_type=r.predicted_cell_type,
                cell_type_confidence_pct=r.cell_type_confidence_pct,
                cell_type_top3=r.cell_type_top3,
                predicted_disease=r.predicted_disease,
                disease_confidence_pct=r.disease_confidence_pct,
                disease_top3=r.disease_top3,
            )
            for r in rows
        ],
        cluster_to_cell_type=ct_map,
        cluster_to_disease=ds_map,
    )


@router.post("/annotate-teddy/stream")
def annotate_teddy_stream(payload: TeddyAnnotateRequest, _: CurrentUserDep):
    """SSE variant of /annotate-teddy. Emits `progress`, then a final
    `result` matching TeddyAnnotateResponse, or an `error` event."""
    try:
        markers_df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Could not download markers_flat.parquet: {e}",
        )

    cluster_col = next(
        (c for c in ["cluster", "leiden", "louvain"] if c in markers_df.columns),
        None,
    )
    if not cluster_col:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No cluster column found in markers_flat.parquet",
        )

    def work(progress_cb):
        rows, ct_map, ds_map = teddy_service.annotate_clusters(
            markers_df,
            cluster_col=cluster_col,
            cells_per_cluster=payload.cells_per_cluster,
            k_neighbors=payload.k_neighbors,
            bias_correct=payload.bias_correct,
            progress_callback=progress_cb,
        )
        annotations = [
            {
                "cluster": r.cluster,
                "n_cells": r.n_cells,
                "predicted_cell_type": r.predicted_cell_type,
                "cell_type_confidence_pct": r.cell_type_confidence_pct,
                "cell_type_top3": r.cell_type_top3,
                "predicted_disease": r.predicted_disease,
                "disease_confidence_pct": r.disease_confidence_pct,
                "disease_top3": r.disease_top3,
            }
            for r in rows
        ]
        try:
            runs_service.save_annotation(
                run_id=payload.run_id,
                model="teddy",
                cluster_col=cluster_col,
                results=annotations,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "TEDDY annotation save to MLflow run %s failed: %s",
                payload.run_id, e,
            )
        return {
            "annotations": annotations,
            "cluster_to_cell_type": ct_map,
            "cluster_to_disease": ds_map,
        }

    return StreamingResponse(
        stream_with_progress(work),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ─── load previously-saved annotations ────────────────────────────────────


class SavedAnnotationsResponse(BaseModel):
    scimilarity: AnnotateResponse | None = None
    teddy: TeddyAnnotateResponse | None = None


@router.get("/annotations", response_model=SavedAnnotationsResponse)
def get_saved_annotations(run_id: str, _: CurrentUserDep) -> SavedAnnotationsResponse:
    """Return whichever of {scimilarity, teddy} annotation artifacts exist on
    the run. UMAP points are intentionally empty here — the panel rebuilds
    overlay traces from the (already-loaded) run-summary umap_points and the
    cluster→label map derived from `annotations`."""
    scim_payload = runs_service.load_annotation(run_id, "scimilarity")
    teddy_payload = runs_service.load_annotation(run_id, "teddy")

    scim_response: AnnotateResponse | None = None
    if scim_payload:
        results = scim_payload.get("results") or []
        scim_response = AnnotateResponse(
            annotations=[
                ClusterAnnotation(
                    cluster=str(r.get("cluster")),
                    predicted_cell_type=str(r.get("predicted_cell_type", "Unknown")),
                    confidence_pct=float(r.get("confidence_pct", 0.0)),
                    top_predictions=str(r.get("top_predictions", "")),
                )
                for r in results
            ],
            umap_points=[],
        )

    teddy_response: TeddyAnnotateResponse | None = None
    if teddy_payload:
        results = teddy_payload.get("results") or []
        annotations = [
            TeddyClusterAnnotation(
                cluster=str(r.get("cluster")),
                n_cells=int(r.get("n_cells", 0)),
                predicted_cell_type=str(r.get("predicted_cell_type", "Unknown")),
                cell_type_confidence_pct=float(r.get("cell_type_confidence_pct", 0.0)),
                cell_type_top3=str(r.get("cell_type_top3", "")),
                predicted_disease=str(r.get("predicted_disease", "Unknown")),
                disease_confidence_pct=float(r.get("disease_confidence_pct", 0.0)),
                disease_top3=str(r.get("disease_top3", "")),
            )
            for r in results
        ]
        teddy_response = TeddyAnnotateResponse(
            annotations=annotations,
            cluster_to_cell_type={a.cluster: a.predicted_cell_type for a in annotations},
            cluster_to_disease={a.cluster: a.predicted_disease for a in annotations},
        )

    return SavedAnnotationsResponse(scimilarity=scim_response, teddy=teddy_response)


def _detect_cluster_col(markers_df) -> str:
    for c in ("cluster", "leiden", "louvain"):
        if c in markers_df.columns:
            return c
    raise HTTPException(
        status.HTTP_400_BAD_REQUEST,
        "No cluster column found in markers_flat.parquet (expected cluster/leiden/louvain)",
    )


class RunInfoRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    top_genes_per_cluster: int = Field(50, ge=5, le=200)


class GeneEntry(BaseModel):
    gene: str
    mean_expr: float


class RunInfoResponse(BaseModel):
    cluster_col: str
    clusters: list[str]
    n_cells: int
    has_umap: bool
    top_genes_by_cluster: dict[str, list[GeneEntry]]


@router.post("/run-info", response_model=RunInfoResponse)
def run_info(payload: RunInfoRequest, _: CurrentUserDep) -> RunInfoResponse:
    try:
        markers_df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Could not download markers_flat.parquet: {e}",
        )

    cluster_col = _detect_cluster_col(markers_df)
    expr_cols = [c for c in markers_df.columns if c.startswith("expr_")]

    clusters_raw = list(markers_df[cluster_col].unique())
    clusters = sorted(
        [str(c) for c in clusters_raw],
        key=lambda x: int(x) if x.isdigit() else x,
    )

    top_genes_by_cluster: dict[str, list[GeneEntry]] = {}
    for cl in clusters:
        mask = markers_df[cluster_col].astype(str) == cl
        if not mask.any() or not expr_cols:
            top_genes_by_cluster[cl] = []
            continue
        mean_expr = markers_df.loc[mask, expr_cols].mean().sort_values(ascending=False)
        top_n = mean_expr.head(payload.top_genes_per_cluster)
        top_genes_by_cluster[cl] = [
            GeneEntry(gene=col.replace("expr_", ""), mean_expr=float(val))
            for col, val in top_n.items()
        ]

    return RunInfoResponse(
        cluster_col=cluster_col,
        clusters=clusters,
        n_cells=int(len(markers_df)),
        has_umap="UMAP_0" in markers_df.columns and "UMAP_1" in markers_df.columns,
        top_genes_by_cluster=top_genes_by_cluster,
    )


class SimilarityRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    cluster: str
    k_neighbors: int = Field(100, ge=10, le=1000)
    cells_per_cluster: int = Field(20, ge=3, le=50)
    reference: str = Field("scimilarity", pattern=r"^(scimilarity|teddy)$")


class CategoryCount(BaseModel):
    name: str
    count: int


class SimilarityResponse(BaseModel):
    total_neighbors: int
    cell_types: list[CategoryCount]
    diseases: list[CategoryCount]
    tissues: list[CategoryCount]
    sources: list[CategoryCount]


@router.post("/similarity", response_model=SimilarityResponse)
def similarity(payload: SimilarityRequest, _: CurrentUserDep) -> SimilarityResponse:
    try:
        markers_df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not download markers: {e}")

    cluster_col = _detect_cluster_col(markers_df)
    expr_cols = [c for c in markers_df.columns if c.startswith("expr_")]
    if not expr_cols:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "markers_flat.parquet has no expr_* columns"
        )

    cluster_mask = markers_df[cluster_col].astype(str) == str(payload.cluster)
    cluster_indices = markers_df.index[cluster_mask].tolist()
    if not cluster_indices:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Cluster '{payload.cluster}' not found in run {payload.run_id}",
        )

    n_sample = min(payload.cells_per_cluster, len(cluster_indices))
    sampled = cluster_indices[:n_sample]

    # SCimilarity pipeline: align + lognorm + embed + per-cell VS search.
    try:
        gene_order = scim.get_gene_order()
    except Exception as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"SCimilarity Gene Order endpoint failed: {e}"
        )

    expr_df = markers_df[expr_cols].copy()
    expr_df.columns = [c.replace("expr_", "") for c in expr_cols]
    aligned = scim.align_to_gene_order(expr_df, gene_order)
    normed = scim.lognorm_counts(aligned)

    sample_normed = normed.loc[sampled]
    try:
        embeddings_result = scim.get_cell_embeddings(sample_normed)
    except Exception as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"SCimilarity Get Embedding failed: {e}",
        )

    cell_type_ctr: Counter[str] = Counter()
    disease_ctr: Counter[str] = Counter()
    tissue_ctr: Counter[str] = Counter()
    source_ctr: Counter[str] = Counter()
    total = 0

    for _, row in embeddings_result.iterrows():
        embedding = row["embedding"]
        if isinstance(embedding, str):
            import json as _json

            embedding = _json.loads(embedding)
        try:
            nn_meta = scim.search_nearest_cells(embedding, k=payload.k_neighbors)
        except Exception:
            continue
        if nn_meta.empty:
            continue
        if "prediction" in nn_meta.columns:
            cell_type_ctr.update(nn_meta["prediction"].dropna().astype(str).tolist())
        if "disease" in nn_meta.columns:
            disease_ctr.update(nn_meta["disease"].dropna().astype(str).tolist())
        if "tissue" in nn_meta.columns:
            tissue_ctr.update(nn_meta["tissue"].dropna().astype(str).tolist())
        if "study" in nn_meta.columns:
            source_ctr.update(nn_meta["study"].dropna().astype(str).tolist())
        total += len(nn_meta)

    def _to_rows(c: Counter[str], top: int = 25) -> list[CategoryCount]:
        return [CategoryCount(name=n, count=v) for n, v in c.most_common(top)]

    return SimilarityResponse(
        total_neighbors=total,
        cell_types=_to_rows(cell_type_ctr),
        diseases=_to_rows(disease_ctr),
        tissues=_to_rows(tissue_ctr),
        sources=_to_rows(source_ctr, top=50),
    )


@router.post("/similarity/stream")
def similarity_stream(payload: SimilarityRequest, _: CurrentUserDep):
    """SSE variant of /similarity. Emits real progress through the embed
    batch + per-cell KNN loops, then a final `result`."""
    try:
        markers_df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not download markers: {e}")

    cluster_col = _detect_cluster_col(markers_df)
    expr_cols = [c for c in markers_df.columns if c.startswith("expr_")]
    if not expr_cols:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "markers_flat.parquet has no expr_* columns"
        )

    cluster_mask = markers_df[cluster_col].astype(str) == str(payload.cluster)
    cluster_indices = markers_df.index[cluster_mask].tolist()
    if not cluster_indices:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Cluster '{payload.cluster}' not found in run {payload.run_id}",
        )

    n_sample = min(payload.cells_per_cluster, len(cluster_indices))
    sampled = cluster_indices[:n_sample]

    is_teddy = payload.reference == "teddy"

    def work(progress_cb):
        import json as _json

        # Per-cell neighbour metadata columns differ by reference:
        #   SCimilarity → prediction / disease / tissue / study
        #   TEDDY       → cell_type  / disease / tissue / dataset_id
        # We aggregate into the same CategoryCount buckets so the UI stays
        # uniform across both pills.
        if is_teddy:
            progress_cb(2, "Checking TEDDY reference assets")
            teddy_service._check_teddy_assets_available()

            gene_names = [c.replace("expr_", "") for c in expr_cols]
            sample_expr = markers_df.loc[sampled, expr_cols].copy()
            sample_expr.columns = gene_names

            embeddings_result = teddy_service.embed_cells(
                sample_expr,
                gene_names,
                progress_callback=progress_cb,
                pct_start=10,
                pct_end=40,
            )
            cell_type_col = "cell_type"
            source_col = "dataset_id"
            search = teddy_service.search_nearest_cells
        else:
            progress_cb(5, "Fetching SCimilarity gene order")
            gene_order = scim.get_gene_order()
            expr_df = markers_df[expr_cols].copy()
            expr_df.columns = [c.replace("expr_", "") for c in expr_cols]
            aligned = scim.align_to_gene_order(expr_df, gene_order)
            normed = scim.lognorm_counts(aligned)
            sample_normed = normed.loc[sampled]

            progress_cb(15, f"Embedding {len(sample_normed)} cells")
            embeddings_result = scim.get_cell_embeddings(sample_normed)
            cell_type_col = "prediction"
            source_col = "study"
            search = scim.search_nearest_cells

        cell_type_ctr: Counter[str] = Counter()
        disease_ctr: Counter[str] = Counter()
        tissue_ctr: Counter[str] = Counter()
        source_ctr: Counter[str] = Counter()
        total = 0

        n_emb = len(embeddings_result)
        for i, (_, row) in enumerate(embeddings_result.iterrows()):
            embedding = row["embedding"]
            if isinstance(embedding, str):
                embedding = _json.loads(embedding)
            try:
                nn_meta = search(embedding, k=payload.k_neighbors)
            except Exception:
                progress_cb(40 + int(((i + 1) / max(n_emb, 1)) * 55),
                            f"KNN failed for cell {i + 1}/{n_emb}")
                continue
            progress_cb(40 + int(((i + 1) / max(n_emb, 1)) * 55),
                        f"Vector Search neighbours {i + 1}/{n_emb}")
            if nn_meta.empty:
                continue
            if cell_type_col in nn_meta.columns:
                cell_type_ctr.update(nn_meta[cell_type_col].dropna().astype(str).tolist())
            if "disease" in nn_meta.columns:
                disease_ctr.update(nn_meta["disease"].dropna().astype(str).tolist())
            if "tissue" in nn_meta.columns:
                tissue_ctr.update(nn_meta["tissue"].dropna().astype(str).tolist())
            if source_col in nn_meta.columns:
                source_ctr.update(nn_meta[source_col].dropna().astype(str).tolist())
            total += len(nn_meta)

        def _to_rows(c: Counter[str], top: int = 25) -> list[dict]:
            return [{"name": n, "count": v} for n, v in c.most_common(top)]

        progress_cb(98, "Aggregating metadata distributions")
        result = {
            "total_neighbors": total,
            "cell_types": _to_rows(cell_type_ctr),
            "diseases": _to_rows(disease_ctr),
            "tissues": _to_rows(tissue_ctr),
            "sources": _to_rows(source_ctr, top=50),
        }
        progress_cb(100, f"Done — {total} neighbours aggregated")
        return result

    return StreamingResponse(
        stream_with_progress(work),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


class PerturbationRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    cluster: str
    perturbation_type: str = Field(..., pattern=r"^(knockout|overexpress)$")
    genes_to_perturb: list[str] = Field(..., min_length=1)


class PerturbationGene(BaseModel):
    gene_name: str
    original_expression: float | None = None
    predicted_expression: float | None = None
    delta: float | None = None
    abs_delta: float | None = None


class PerturbationResponse(BaseModel):
    results: list[PerturbationGene]
    summary_total_genes: int
    summary_max_abs_delta: float
    summary_significant_count: int  # top 5% by |delta|


@router.post("/perturbation", response_model=PerturbationResponse)
def perturbation(payload: PerturbationRequest, _: CurrentUserDep) -> PerturbationResponse:
    try:
        markers_df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not download markers: {e}")

    cluster_col = _detect_cluster_col(markers_df)
    expr_cols = [c for c in markers_df.columns if c.startswith("expr_")]
    if not expr_cols:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "markers_flat.parquet has no expr_* columns"
        )

    cluster_mask = markers_df[cluster_col].astype(str) == str(payload.cluster)
    cluster_cells = markers_df.loc[cluster_mask]
    if cluster_cells.empty:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Cluster '{payload.cluster}' not found in run {payload.run_id}",
        )

    gene_names = [c.replace("expr_", "") for c in expr_cols]
    mean_expression = cluster_cells[expr_cols].mean().values.tolist()

    try:
        result_df = scgpt_service.predict_perturbation(
            expression=mean_expression,
            gene_names=gene_names,
            genes_to_perturb=payload.genes_to_perturb,
            perturbation_type=payload.perturbation_type,
        )
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"scGPT call failed: {e}")

    if result_df.empty:
        return PerturbationResponse(
            results=[], summary_total_genes=0, summary_max_abs_delta=0.0, summary_significant_count=0
        )

    if "abs_delta" in result_df.columns:
        result_df = result_df.sort_values("abs_delta", ascending=False)
        threshold = result_df["abs_delta"].quantile(0.95)
        significant = int((result_df["abs_delta"] > threshold).sum())
        max_abs = float(result_df["abs_delta"].max())
    else:
        significant = 0
        max_abs = 0.0

    rows: list[PerturbationGene] = []
    for _, r in result_df.head(500).iterrows():
        rows.append(
            PerturbationGene(
                gene_name=str(r.get("gene_name", "")),
                original_expression=_safe_float(r.get("original_expression")),
                predicted_expression=_safe_float(r.get("predicted_expression")),
                delta=_safe_float(r.get("delta")),
                abs_delta=_safe_float(r.get("abs_delta")),
            )
        )

    return PerturbationResponse(
        results=rows,
        summary_total_genes=int(len(result_df)),
        summary_max_abs_delta=max_abs,
        summary_significant_count=significant,
    )


@router.post("/perturbation/stream")
def perturbation_stream(payload: PerturbationRequest, _: CurrentUserDep):
    """SSE variant of /perturbation. Phase progress only — the scGPT call
    itself is a single endpoint round-trip with no internal stages."""
    try:
        markers_df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not download markers: {e}")

    cluster_col = _detect_cluster_col(markers_df)
    expr_cols = [c for c in markers_df.columns if c.startswith("expr_")]
    if not expr_cols:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "markers_flat.parquet has no expr_* columns"
        )

    cluster_mask = markers_df[cluster_col].astype(str) == str(payload.cluster)
    cluster_cells = markers_df.loc[cluster_mask]
    if cluster_cells.empty:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Cluster '{payload.cluster}' not found in run {payload.run_id}",
        )

    def work(progress_cb):
        progress_cb(10, "Computing mean expression for selected cluster")
        gene_names = [c.replace("expr_", "") for c in expr_cols]
        mean_expression = cluster_cells[expr_cols].mean().values.tolist()

        progress_cb(30, f"Calling scGPT perturbation endpoint ({payload.perturbation_type})")
        result_df = scgpt_service.predict_perturbation(
            expression=mean_expression,
            gene_names=gene_names,
            genes_to_perturb=payload.genes_to_perturb,
            perturbation_type=payload.perturbation_type,
        )

        if result_df.empty:
            progress_cb(100, "scGPT returned no rows")
            return {
                "results": [],
                "summary_total_genes": 0,
                "summary_max_abs_delta": 0.0,
                "summary_significant_count": 0,
            }

        progress_cb(85, f"Ranking {len(result_df)} genes by |Δ expression|")
        if "abs_delta" in result_df.columns:
            result_df = result_df.sort_values("abs_delta", ascending=False)
            threshold = result_df["abs_delta"].quantile(0.95)
            significant = int((result_df["abs_delta"] > threshold).sum())
            max_abs = float(result_df["abs_delta"].max())
        else:
            significant = 0
            max_abs = 0.0

        rows: list[dict] = []
        for _, r in result_df.head(500).iterrows():
            rows.append({
                "gene_name": str(r.get("gene_name", "")),
                "original_expression": _safe_float(r.get("original_expression")),
                "predicted_expression": _safe_float(r.get("predicted_expression")),
                "delta": _safe_float(r.get("delta")),
                "abs_delta": _safe_float(r.get("abs_delta")),
            })
        progress_cb(100, f"Returning top {len(rows)} genes")
        return {
            "results": rows,
            "summary_total_genes": int(len(result_df)),
            "summary_max_abs_delta": max_abs,
            "summary_significant_count": significant,
        }

    return StreamingResponse(
        stream_with_progress(work),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _safe_float(value) -> float | None:
    try:
        f = float(value)
        return f if f == f else None  # filter NaN
    except (TypeError, ValueError):
        return None


# ─── UMAP color-points ──────────────────────────────────────────────────────


class ColorPointsRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    color_column: str
    max_points: int = Field(10_000, ge=100, le=50_000)


class ColorPointsResponse(BaseModel):
    is_categorical: bool
    umap_0: list[float]
    umap_1: list[float]
    values_str: list[str] | None
    values_num: list[float] | None


@router.post("/run-color-points", response_model=ColorPointsResponse)
def run_color_points(payload: ColorPointsRequest, _: CurrentUserDep) -> ColorPointsResponse:
    import math
    import pandas as pd

    try:
        df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not download markers: {e}")
    if "UMAP_0" not in df.columns or "UMAP_1" not in df.columns:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Run has no UMAP_0/UMAP_1 columns")
    if payload.color_column not in df.columns:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Column '{payload.color_column}' not present in markers_flat",
        )

    if len(df) > payload.max_points:
        df = df.sample(n=payload.max_points, random_state=42)

    is_cat = df[payload.color_column].dtype == "object" or pd.api.types.is_categorical_dtype(
        df[payload.color_column]
    )
    umap_0: list[float] = []
    umap_1: list[float] = []
    values_str: list[str] | None = [] if is_cat else None
    values_num: list[float] | None = None if is_cat else []
    for _, row in df.iterrows():
        try:
            x = float(row["UMAP_0"])
            y = float(row["UMAP_1"])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if is_cat:
            values_str.append(str(row[payload.color_column]))
        else:
            try:
                v = float(row[payload.color_column])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            values_num.append(v)
        umap_0.append(x)
        umap_1.append(y)

    return ColorPointsResponse(
        is_categorical=is_cat,
        umap_0=umap_0,
        umap_1=umap_1,
        values_str=values_str,
        values_num=values_num,
    )


# ─── Marker Genes dotplot ───────────────────────────────────────────────────


class DotplotRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    n_top_genes_per_cluster: int = Field(3, ge=1, le=20)
    selected_genes: list[str] | None = None
    scale_data: bool = True


class DotplotCell(BaseModel):
    cluster: str
    gene: str
    expression: float
    size: float


class DotplotResponse(BaseModel):
    cells: list[DotplotCell]
    color_label: str
    color_scale: str
    clusters: list[str]
    genes: list[str]


@router.post("/run-dotplot", response_model=DotplotResponse)
def run_dotplot(payload: DotplotRequest, _: CurrentUserDep) -> DotplotResponse:
    import math
    import pandas as pd

    try:
        df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not download markers: {e}")

    cluster_col = _detect_cluster_col(df)
    expr_cols = [c for c in df.columns if c.startswith("expr_")]
    if not expr_cols:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No expr_* columns found")

    # Choose genes
    if payload.selected_genes:
        genes_ordered = payload.selected_genes
        expr_cols_to_plot = [f"expr_{g}" for g in genes_ordered if f"expr_{g}" in df.columns]
        genes_ordered = [c.replace("expr_", "") for c in expr_cols_to_plot]
    else:
        # Top N genes per cluster by z-score (fallback when scanpy result is missing)
        mean_expr = df.groupby(cluster_col)[expr_cols].mean()
        mean_z = (mean_expr - mean_expr.mean()) / mean_expr.std()
        ordered: list[str] = []
        for cl in sorted(mean_z.index, key=lambda x: int(str(x)) if str(x).isdigit() else str(x)):
            top = mean_z.loc[cl].nlargest(payload.n_top_genes_per_cluster).index.tolist()
            ordered.extend([g.replace("expr_", "") for g in top])
        genes_ordered = list(dict.fromkeys(ordered))
        expr_cols_to_plot = [f"expr_{g}" for g in genes_ordered]

    heatmap = df.groupby(cluster_col)[expr_cols_to_plot].mean()
    heatmap.columns = [c.replace("expr_", "") for c in heatmap.columns]
    if payload.scale_data:
        heatmap = (heatmap - heatmap.mean()) / heatmap.std()
        color_label = "Z-score"
        color_scale = "RdBu_r"
    else:
        color_label = "Mean expression"
        color_scale = "Viridis"

    clusters = sorted(
        [str(c) for c in heatmap.index],
        key=lambda x: int(x) if x.isdigit() else x,
    )
    cells: list[DotplotCell] = []
    for cl in clusters:
        for gene in genes_ordered:
            try:
                v = float(heatmap.loc[cl if cl in heatmap.index else int(cl), gene])
            except Exception:
                v = float("nan")
            if not math.isfinite(v):
                continue
            cells.append(
                DotplotCell(
                    cluster=cl,
                    gene=gene,
                    expression=v,
                    size=abs(v) if payload.scale_data else max(v, 0.0),
                )
            )

    return DotplotResponse(
        cells=cells,
        color_label=color_label,
        color_scale=color_scale,
        clusters=clusters,
        genes=genes_ordered,
    )


# ─── Differential Expression (Mann-Whitney) ────────────────────────────────


class DERequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    cluster_a: str
    cluster_b: str


class DEGene(BaseModel):
    gene: str
    log2fc: float
    p_value: float
    p_adj: float
    neg_log10_p_adj: float
    mean_a: float
    mean_b: float
    significant: bool


class DEResponse(BaseModel):
    genes: list[DEGene]
    n_significant: int


@router.post("/run-de", response_model=DEResponse)
def run_de(payload: DERequest, _: CurrentUserDep) -> DEResponse:
    import math

    import numpy as np

    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "scipy not installed in backend image",
        )

    try:
        df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not download markers: {e}")

    cluster_col = _detect_cluster_col(df)
    expr_cols = [c for c in df.columns if c.startswith("expr_")]
    if not expr_cols:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No expr_* columns found")
    if payload.cluster_a == payload.cluster_b:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Pick two different clusters")

    a_cells = df[df[cluster_col].astype(str) == str(payload.cluster_a)]
    b_cells = df[df[cluster_col].astype(str) == str(payload.cluster_b)]
    if a_cells.empty or b_cells.empty:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"No cells found for one of the clusters ({payload.cluster_a} / {payload.cluster_b})",
        )

    rows: list[dict] = []
    for col in expr_cols:
        gene = col.replace("expr_", "")
        va = a_cells[col].values
        vb = b_cells[col].values
        mean_a = float(va.mean())
        mean_b = float(vb.mean())
        log2fc = float(np.log2((mean_a + 1e-9) / (mean_b + 1e-9)))
        try:
            _, pval = mannwhitneyu(va, vb, alternative="two-sided")
        except ValueError:
            pval = 1.0
        rows.append(
            {
                "Gene": gene,
                "log2FC": log2fc,
                "p_value": float(pval),
                "Mean A": mean_a,
                "Mean B": mean_b,
            }
        )

    # Benjamini-Hochberg-ish p-value adjustment
    rows.sort(key=lambda r: r["p_value"])
    n = len(rows)
    out: list[DEGene] = []
    n_sig = 0
    for i, r in enumerate(rows, start=1):
        p_adj = min(r["p_value"] * n / i, 1.0)
        neg = -math.log10(max(p_adj, 1e-300))
        significant = (p_adj < 0.05) and (abs(r["log2FC"]) > 1)
        if significant:
            n_sig += 1
        out.append(
            DEGene(
                gene=r["Gene"],
                log2fc=r["log2FC"],
                p_value=r["p_value"],
                p_adj=p_adj,
                neg_log10_p_adj=neg,
                mean_a=r["Mean A"],
                mean_b=r["Mean B"],
                significant=significant,
            )
        )
    return DEResponse(genes=out, n_significant=n_sig)


# ─── Pathway Enrichment (GMT + Fisher's exact) ─────────────────────────────


GMT_DEFAULT_DBS = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023",
]


class EnrichmentRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    cluster: str
    dbs: list[str] = Field(default_factory=lambda: ["GO_Biological_Process_2023"])
    top_genes_for_cluster: int = Field(50, ge=10, le=500)


class EnrichmentTerm(BaseModel):
    term: str
    overlap: str
    p_value: float
    p_adj: float
    neg_log10_p_adj: float
    genes: str
    gene_set: str


class EnrichmentResponse(BaseModel):
    terms: list[EnrichmentTerm]
    available_dbs: list[str]


def _load_gmt(path: str) -> dict[str, set[str]]:
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    response = w.files.download(path)
    content = response.contents.read().decode("utf-8")
    out: dict[str, set[str]] = {}
    for line in content.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        term = parts[0]
        genes = {g for g in parts[2:] if g}
        if genes:
            out[term] = genes
    return out


@router.post("/run-enrichment", response_model=EnrichmentResponse)
def run_enrichment(payload: EnrichmentRequest, _: CurrentUserDep) -> EnrichmentResponse:
    import logging
    import math
    import traceback

    logger = logging.getLogger(__name__)

    try:
        from scipy.stats import fisher_exact
    except ImportError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "scipy not installed in backend image"
        )

    try:
        df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not download markers: {e}")

    cluster_col = _detect_cluster_col(df)
    expr_cols = [c for c in df.columns if c.startswith("expr_")]
    if not expr_cols:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No expr_* columns found")

    cluster_cells = df[df[cluster_col].astype(str) == str(payload.cluster)]
    if cluster_cells.empty:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"Cluster '{payload.cluster}' not found"
        )

    # Run the rest under a single try so any unexpected error surfaces with a
    # readable message in the HTTP response (was silently 500'ing before).
    try:
        # Pull the per-cluster Wilcoxon markers from MLflow — matches the
        # Default pipeline. Top-by-mean-expression (the old approach) was
        # dominated by housekeeping genes, giving identical enrichment hits
        # across clusters.
        try:
            markers_mapping = runs_service.download_cluster_markers_mapping(payload.run_id)
            cl_str = str(payload.cluster)
            if cl_str in markers_mapping.columns:
                col = markers_mapping[cl_str]
            else:
                # Fall back to first column when the cluster key isn't a
                # direct match — same convention used by the upstream gene-symbol matcher.
                col = markers_mapping.iloc[:, 0]
            gene_list = col.dropna().astype(str).head(payload.top_genes_for_cluster).tolist()
        except Exception as e:
            logger.warning(
                "top_markers_per_cluster.csv unavailable for run %s (%s); "
                "falling back to top-by-mean-expression",
                payload.run_id,
                e,
            )
            cluster_mean = cluster_cells[expr_cols].mean().nlargest(payload.top_genes_for_cluster)
            gene_list = [c.replace("expr_", "") for c in cluster_mean.index]
        background_genes = {c.replace("expr_", "") for c in expr_cols}

        s = get_settings()
        gmt_dir = f"/Volumes/{s.catalog}/{s.schema}/scanpy_reference/genesets"

        all_terms: list[EnrichmentTerm] = []
        available: list[str] = []
        gmt_errors: list[str] = []
        for db_name in payload.dbs:
            gmt_path = f"{gmt_dir}/{db_name}.gmt"
            try:
                gmt = _load_gmt(gmt_path)
            except Exception as e:
                gmt_errors.append(f"{db_name}: {e}")
                logger.warning("GMT load failed for %s: %s", db_name, e)
                continue
            available.append(db_name)

            query = set(gene_list) & background_genes
            bg_size = len(background_genes)
            n_query = len(query)
            if n_query == 0:
                continue

            rows: list[dict] = []
            for term, term_genes in gmt.items():
                term_in_bg = term_genes & background_genes
                if not term_in_bg:
                    continue
                overlap = query & term_in_bg
                if not overlap:
                    continue
                a = len(overlap)
                b = n_query - a
                c = len(term_in_bg) - a
                d = bg_size - a - b - c
                try:
                    _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
                except Exception:
                    pval = 1.0
                rows.append(
                    {
                        "Term": term,
                        "Overlap": f"{a}/{len(term_in_bg)}",
                        "P-value": float(pval),
                        "Genes": ";".join(sorted(overlap)),
                        "Gene_set": db_name,
                    }
                )

            rows.sort(key=lambda r: r["P-value"])
            n = len(rows)
            for i, r in enumerate(rows, start=1):
                p_adj = min(r["P-value"] * n / i, 1.0)
                all_terms.append(
                    EnrichmentTerm(
                        term=r["Term"],
                        overlap=r["Overlap"],
                        p_value=r["P-value"],
                        p_adj=p_adj,
                        neg_log10_p_adj=-math.log10(max(p_adj, 1e-300)),
                        genes=r["Genes"],
                        gene_set=r["Gene_set"],
                    )
                )

        # If every DB failed to load, surface that clearly.
        if not available and gmt_errors:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                "All gene-set DBs failed to load: " + "; ".join(gmt_errors),
            )

        all_terms.sort(key=lambda t: t.p_adj)
        return EnrichmentResponse(terms=all_terms, available_dbs=available)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Enrichment failed for run_id=%s cluster=%s: %s\n%s",
            payload.run_id,
            payload.cluster,
            e,
            traceback.format_exc(),
        )
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Enrichment failed: {type(e).__name__}: {e}",
        )


# ─── Trajectory (pseudotime) ───────────────────────────────────────────────


class TrajectoryRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    gene: str | None = None
    max_points: int = Field(10_000, ge=100, le=50_000)


class TrajectoryUmapPoint(BaseModel):
    umap_0: float
    umap_1: float
    pseudotime: float


class TrajectoryGenePoint(BaseModel):
    pseudotime: float
    expression: float


class TrajectoryResponse(BaseModel):
    has_pseudotime: bool
    umap_points: list[TrajectoryUmapPoint]
    gene_points: list[TrajectoryGenePoint]
    genes: list[str]


@router.post("/run-trajectory", response_model=TrajectoryResponse)
def run_trajectory(payload: TrajectoryRequest, _: CurrentUserDep) -> TrajectoryResponse:
    import math

    try:
        df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not download markers: {e}")

    if "dpt_pseudotime" not in df.columns:
        return TrajectoryResponse(has_pseudotime=False, umap_points=[], gene_points=[], genes=[])

    expr_cols = [c for c in df.columns if c.startswith("expr_")]
    genes = [c.replace("expr_", "") for c in expr_cols]

    sampled = df.sample(n=min(payload.max_points, len(df)), random_state=42)
    umap_points: list[TrajectoryUmapPoint] = []
    if "UMAP_0" in df.columns and "UMAP_1" in df.columns:
        for _, row in sampled.iterrows():
            try:
                x = float(row["UMAP_0"])
                y = float(row["UMAP_1"])
                t = float(row["dpt_pseudotime"])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(t)):
                continue
            umap_points.append(TrajectoryUmapPoint(umap_0=x, umap_1=y, pseudotime=t))

    gene_points: list[TrajectoryGenePoint] = []
    if payload.gene:
        gene_col = f"expr_{payload.gene}"
        if gene_col not in df.columns:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Gene '{payload.gene}' not present in markers_flat",
            )
        sub = df[["dpt_pseudotime", gene_col]].dropna().sort_values("dpt_pseudotime")
        if len(sub) > payload.max_points:
            sub = sub.sample(n=payload.max_points, random_state=42).sort_values("dpt_pseudotime")
        for _, row in sub.iterrows():
            try:
                t = float(row["dpt_pseudotime"])
                v = float(row[gene_col])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(t) and math.isfinite(v)):
                continue
            gene_points.append(TrajectoryGenePoint(pseudotime=t, expression=v))

    return TrajectoryResponse(
        has_pseudotime=True,
        umap_points=umap_points,
        gene_points=gene_points,
        genes=sorted(genes),
    )


# ─── Raw data table ────────────────────────────────────────────────────────


class RawDataRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    columns: list[str] = Field(default_factory=list)
    limit: int = Field(100, ge=1, le=2_000)


class RawDataResponse(BaseModel):
    columns: list[str]
    rows: list[dict]
    total_cells: int


@router.post("/run-rawdata", response_model=RawDataResponse)
def run_rawdata(payload: RawDataRequest, _: CurrentUserDep) -> RawDataResponse:
    import math

    try:
        df = runs_service.download_markers_df(payload.run_id)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not download markers: {e}")

    cols = [c for c in payload.columns if c in df.columns]
    if not cols:
        cluster_col = _detect_cluster_col(df)
        cols = [cluster_col]
        if "UMAP_0" in df.columns:
            cols += ["UMAP_0", "UMAP_1"]
        expr_cols = [c for c in df.columns if c.startswith("expr_")]
        cols += expr_cols[:3]

    sub = df[cols].head(payload.limit)
    rows: list[dict] = []
    for _, r in sub.iterrows():
        d: dict = {}
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                d[c] = v if math.isfinite(v) else None
            else:
                try:
                    d[c] = v.item() if hasattr(v, "item") else v
                except Exception:
                    d[c] = str(v)
        rows.append(d)

    return RawDataResponse(columns=cols, rows=rows, total_cells=int(len(df)))

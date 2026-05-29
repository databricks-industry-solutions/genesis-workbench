"""Disease Biology workflow routes — async-job dispatchers + MLflow result
loaders for Variant Calling, GWAS, VCF Ingestion, Variant Annotation."""
from __future__ import annotations

import logging
import math
import os
from typing import Optional

from databricks.sdk import WorkspaceClient
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.auth import CurrentUserDep
from app.routers.protein import _build_user_info
from app.services import disease_biology as db
from app.services import workbench
from app.services.databricks_links import dashboard_embed_url, job_run_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/disease_biology", tags=["disease_biology"])


# ─── Form defaults (sourced from env vars set during deploy) ──────────────


class VariantCallingDefaults(BaseModel):
    fastq_r1: str = ""
    fastq_r2: str = ""
    reference_genome_path: str = ""
    output_volume_path: str = ""


class GwasDefaults(BaseModel):
    vcf_path: str = ""
    phenotype_path: str = ""
    phenotype_column: str = "phenotype"
    contigs: str = "6"
    hwe_cutoff: str = "0.01"
    pvalue_threshold: str = "0.01"


class VcfIngestionDefaults(BaseModel):
    vcf_path: str = ""


class DiseaseBiologyDefaultsResponse(BaseModel):
    variant_calling: VariantCallingDefaults
    gwas: GwasDefaults
    vcf_ingestion: VcfIngestionDefaults


@router.get("/defaults", response_model=DiseaseBiologyDefaultsResponse)
def disease_biology_defaults(_: CurrentUserDep) -> DiseaseBiologyDefaultsResponse:
    """Form defaults — paths to the sample datasets pre-staged by the
    `disease_biology/` deploy. The deploy notebooks write everything under
    deterministic per-catalog/schema paths, so we compute them here rather
    than relying on per-sample-path env vars (the Streamlit form reads vars
    like `SAMPLE_FASTQ_R1` that the disease_biology deploy never sets,
    which is why the form lands empty for new installs)."""
    catalog = os.environ.get("CORE_CATALOG_NAME", "catalog")
    schema = os.environ.get("CORE_SCHEMA_NAME", "schema")
    base = f"/Volumes/{catalog}/{schema}"
    return DiseaseBiologyDefaultsResponse(
        variant_calling=VariantCallingDefaults(
            fastq_r1=f"{base}/gwas_data/sample_fastq/sample_1.fq.gz",
            fastq_r2=f"{base}/gwas_data/sample_fastq/sample_2.fq.gz",
            reference_genome_path=(
                f"{base}/gwas_reference/genomes/"
                "GRCh38_full_analysis_set_plus_decoy_hla.fa"
            ),
            output_volume_path=f"{base}/gwas_data",
        ),
        gwas=GwasDefaults(
            vcf_path=(
                f"{base}/gwas_data/sample_vcf/"
                "ALL.chr6.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"
            ),
            phenotype_path=f"{base}/gwas_data/sample_phenotype/breast_cancer_phenotype.tsv",
        ),
        vcf_ingestion=VcfIngestionDefaults(
            vcf_path=f"{base}/variant_annotation_data/sample/brca_pathogenic_corrected.vcf",
        ),
    )


# ─── Shared shapes ─────────────────────────────────────────────────────────


class DBRunRow(BaseModel):
    run_id: str
    run_name: str
    experiment_name: str
    status: str
    progress: str
    start_time_ms: Optional[int] = None
    detail: str
    # Workspace UI link to the dispatched job's run page. Empty string when
    # the run lacks a job_run_id tag.
    run_url: str = ""


class DBSearchResponse(BaseModel):
    runs: list[DBRunRow]


class JobDispatchResponse(BaseModel):
    job_run_id: int
    run_url: str


def _run_url(job_id: int | str, job_run_id: int) -> str:
    return job_run_url(job_id, job_run_id)


def _runs_response(rows: list[db.DBRun]) -> DBSearchResponse:
    return DBSearchResponse(
        runs=[
            DBRunRow(
                run_id=r.run_id,
                run_name=r.run_name,
                experiment_name=r.experiment_name,
                status=r.status,
                progress=r.progress,
                start_time_ms=r.start_time_ms,
                detail=r.detail,
                run_url=r.run_url,
            )
            for r in rows
        ]
    )


# ─── Variant Calling (Parabricks alignment) ────────────────────────────────


class VariantCallingStartRequest(BaseModel):
    fastq_r1: str = Field(..., min_length=1)
    fastq_r2: str = Field(..., min_length=1)
    reference_genome_path: str = Field(..., min_length=1)
    output_volume_path: str = Field(..., min_length=1)
    mlflow_experiment: str = Field("gwb_variant_calling", min_length=1)
    mlflow_run_name: str = Field(..., min_length=1)


@router.post("/variant_calling/start", response_model=JobDispatchResponse)
def variant_calling_start(
    payload: VariantCallingStartRequest, user: CurrentUserDep
) -> JobDispatchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    user_info = _build_user_info(user, WorkspaceClient())
    try:
        job_run_id = db.start_parabricks_alignment(
            user_info=user_info,
            fastq_r1=payload.fastq_r1,
            fastq_r2=payload.fastq_r2,
            reference_genome_path=payload.reference_genome_path,
            output_volume_path=payload.output_volume_path,
            mlflow_experiment_name=payload.mlflow_experiment,
            mlflow_run_name=payload.mlflow_run_name,
        )
    except Exception as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Failed to dispatch Parabricks alignment job: {e}",
        )
    return JobDispatchResponse(
        job_run_id=job_run_id,
        run_url=_run_url(workbench.get_job_id("parabricks_alignment_job_id"), job_run_id),
    )


@router.get("/variant_calling/search", response_model=DBSearchResponse)
def variant_calling_search(
    user: CurrentUserDep,
    by: str = Query("run_name", pattern=r"^(run_name|experiment_name)$"),
    text: str = Query(..., min_length=1),
) -> DBSearchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    return _runs_response(db.search_variant_calling(user.email, by, text.strip()))


class VariantCallingPickerRow(BaseModel):
    run_id: str
    run_name: str
    experiment_name: str
    output_vcf: str
    start_time_ms: Optional[int] = None


class VariantCallingPickerResponse(BaseModel):
    runs: list[VariantCallingPickerRow]


@router.get("/variant_calling/successful", response_model=VariantCallingPickerResponse)
def variant_calling_successful(user: CurrentUserDep) -> VariantCallingPickerResponse:
    """Successful (alignment_complete) runs — input picker for the GWAS form."""
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    return VariantCallingPickerResponse(
        runs=[VariantCallingPickerRow(**r) for r in db.list_successful_variant_calling(user.email)]
    )


# ─── GWAS ──────────────────────────────────────────────────────────────────


class GwasStartRequest(BaseModel):
    vcf_path: str = Field(..., min_length=1)
    phenotype_path: str = Field(..., min_length=1)
    phenotype_column: str = Field(..., min_length=1)
    contigs: str = Field("", description="Comma-separated chromosome list; empty = all")
    hwe_cutoff: str = Field("1e-6")
    pvalue_threshold: str = Field("5e-8")
    mlflow_experiment: str = Field("gwb_gwas", min_length=1)
    mlflow_run_name: str = Field(..., min_length=1)


@router.post("/gwas/start", response_model=JobDispatchResponse)
def gwas_start(payload: GwasStartRequest, user: CurrentUserDep) -> JobDispatchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    user_info = _build_user_info(user, WorkspaceClient())
    try:
        job_run_id = db.start_gwas_analysis(
            user_info=user_info,
            vcf_path=payload.vcf_path,
            phenotype_path=payload.phenotype_path,
            phenotype_column=payload.phenotype_column,
            contigs=payload.contigs,
            hwe_cutoff=payload.hwe_cutoff,
            pvalue_threshold=payload.pvalue_threshold,
            mlflow_experiment_name=payload.mlflow_experiment,
            mlflow_run_name=payload.mlflow_run_name,
        )
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Failed to dispatch GWAS job: {e}")
    return JobDispatchResponse(
        job_run_id=job_run_id,
        run_url=_run_url(workbench.get_job_id("gwas_analysis_job_id"), job_run_id),
    )


@router.get("/gwas/search", response_model=DBSearchResponse)
def gwas_search(
    user: CurrentUserDep,
    by: str = Query("run_name", pattern=r"^(run_name|experiment_name)$"),
    text: str = Query(..., min_length=1),
) -> DBSearchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    return _runs_response(db.search_gwas(user.email, by, text.strip()))


class GwasHit(BaseModel):
    contig: str
    position: int
    pvalue: float
    neg_log_pval: Optional[float] = None
    reference_allele: str
    alternate_alleles: str
    effect: Optional[float] = None
    phenotype: Optional[str] = None


class GwasResultsResponse(BaseModel):
    total_variants: int
    significant_count: int
    min_pvalue: Optional[float] = None
    # Top 50 by pvalue for the table.
    top_hits: list[GwasHit]
    # All non-null rows downsampled for the Manhattan scatter — (start, neg_log_pval).
    manhattan_points: list[dict]


@router.get("/gwas/results", response_model=GwasResultsResponse)
def gwas_results(user: CurrentUserDep, run_id: str = Query(..., min_length=1)) -> GwasResultsResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    df = db.pull_gwas_results(run_id)
    if df.empty:
        return GwasResultsResponse(
            total_variants=0,
            significant_count=0,
            min_pvalue=None,
            top_hits=[],
            manhattan_points=[],
        )
    total = int(len(df))
    sig = int((df["pvalue"] < 5e-8).sum())
    min_p = float(df["pvalue"].min())
    top = df.head(50)
    top_hits = [
        GwasHit(
            contig=str(r["contigName"]),
            position=int(r["start"]),
            pvalue=float(r["pvalue"]),
            neg_log_pval=(
                float(r["neg_log_pval"]) if r.get("neg_log_pval") is not None else None
            ),
            reference_allele=str(r["referenceAllele"]),
            alternate_alleles=str(r["alternateAlleles"]),
            effect=float(r["effect"]) if r.get("effect") is not None else None,
            phenotype=str(r["phenotype"]) if r.get("phenotype") else None,
        )
        for _, r in top.iterrows()
    ]
    # Manhattan scatter — drop nulls; if huge, downsample to keep payload light.
    manhattan = df[df["neg_log_pval"].notna()][["start", "neg_log_pval"]].copy()
    if len(manhattan) > 5000:
        manhattan = manhattan.sample(n=5000, random_state=0)
    manhattan_points = [
        {"x": int(r["start"]), "y": float(r["neg_log_pval"])}
        for _, r in manhattan.iterrows()
    ]
    return GwasResultsResponse(
        total_variants=total,
        significant_count=sig,
        min_pvalue=min_p,
        top_hits=top_hits,
        manhattan_points=manhattan_points,
    )


# ─── VCF Ingestion ─────────────────────────────────────────────────────────


class VcfIngestionStartRequest(BaseModel):
    vcf_path: str = Field(..., min_length=1)
    output_table_name: str = Field(..., min_length=1)
    mlflow_experiment: str = Field("gwb_vcf_ingestion", min_length=1)
    mlflow_run_name: str = Field(..., min_length=1)


@router.post("/vcf_ingestion/start", response_model=JobDispatchResponse)
def vcf_ingestion_start(
    payload: VcfIngestionStartRequest, user: CurrentUserDep
) -> JobDispatchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    user_info = _build_user_info(user, WorkspaceClient())
    try:
        job_run_id = db.start_vcf_ingestion(
            user_info=user_info,
            vcf_path=payload.vcf_path,
            output_table_name=payload.output_table_name,
            mlflow_experiment_name=payload.mlflow_experiment,
            mlflow_run_name=payload.mlflow_run_name,
        )
    except Exception as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Failed to dispatch VCF ingestion job: {e}",
        )
    return JobDispatchResponse(
        job_run_id=job_run_id,
        run_url=_run_url(workbench.get_job_id("vcf_ingestion_job_id"), job_run_id),
    )


@router.get("/vcf_ingestion/search", response_model=DBSearchResponse)
def vcf_ingestion_search(
    user: CurrentUserDep,
    by: str = Query("run_name", pattern=r"^(run_name|experiment_name)$"),
    text: str = Query(..., min_length=1),
) -> DBSearchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    return _runs_response(db.search_vcf_ingestion(user.email, by, text.strip()))


class VcfIngestionPickerRow(BaseModel):
    run_id: str
    run_name: str
    experiment_name: str
    output_table: str
    start_time_ms: Optional[int] = None


class VcfIngestionPickerResponse(BaseModel):
    runs: list[VcfIngestionPickerRow]


@router.get("/vcf_ingestion/successful", response_model=VcfIngestionPickerResponse)
def vcf_ingestion_successful(user: CurrentUserDep) -> VcfIngestionPickerResponse:
    """Successful (ingestion_complete) runs — input picker for the Variant
    Annotation form."""
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    return VcfIngestionPickerResponse(
        runs=[VcfIngestionPickerRow(**r) for r in db.list_successful_vcf_ingestion(user.email)]
    )


# ─── Variant Annotation ────────────────────────────────────────────────────


class VariantAnnotationStartRequest(BaseModel):
    variants_table: str = Field(..., min_length=1)
    gene_regions: str = Field("")
    # Empty string is fine — the orchestrator uses ClinVar by default and
    # the input was only kept for early-deployment testing.
    pathogenic_vcf_path: str = Field("")
    gene_panel_mode: str = Field("custom", pattern=r"^(custom|acmg)$")
    mlflow_experiment: str = Field("gwb_variant_annotation", min_length=1)
    mlflow_run_name: str = Field(..., min_length=1)


@router.post("/variant_annotation/start", response_model=JobDispatchResponse)
def variant_annotation_start(
    payload: VariantAnnotationStartRequest, user: CurrentUserDep
) -> JobDispatchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    user_info = _build_user_info(user, WorkspaceClient())
    try:
        job_run_id = db.start_variant_annotation(
            user_info=user_info,
            variants_table=payload.variants_table,
            gene_regions=payload.gene_regions,
            pathogenic_vcf_path=payload.pathogenic_vcf_path,
            mlflow_experiment_name=payload.mlflow_experiment,
            mlflow_run_name=payload.mlflow_run_name,
            gene_panel_mode=payload.gene_panel_mode,
        )
    except Exception as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Failed to dispatch variant annotation job: {e}",
        )
    return JobDispatchResponse(
        job_run_id=job_run_id,
        run_url=_run_url(workbench.get_job_id("variant_annotation_job_id"), job_run_id),
    )


@router.get("/variant_annotation/search", response_model=DBSearchResponse)
def variant_annotation_search(
    user: CurrentUserDep,
    by: str = Query("run_name", pattern=r"^(run_name|experiment_name)$"),
    text: str = Query(..., min_length=1),
) -> DBSearchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    return _runs_response(db.search_variant_annotation(user.email, by, text.strip()))


class AnnotationVariant(BaseModel):
    gene: str
    chromosome: str
    position: int
    ref: str
    alt: str
    zygosity: Optional[str] = None
    clinical_significance: Optional[str] = None
    disease_name: Optional[str] = None
    category: Optional[str] = None
    condition: Optional[str] = None


class VariantAnnotationResultsResponse(BaseModel):
    variants: list[AnnotationVariant]
    total: int


@router.get("/variant_annotation/results", response_model=VariantAnnotationResultsResponse)
def variant_annotation_results(
    user: CurrentUserDep, run_id: str = Query(..., min_length=1)
) -> VariantAnnotationResultsResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    df = db.pull_annotation_results(run_id)
    variants: list[AnnotationVariant] = []
    for _, r in df.iterrows():
        variants.append(
            AnnotationVariant(
                gene=str(r.get("gene", "")),
                chromosome=str(r.get("chromosome", "")),
                position=int(r.get("position", 0)),
                ref=str(r.get("ref", "")),
                alt=str(r.get("alt", "")),
                zygosity=str(r.get("zygosity", "")) if r.get("zygosity") is not None else None,
                clinical_significance=str(r.get("clinical_significance", ""))
                if r.get("clinical_significance") is not None else None,
                disease_name=str(r.get("disease_name", ""))
                if r.get("disease_name") is not None else None,
                category=str(r.get("category", "")) if "category" in df.columns else None,
                condition=str(r.get("condition", "")) if "condition" in df.columns else None,
            )
        )
    return VariantAnnotationResultsResponse(variants=variants, total=len(variants))


class VariantAnnotationDashboardResponse(BaseModel):
    embed_url: str
    run_name: str | None = None


@router.get("/variant_annotation/dashboard", response_model=VariantAnnotationDashboardResponse)
def variant_annotation_dashboard(
    _: CurrentUserDep, run_name: Optional[str] = Query(None)
) -> VariantAnnotationDashboardResponse:
    dash_id = workbench.get_app_setting("variant_annotation_dashboard_id")
    params = {"run_name": run_name} if run_name else None
    return VariantAnnotationDashboardResponse(
        embed_url=dashboard_embed_url(dash_id, params),
        run_name=run_name,
    )


# ─── Run details (shared View dialogs) ────────────────────────────────────


class RunDetailsResponse(BaseModel):
    run_name: str
    experiment_id: str
    status: str
    job_status: str
    job_run_id: str
    params: dict[str, str]
    tags: dict[str, str]


@router.get("/run/details", response_model=RunDetailsResponse)
def run_details(
    user: CurrentUserDep, run_id: str = Query(..., min_length=1)
) -> RunDetailsResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    try:
        details = db.get_run_details(run_id)
    except Exception as e:
        logger.exception("Failed to load run details for %s", run_id)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Could not load MLflow run {run_id}: {type(e).__name__}: {e}",
        )
    return RunDetailsResponse(
        run_name=details["run_name"],
        experiment_id=details["experiment_id"],
        status=details["status"],
        job_status=details["job_status"],
        job_run_id=str(details["job_run_id"]),
        params=details["params"],
        tags=details["tags"],
    )

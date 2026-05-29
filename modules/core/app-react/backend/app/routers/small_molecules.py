"""Small Molecules workflow routes. First port: DiffDock molecular docking."""
from __future__ import annotations

import logging
import os
from typing import Optional

import mlflow
from databricks.sdk import WorkspaceClient
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from genesis_workbench.models import set_mlflow_experiment
from pydantic import BaseModel, Field

from app.auth import CurrentUserDep
from app.routers.protein import _build_user_info
from app.services import admet_safety as admet_pipeline
from app.services import enzyme_optimization as enzyme_pipeline
from app.services import ligand_binder_design as ligand_pipeline
from app.services import molecular_docking as docking
from app.services import motif_scaffolding as motif_pipeline
from app.services import protein_binder_design as binder_pipeline
from app.services.molstar import molstar_html_multibody, molstar_html_singlebody
from app.services.sse import stream_with_progress

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/small_molecules", tags=["small_molecules"])

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class DockingExampleResponse(BaseModel):
    smiles: str
    pdb: str


@router.get("/diffdock/example", response_model=DockingExampleResponse)
def diffdock_example(_: CurrentUserDep) -> DockingExampleResponse:
    """Defaults for the docking form — server-side fetch of a 50-residue
    slice of chain A of PDB 6agt + a small ligand SMILES."""
    return DockingExampleResponse(
        smiles=docking.EXAMPLE_SMILES,
        pdb=docking.get_example_pdb(),
    )


class DockingRequest(BaseModel):
    protein_pdb: str = Field(..., min_length=10)
    ligand_smiles: str = Field(..., min_length=1)
    num_samples: int = Field(5, ge=1, le=20)
    mlflow_experiment: str = Field("gwb_molecular_docking", min_length=1)
    mlflow_run_name: str = Field(..., min_length=1)


class DockingPose(BaseModel):
    rank: int
    confidence: float
    ligand_sdf: str
    # Per-pose Mol* viewer HTML (protein + ligand HETATM overlay). Built
    # server-side so the frontend can flip through poses without round-
    # tripping the protein/ligand to a render endpoint per click.
    viewer_html: str
    error: Optional[str] = None


class DockingResponse(BaseModel):
    poses: list[DockingPose]
    experiment_id: str
    run_id: str
    n_success: int


@router.post("/diffdock/stream")
def diffdock_stream(payload: DockingRequest, user: CurrentUserDep):
    """SSE variant of DiffDock docking. Two real phases (ESM embed →
    DiffDock score), then server-side Mol* viewer builds, then a terminal
    `result` matching DockingResponse."""
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    if not payload.protein_pdb.strip() or not payload.ligand_smiles.strip():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Both protein PDB and ligand SMILES are required"
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
                "ligand_smiles": payload.ligand_smiles,
                "num_samples": payload.num_samples,
            })

            results_df = docking.hit_diffdock(
                protein_pdb=payload.protein_pdb,
                ligand_smiles=payload.ligand_smiles,
                samples_per_complex=payload.num_samples,
                progress_callback=progress_cb,
            )

            if results_df.empty:
                progress_cb(100, "DiffDock returned no poses")
                return {
                    "poses": [],
                    "experiment_id": str(experiment.experiment_id),
                    "run_id": mlflow_run_id,
                    "n_success": 0,
                }

            progress_cb(88, f"Logging {len(results_df)} pose ranks to MLflow")
            mlflow.log_dict(
                results_df[["rank", "confidence"]].to_dict(), "diffdock_results.json"
            )

            progress_cb(90, "Building Mol* viewers per pose")
            poses: list[dict] = []
            n_success = 0
            for _, row in results_df.iterrows():
                ligand_sdf = str(row.get("ligand_sdf", ""))
                err_msg: str | None = None
                if ligand_sdf.startswith("ERROR"):
                    err_msg = ligand_sdf
                    viewer_html = molstar_html_multibody(
                        [payload.protein_pdb], names=["target"], with_iframe=False
                    )
                else:
                    hetatm = docking.sdf_to_hetatm(ligand_sdf)
                    ligand_pdb = hetatm + "\nEND\n" if hetatm else ""
                    pdbs = [payload.protein_pdb] + ([ligand_pdb] if ligand_pdb else [])
                    names = ["target"] + (["pose"] if ligand_pdb else [])
                    viewer_html = molstar_html_multibody(pdbs, names=names, with_iframe=False)
                    n_success += 1

                poses.append({
                    "rank": int(row.get("rank", 0)),
                    "confidence": float(row.get("confidence", 0.0)),
                    "ligand_sdf": ligand_sdf,
                    "viewer_html": viewer_html,
                    "error": err_msg,
                })

            progress_cb(100, f"Done — {n_success}/{len(poses)} successful pose(s)")
            return {
                "poses": poses,
                "experiment_id": str(experiment.experiment_id),
                "run_id": mlflow_run_id,
                "n_success": n_success,
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


# ─── Ligand Binder Design ─────────────────────────────────────────────────


class LigandBinderDesignRequest(BaseModel):
    # Exactly one of ligand_pdb / ligand_smiles must be set. SMILES is
    # routed through RDKit ETKDGv3 + MMFF94 on the server to get a 3D PDB.
    ligand_pdb: Optional[str] = None
    ligand_smiles: Optional[str] = None
    binder_length_min: int = Field(50, ge=20, le=200)
    binder_length_max: int = Field(80, ge=20, le=300)
    num_samples: int = Field(2, ge=1, le=10)
    validate_esmfold: bool = True
    # DiffDock validation needs a SMILES string regardless of input mode.
    validate_diffdock: bool = True
    mlflow_experiment: str = Field("gwb_ligand_binder_design", min_length=1)
    mlflow_run_name: str = Field(..., min_length=1)


class LigandBinderDesign(BaseModel):
    sample_id: str
    sequence: str
    rewards: float
    esmfold_validated: bool
    dock_confidence: Optional[float] = None
    # Pre-built Mol* HTMLs for the four view choices. None when the
    # underlying structure isn't available.
    viewer_html_ca_backbone: Optional[str] = None
    viewer_html_esmfold: Optional[str] = None
    viewer_html_ca_plus_dock: Optional[str] = None
    viewer_html_esmfold_plus_dock: Optional[str] = None


class LigandBinderDesignResponse(BaseModel):
    designs: list[LigandBinderDesign]
    ligand_pdb: str
    experiment_id: str
    run_id: str
    warnings: list[str]


@router.post("/ligand_binder_design/stream")
def ligand_binder_design_stream(payload: LigandBinderDesignRequest, user: CurrentUserDep):
    """SSE variant. Stages mirror the frontend's `RealtimeProgress`:
        0 →15  SMILES → PDB (RDKit, skipped if ligand_pdb supplied)
       15 →35  Proteina-Complexa-Ligand binder generation
       35 →60  ESMFold validation per design (optional)
       60 →90  DiffDock validation per design (optional)
       90 →100 Build Mol* viewers + log to MLflow"""
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    if not payload.ligand_pdb and not payload.ligand_smiles:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Provide either ligand_pdb or ligand_smiles"
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
                "ligand_smiles": payload.ligand_smiles or "",
                "binder_len_min": payload.binder_length_min,
                "binder_len_max": payload.binder_length_max,
                "num_samples": payload.num_samples,
                "validate_esmfold": payload.validate_esmfold,
                "validate_diffdock": payload.validate_diffdock,
                "input_mode": "PDB" if payload.ligand_pdb else "SMILES",
            })

            result = ligand_pipeline.run_ligand_binder_design(
                ligand_pdb=payload.ligand_pdb,
                ligand_smiles=payload.ligand_smiles,
                binder_length_min=payload.binder_length_min,
                binder_length_max=payload.binder_length_max,
                num_samples=payload.num_samples,
                validate_esmfold=payload.validate_esmfold,
                validate_diffdock=payload.validate_diffdock,
                progress_callback=progress_cb,
            )

            ligand_pdb = result["ligand_pdb"]
            designs_raw = result["designs"]
            warnings = result["warnings"]

            mlflow.log_dict(
                {
                    "designs": [
                        {
                            "sample_id": d["sample_id"],
                            "sequence": d["sequence"],
                            "rewards": d["rewards"],
                            "dock_confidence": d.get("dock_confidence"),
                        }
                        for d in designs_raw
                    ],
                },
                "proteina_complexa_ligand_results.json",
            )

            progress_cb(92, "Building Mol* viewers")
            designs: list[dict] = []
            for d in designs_raw:
                ca_pdb = d.get("pdb_output")
                esmfold_pdb = d.get("esmfold_pdb")
                dock_sdf = d.get("best_dock_sdf")
                lig_hetatm = docking.sdf_to_hetatm(dock_sdf) if dock_sdf else ""
                lig_pdb_block = (lig_hetatm + "\nEND\n") if lig_hetatm else ""

                viewer_ca = (
                    molstar_html_multibody([ca_pdb], names=["backbone"], with_iframe=False)
                    if ca_pdb
                    else None
                )
                viewer_esmfold = (
                    molstar_html_multibody(
                        [esmfold_pdb], names=["esmfold"], with_iframe=False
                    )
                    if esmfold_pdb
                    else None
                )
                viewer_ca_dock = (
                    molstar_html_multibody(
                        [ca_pdb, lig_pdb_block],
                        names=["backbone", "ligand"],
                        with_iframe=False,
                    )
                    if (ca_pdb and lig_pdb_block)
                    else None
                )
                viewer_esmfold_dock = (
                    molstar_html_multibody(
                        [esmfold_pdb, lig_pdb_block],
                        names=["esmfold", "ligand"],
                        with_iframe=False,
                    )
                    if (esmfold_pdb and lig_pdb_block)
                    else None
                )

                designs.append({
                    "sample_id": d["sample_id"],
                    "sequence": d["sequence"],
                    "rewards": d["rewards"],
                    "esmfold_validated": bool(esmfold_pdb),
                    "dock_confidence": d.get("dock_confidence"),
                    "viewer_html_ca_backbone": viewer_ca,
                    "viewer_html_esmfold": viewer_esmfold,
                    "viewer_html_ca_plus_dock": viewer_ca_dock,
                    "viewer_html_esmfold_plus_dock": viewer_esmfold_dock,
                })

            progress_cb(100, f"Done — {len(designs)} design(s) ready")
            return {
                "designs": designs,
                "ligand_pdb": ligand_pdb,
                "experiment_id": str(experiment.experiment_id),
                "run_id": mlflow_run_id,
                "warnings": warnings,
            }

    return StreamingResponse(
        stream_with_progress(work),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ─── Motif Scaffolding ────────────────────────────────────────────────────


class MotifScaffoldingRequest(BaseModel):
    motif_pdb: str = Field(..., min_length=10)
    target_chain: str = Field("B", min_length=1, max_length=4)
    scaffold_length_min: int = Field(50, ge=20, le=200)
    scaffold_length_max: int = Field(80, ge=20, le=300)
    num_samples: int = Field(2, ge=1, le=10)
    optimize_mpnn: bool = True
    validate_esmfold: bool = True
    mlflow_experiment: str = Field("gwb_motif_scaffolding", min_length=1)
    mlflow_run_name: str = Field(..., min_length=1)


class MotifScaffold(BaseModel):
    sample_id: str
    sequence: str
    mpnn_sequence: Optional[str] = None
    rewards: float
    esmfold_validated: bool
    # Single Mol* HTML overlaying motif + scaffold (ESMFold preferred,
    # CA-only backbone fallback).
    viewer_html: Optional[str] = None


class MotifScaffoldingResponse(BaseModel):
    scaffolds: list[MotifScaffold]
    motif_pdb: str
    experiment_id: str
    run_id: str
    warnings: list[str]


@router.post("/motif_scaffolding/stream")
def motif_scaffolding_stream(payload: MotifScaffoldingRequest, user: CurrentUserDep):
    """SSE variant. Stages:
        0 →35  Proteina-Complexa-AME scaffold generation
       35 →60  ProteinMPNN sequence optimisation (optional)
       60 →90  ESMFold validation per scaffold (optional)
       90 →100 Build Mol* viewers + log to MLflow"""
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
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
                "scaffold_len_min": payload.scaffold_length_min,
                "scaffold_len_max": payload.scaffold_length_max,
                "num_samples": payload.num_samples,
                "optimize_mpnn": payload.optimize_mpnn,
                "validate_esmfold": payload.validate_esmfold,
            })

            result = motif_pipeline.run_motif_scaffolding(
                motif_pdb=payload.motif_pdb,
                target_chain=payload.target_chain,
                scaffold_length_min=payload.scaffold_length_min,
                scaffold_length_max=payload.scaffold_length_max,
                num_samples=payload.num_samples,
                optimize_mpnn=payload.optimize_mpnn,
                validate_esmfold=payload.validate_esmfold,
                progress_callback=progress_cb,
            )

            motif_pdb = result["motif_pdb"]
            scaffolds_raw = result["scaffolds"]
            warnings = result["warnings"]

            mlflow.log_dict(
                {
                    "scaffolds": [
                        {
                            "sample_id": s["sample_id"],
                            "sequence": s["sequence"],
                            "mpnn_sequence": s.get("mpnn_sequence"),
                            "rewards": s["rewards"],
                            "esmfold_validated": s["esmfold_validated"],
                        }
                        for s in scaffolds_raw
                    ],
                },
                "proteina_complexa_ame_results.json",
            )

            progress_cb(92, "Building Mol* viewers")
            scaffolds: list[dict] = []
            for s in scaffolds_raw:
                display_pdb = s.get("esmfold_pdb") or s.get("pdb_output") or None
                viewer = (
                    molstar_html_multibody(
                        [motif_pdb, display_pdb],
                        names=["motif", "scaffold"],
                        with_iframe=False,
                    )
                    if display_pdb
                    else None
                )
                scaffolds.append({
                    "sample_id": s["sample_id"],
                    "sequence": s["sequence"],
                    "mpnn_sequence": s.get("mpnn_sequence"),
                    "rewards": s["rewards"],
                    "esmfold_validated": s["esmfold_validated"],
                    "viewer_html": viewer,
                })

            progress_cb(100, f"Done — {len(scaffolds)} scaffold(s) ready")
            return {
                "scaffolds": scaffolds,
                "motif_pdb": motif_pdb,
                "experiment_id": str(experiment.experiment_id),
                "run_id": mlflow_run_id,
                "warnings": warnings,
            }

    return StreamingResponse(
        stream_with_progress(work),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ─── ADMET & Safety ───────────────────────────────────────────────────────


class AdmetRequest(BaseModel):
    smiles: list[str] = Field(..., min_length=1)
    run_bbbp: bool = True
    run_clintox: bool = True
    run_admet: bool = True
    mlflow_experiment: str = Field("gwb_admet_safety", min_length=1)
    mlflow_run_name: str = Field(..., min_length=1)


class AdmetResponse(BaseModel):
    smiles: list[str]
    bbbp: Optional[list[Optional[float]]] = None
    clintox: Optional[list[Optional[float]]] = None
    # Per-molecule dict keyed by ADMET task name (e.g. "Caco2", "Lipophilicity", etc).
    admet: Optional[list[dict]] = None
    experiment_id: str
    run_id: str
    warnings: list[str]


@router.post("/admet/stream")
def admet_stream(payload: AdmetRequest, user: CurrentUserDep):
    """SSE variant. Each enabled predictor occupies one stage; the route
    only runs the ones the caller checked."""
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    clean = [s.strip() for s in payload.smiles if s and s.strip()]
    if not clean:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No SMILES provided")
    if not (payload.run_bbbp or payload.run_clintox or payload.run_admet):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Select at least one predictor (BBB, ClinTox, or ADMET)",
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
                "num_molecules": len(clean),
                "run_bbbp": payload.run_bbbp,
                "run_clintox": payload.run_clintox,
                "run_admet": payload.run_admet,
            })

            result = admet_pipeline.run_admet_profiling(
                smiles_list=clean,
                run_bbbp=payload.run_bbbp,
                run_clintox=payload.run_clintox,
                run_admet=payload.run_admet,
                progress_callback=progress_cb,
            )

            if "bbbp" in result:
                mlflow.log_dict({"bbbp_predictions": result["bbbp"]}, "bbbp_results.json")
            if "clintox" in result:
                mlflow.log_dict({"clintox_predictions": result["clintox"]}, "clintox_results.json")
            if "admet" in result:
                mlflow.log_dict({"admet_predictions": result["admet"]}, "admet_results.json")

            progress_cb(100, f"Done — profiled {len(clean)} molecule(s)")
            return {
                "smiles": result["smiles"],
                "bbbp": result.get("bbbp"),
                "clintox": result.get("clintox"),
                "admet": result.get("admet"),
                "experiment_id": str(experiment.experiment_id),
                "run_id": mlflow_run_id,
                "warnings": result["warnings"],
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

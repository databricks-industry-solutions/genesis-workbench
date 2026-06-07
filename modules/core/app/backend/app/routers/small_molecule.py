"""Small Molecule workflow routes. First port: DiffDock molecular docking."""
from __future__ import annotations

import logging
from typing import Optional

import mlflow
from databricks.sdk import WorkspaceClient
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from genesis_workbench.models import set_mlflow_experiment
from pydantic import BaseModel, Field

from app.auth import CurrentUserDep
from app.routers.large_molecule import _build_user_info
from app.services import admet_safety as admet_pipeline
from app.services import genmol as genmol_svc
from app.services import ligand_binder_design as ligand_pipeline
from app.services import molecule_optimization as mol_opt
from app.services import molecular_docking as docking
from app.services import motif_scaffolding as motif_pipeline
from app.services.molstar import molstar_html_multibody
from app.services.sse import stream_with_progress

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/small_molecule", tags=["small_molecule"])

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


# ── GenMol — generative small-molecule design ──────────────────────────────────

class GenMolGenerateRequest(BaseModel):
    # Each seed: "" => de novo; a SMILES fragment => scaffold decoration.
    seeds: list[str] = Field(default_factory=lambda: [""])
    num_molecules: int = 20
    temperature: float = 1.0
    randomness: float = 1.0
    scoring: str = "qed"  # qed | logp
    unique: bool = True


class GenMolMolecule(BaseModel):
    seed: str
    smiles: str
    score: Optional[float] = None


class GenMolGenerateResponse(BaseModel):
    molecules: list[GenMolMolecule]


@router.post("/genmol/generate/stream")
def genmol_generate_stream(payload: GenMolGenerateRequest, user: CurrentUserDep):
    """SSE generate. GenMol scales to zero, so the first call cold-starts
    (~30-60s) — streaming keeps the connection alive past proxy timeouts."""
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    seeds = [s.strip() for s in payload.seeds] if payload.seeds else [""]
    if not seeds:
        seeds = [""]

    def work(progress_cb):
        progress_cb(5, "Submitting to GenMol")
        progress_cb(
            20,
            "Generating molecules — the endpoint may cold-start (~30-60s on first call)",
        )
        mols = genmol_svc.generate(
            seeds=seeds,
            num_molecules=payload.num_molecules,
            temperature=payload.temperature,
            randomness=payload.randomness,
            scoring=payload.scoring,
            unique=payload.unique,
        )
        progress_cb(95, f"Generated {len(mols)} molecule(s)")
        return {"molecules": mols}

    return StreamingResponse(
        stream_with_progress(work),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


class SeedMotif(BaseModel):
    scaffold: str
    count: int
    best_pchembl: Optional[float] = None
    example_smiles: str


class SeedMotifsResponse(BaseModel):
    gene: Optional[str] = None
    motifs: list[SeedMotif]


@router.get("/genmol/seed_motifs", response_model=SeedMotifsResponse)
def genmol_seed_motifs(
    _: CurrentUserDep,
    gene: str = Query("", description="Target gene symbol, e.g. PARP1"),
    sequence: str = Query("", description="Protein sequence to reverse-resolve to a gene"),
):
    """Binding motifs (Murcko scaffolds of known ChEMBL binders) for a target —
    seed candidates for GenMol's fragment mode. Pass a gene, or a protein
    sequence to reverse-resolve. Returns {gene, motifs:[]} (empty if the
    target_binders table isn't built or the target has no known binders)."""
    from app.services import target_motifs

    return target_motifs.seed_motifs(gene=gene or None, sequence=sequence or None)


# ── Guided Molecule Optimization ───────────────────────────────────────────────

class MoleculeOptimizeRequest(BaseModel):
    seed_smiles: list[str] = Field(..., min_length=1)  # binding-motif scaffolds
    num_samples: int = 24
    num_iterations: int = 5
    select_top: int = 3
    dock_top_k: int = 5
    weights: dict[str, float] = Field(default_factory=lambda: {"qed": 1.0, "admet": 1.0, "dock": 1.0})
    temperature: float = 1.2
    randomness: float = 2.0
    target_sequence: str = ""      # target protein sequence (folded → docked in-reward)
    dock_per_iter: int = 8
    dock_samples: int = 3
    mlflow_experiment: str = "gwb_molecule_optimization"
    mlflow_run_name: str = Field(..., min_length=1)


class MoleculeOptimizeStartResponse(BaseModel):
    mlflow_run_id: str
    job_run_id: int
    experiment_id: str


@router.post("/molecule_optimization/start", response_model=MoleculeOptimizeStartResponse)
def molecule_optimization_start(payload: MoleculeOptimizeRequest, user: CurrentUserDep):
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    seeds = [s.strip() for s in payload.seed_smiles if s and s.strip()]
    if not seeds:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "At least one seed SMILES is required")
    try:
        res = mol_opt.start_molecule_optimization_job(
            user_email=user.email,
            mlflow_experiment=payload.mlflow_experiment,
            mlflow_run_name=payload.mlflow_run_name,
            seed_smiles=seeds,
            num_samples=payload.num_samples,
            num_iterations=payload.num_iterations,
            select_top=payload.select_top,
            dock_top_k=payload.dock_top_k,
            weights=payload.weights,
            temperature=payload.temperature,
            randomness=payload.randomness,
            target_sequence=payload.target_sequence,
            dock_per_iter=payload.dock_per_iter,
            dock_samples=payload.dock_samples,
        )
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Failed to start optimization: {e}")
    return MoleculeOptimizeStartResponse(
        mlflow_run_id=res["mlflow_run_id"],
        job_run_id=res["job_run_id"],
        experiment_id=res["experiment_id"],
    )


@router.get("/molecule_optimization/status")
def molecule_optimization_status(_: CurrentUserDep, run_id: str = Query(..., min_length=1)):
    return mol_opt.get_run_status(run_id)


@router.get("/molecule_optimization/top-k")
def molecule_optimization_top_k(_: CurrentUserDep, run_id: str = Query(..., min_length=1)):
    return {"top_k": mol_opt.load_top_k(run_id)}


@router.get("/molecule_optimization/search")
def molecule_optimization_search(
    user: CurrentUserDep,
    by: str = Query("run_name"),
    text: str = Query(""),
):
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    return {"runs": mol_opt.search_runs(user.email, by, text)}

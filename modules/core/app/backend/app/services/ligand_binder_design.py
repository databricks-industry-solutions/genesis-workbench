"""Ligand Binder Design pipeline. Proteina-Complexa-Ligand generates
protein binders for a given small molecule. Optional ESMFold + DiffDock
re-validation per design.

Ported from modules/core/app/views/small_molecule_workflows/ligand_binder_design.py
and modules/core/app/utils/small_molecule_tools.py."""
from __future__ import annotations

import logging
from typing import Callable

import pandas as pd
from databricks.sdk import WorkspaceClient

from app.services import molecular_docking as docking
from app.services.dataframe_endpoint import query_dataframe_endpoint
from app.services.endpoints import get_endpoint_name
from app.services.protein import hit_esmfold

logger = logging.getLogger(__name__)


def smiles_to_pdb(smiles: str) -> str:
    """Convert a SMILES string into a 3D-embedded PDB block via RDKit
    (ETKDGv3 → MMFF94). Raises ValueError on invalid input."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.MolToPDBBlock(mol)


def hit_proteina_complexa_ligand(
    target_pdb: str,
    binder_length_min: int = 50,
    binder_length_max: int = 80,
    num_samples: int = 2,
) -> pd.DataFrame:
    """Proteina-Complexa-Ligand endpoint. Expects a ligand-PDB target +
    binder length bounds. Returns rows of (sample_id, sequence, rewards,
    pdb_output)."""
    endpoint_name = get_endpoint_name("Proteina-Complexa Ligand")
    result = query_dataframe_endpoint(
        endpoint_name,
        columns=[
            "target_pdb",
            "binder_length_min",
            "binder_length_max",
            "num_samples",
            "hotspot_residues",
            "target_chain",
        ],
        data=[
            [target_pdb, binder_length_min, binder_length_max, num_samples, "", "A"]
        ],
    )
    predictions = result.get("predictions", result)
    return pd.DataFrame(predictions)


def run_ligand_binder_design(
    ligand_pdb: str | None,
    ligand_smiles: str | None,
    binder_length_min: int,
    binder_length_max: int,
    num_samples: int,
    validate_esmfold: bool,
    validate_diffdock: bool,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict:
    """End-to-end ligand binder design. Caller owns the MLflow run; this
    function focuses on the compute pipeline + per-design assembly."""
    def _p(pct: int, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    w = WorkspaceClient()

    # Step 1: resolve ligand → PDB (HETATM block) -------------------------
    if not ligand_pdb:
        if not ligand_smiles:
            raise ValueError("Either ligand_pdb or ligand_smiles must be provided")
        _p(5, "Converting SMILES → 3D coordinates (RDKit ETKDGv3 + MMFF94)")
        ligand_pdb = smiles_to_pdb(ligand_smiles.strip())
    _p(12, "Ligand structure ready")

    # Step 2: Proteina-Complexa-Ligand binder generation -------------------
    _p(15, f"Generating {num_samples} protein binder(s) for the ligand")
    results_df = hit_proteina_complexa_ligand(
        target_pdb=ligand_pdb,
        binder_length_min=binder_length_min,
        binder_length_max=binder_length_max,
        num_samples=num_samples,
    )
    if results_df.empty:
        _p(100, "Proteina-Complexa-Ligand returned no designs")
        return {"designs": [], "ligand_pdb": ligand_pdb, "warnings": []}
    _p(50 if not (validate_esmfold or validate_diffdock) else 35,
       f"Got {len(results_df)} candidate(s)")

    warnings: list[str] = []
    total = len(results_df)

    # Step 3: ESMFold per design (optional) -------------------------------
    if validate_esmfold:
        esmfold_pdbs: list[str | None] = []
        for i, (_, row) in enumerate(results_df.iterrows()):
            pct = 35 + int(((i + 1) / total) * 25)
            _p(pct, f"Folding design {i + 1}/{total} with ESMFold")
            try:
                esmfold_pdbs.append(hit_esmfold(w, str(row["sequence"])))
            except Exception as e:
                logger.warning("ESMFold failed for design %d: %s", i + 1, e)
                esmfold_pdbs.append(None)
        results_df["esmfold_pdb"] = esmfold_pdbs
        failed = sum(1 for p in esmfold_pdbs if p is None)
        if failed:
            warnings.append(
                f"ESMFold validation failed for {failed}/{total} design(s); using CA backbone."
            )

    # Step 4: DiffDock per design (optional) -------------------------------
    if validate_diffdock and ligand_smiles and ligand_smiles.strip():
        dock_sdfs: list[str | None] = []
        dock_scores: list[float | None] = []
        for i, (_, row) in enumerate(results_df.iterrows()):
            pct = 60 + int(((i + 1) / total) * 30)
            _p(pct, f"Docking design {i + 1}/{total} with DiffDock")
            try:
                # ESMFold-validated structure preferred; fall back to the
                # CA-only backbone from Proteina-Complexa-Ligand.
                dock_pdb = (
                    row.get("esmfold_pdb") if "esmfold_pdb" in results_df.columns else None
                ) or row["pdb_output"]
                dock_df = docking.hit_diffdock(
                    dock_pdb, ligand_smiles, samples_per_complex=5
                )
                best = dock_df.sort_values("confidence", ascending=False).iloc[0]
                sdf = str(best["ligand_sdf"])
                # DiffDock returns "ERROR…" inline for per-pose failures.
                if sdf.startswith("ERROR"):
                    dock_sdfs.append(None)
                    dock_scores.append(None)
                else:
                    dock_sdfs.append(sdf)
                    dock_scores.append(float(best["confidence"]))
            except Exception as e:
                logger.warning("DiffDock failed for design %d: %s", i + 1, e)
                dock_sdfs.append(None)
                dock_scores.append(None)
        results_df["best_dock_sdf"] = dock_sdfs
        results_df["dock_confidence"] = dock_scores
        dock_failed = sum(1 for s in dock_sdfs if s is None)
        if dock_failed:
            warnings.append(
                f"DiffDock validation failed for {dock_failed}/{total} design(s); docked pose unavailable."
            )

    designs: list[dict] = []
    for _, row in results_df.iterrows():
        designs.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "sequence": str(row.get("sequence", "")),
                "rewards": float(row.get("rewards", 0.0) or 0.0),
                "pdb_output": str(row.get("pdb_output", "")),
                "esmfold_pdb": (
                    str(row["esmfold_pdb"]) if row.get("esmfold_pdb") else None
                ) if "esmfold_pdb" in results_df.columns else None,
                "best_dock_sdf": (
                    str(row["best_dock_sdf"]) if row.get("best_dock_sdf") else None
                ) if "best_dock_sdf" in results_df.columns else None,
                "dock_confidence": (
                    float(row["dock_confidence"]) if row.get("dock_confidence") is not None else None
                ) if "dock_confidence" in results_df.columns else None,
            }
        )

    _p(95, "Assembling response")
    return {"designs": designs, "ligand_pdb": ligand_pdb, "warnings": warnings}

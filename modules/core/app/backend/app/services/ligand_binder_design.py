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
from databricks.sdk.core import Config

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

    w = WorkspaceClient(config=Config(http_timeout_seconds=600))

    # Step 1: resolve ligand → PDB (HETATM block). SMILES→PDB stays in the UI
    # (RDKit) so the shared executor core stays dependency-free.
    if not ligand_pdb:
        if not ligand_smiles:
            raise ValueError("Either ligand_pdb or ligand_smiles must be provided")
        _p(5, "Converting SMILES → 3D coordinates (RDKit ETKDGv3 + MMFF94)")
        ligand_pdb = smiles_to_pdb(ligand_smiles.strip())
    _p(12, "Ligand structure ready")

    # Steps 2-4 (Proteina-Complexa-Ligand → ESMFold → DiffDock) run in the shared
    # executor chain; the UI keeps MLflow + SSE progress + viewer assembly.
    from genesis_workbench.executor import run_chain

    result = run_chain(
        "ligand_binder_design",
        {"ligand_pdb": ligand_pdb},
        {
            "binder_length_min": binder_length_min,
            "binder_length_max": binder_length_max,
            "num_samples": num_samples,
            "validate_esmfold": validate_esmfold,
            "validate_diffdock": validate_diffdock,
            "ligand_smiles": ligand_smiles or "",
        },
        w,
        progress=_p,
    )
    designs = result.get("designs", [])
    total = len(designs)
    warnings: list[str] = []
    if validate_esmfold and total:
        failed = sum(1 for d in designs if not d.get("esmfold_pdb"))
        if failed:
            warnings.append(
                f"ESMFold validation failed for {failed}/{total} design(s); using CA backbone."
            )
    if validate_diffdock and ligand_smiles and ligand_smiles.strip() and total:
        dock_failed = sum(1 for d in designs if not d.get("best_dock_sdf"))
        if dock_failed:
            warnings.append(
                f"DiffDock validation failed for {dock_failed}/{total} design(s); docked pose unavailable."
            )
    return {"designs": designs, "ligand_pdb": result.get("ligand_pdb", ligand_pdb), "warnings": warnings}

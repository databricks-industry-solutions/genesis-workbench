"""Protein Binder Design pipeline.

Pipeline:
  1. (sequence-mode only) ESMFold the input sequence → PDB.
  2. Call the Proteina-Complexa binder-design endpoint with the target PDB
     + binder length range + (optional) hotspot residues.
  3. (optional) For each returned design, ESMFold the binder sequence to
     validate that the redesigned chain folds. Store esmfold_pdb +
     esmfold_validated.

Ported from modules/core/app/views/small_molecule_workflows/binder_design.py
and modules/core/app/utils/small_molecule_tools.hit_proteina_complexa."""
from __future__ import annotations

import logging
from typing import Callable

import pandas as pd
from databricks.sdk import WorkspaceClient

from app.services.dataframe_endpoint import query_dataframe_endpoint
from app.services.endpoints import get_endpoint_name
from app.services.protein import hit_esmfold

logger = logging.getLogger(__name__)


def hit_proteina_complexa(
    target_pdb: str,
    target_chain: str = "A",
    hotspot_residues: str = "",
    binder_length_min: int = 50,
    binder_length_max: int = 80,
    num_samples: int = 2,
) -> pd.DataFrame:
    """One-shot call to the Proteina-Complexa binder endpoint. Returns a
    DataFrame keyed by `sample_id` with columns: sequence, rewards,
    pdb_output (CA-only backbone), plus whatever extras the endpoint adds."""
    endpoint_name = get_endpoint_name("Proteina-Complexa Binder")
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
            [
                target_pdb,
                binder_length_min,
                binder_length_max,
                num_samples,
                hotspot_residues,
                target_chain,
            ]
        ],
    )
    predictions = result.get("predictions", result)
    return pd.DataFrame(predictions)


def run_binder_design(
    target_pdb: str | None,
    target_sequence: str | None,
    target_chain: str,
    hotspot_residues: str,
    binder_length_min: int,
    binder_length_max: int,
    num_samples: int,
    validate_esmfold: bool,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict:
    """End-to-end binder design. Caller is responsible for the MLflow run
    bookkeeping (this function focuses on the compute + viewer assembly)."""
    def _p(pct: int, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    w = WorkspaceClient()

    # Step 1: resolve target → PDB ----------------------------------------
    if not target_pdb:
        if not target_sequence:
            raise ValueError("Either target_pdb or target_sequence must be provided")
        _p(5, "Folding target sequence with ESMFold")
        target_pdb = hit_esmfold(w, target_sequence.strip())
    _p(15, "Target PDB ready")

    # Step 2: Proteina-Complexa binder design -----------------------------
    _p(20, f"Generating {num_samples} binder design(s) with Proteina-Complexa")
    results_df = hit_proteina_complexa(
        target_pdb=target_pdb,
        target_chain=target_chain,
        hotspot_residues=hotspot_residues,
        binder_length_min=binder_length_min,
        binder_length_max=binder_length_max,
        num_samples=num_samples,
    )
    if results_df.empty:
        _p(100, "Proteina-Complexa returned no designs")
        return {
            "designs": [],
            "target_pdb": target_pdb,
            "warnings": [],
        }
    _p(50, f"Proteina-Complexa returned {len(results_df)} design(s)")

    # Step 3: optional ESMFold validation per design ----------------------
    warnings: list[str] = []
    if validate_esmfold:
        esmfold_pdbs: list[str | None] = []
        validated: list[bool] = []
        total = len(results_df)
        for i, (_, row) in enumerate(results_df.iterrows()):
            pct = 50 + int(((i + 1) / total) * 45)
            _p(pct, f"Validating design {i + 1}/{total} with ESMFold")
            try:
                pdb = hit_esmfold(w, str(row["sequence"]))
                esmfold_pdbs.append(pdb)
                validated.append(True)
            except Exception as e:
                logger.warning("ESMFold validation failed for design %d: %s", i + 1, e)
                esmfold_pdbs.append(None)
                validated.append(False)
        results_df["esmfold_pdb"] = esmfold_pdbs
        results_df["esmfold_validated"] = validated
        failed = sum(1 for v in validated if not v)
        if failed:
            warnings.append(
                f"ESMFold validation failed for {failed}/{total} design(s)."
            )

    designs: list[dict] = []
    for _, row in results_df.iterrows():
        # Prefer the ESMFold-validated structure when available; fall back
        # to the Proteina-Complexa CA-only backbone.
        esmfold_pdb = row.get("esmfold_pdb") if "esmfold_pdb" in results_df.columns else None
        pdb_output = row.get("pdb_output")
        binder_pdb = esmfold_pdb or pdb_output or None
        designs.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "sequence": str(row.get("sequence", "")),
                "rewards": float(row.get("rewards", 0.0) or 0.0),
                "esmfold_validated": bool(
                    row.get("esmfold_validated", False)
                    if "esmfold_validated" in results_df.columns
                    else False
                ),
                "binder_pdb": binder_pdb,
            }
        )

    _p(96, "Assembling response")
    return {
        "designs": designs,
        "target_pdb": target_pdb,
        "warnings": warnings,
    }

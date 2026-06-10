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

    w = WorkspaceClient(http_timeout_seconds=600)

    # Delegate the ESMFold(target) -> Proteina-Complexa -> ESMFold(validate)
    # chain to the shared executor (same code Vortex/MCP run). The UI keeps the
    # MLflow run + SSE progress (via `_p`) and assembles the viewer response.
    from genesis_workbench.executor import run_chain

    result = run_chain(
        "protein_binder_design",
        {"target_pdb": target_pdb, "target_sequence": target_sequence},
        {
            "target_chain": target_chain,
            "hotspot_residues": hotspot_residues,
            "binder_length_min": binder_length_min,
            "binder_length_max": binder_length_max,
            "num_samples": num_samples,
            "validate_esmfold": validate_esmfold,
        },
        w,
        progress=_p,
    )
    designs = result.get("designs", [])
    warnings: list[str] = []
    if validate_esmfold and designs:
        failed = sum(1 for d in designs if not d.get("esmfold_validated"))
        if failed:
            warnings.append(f"ESMFold validation failed for {failed}/{len(designs)} design(s).")
    return {
        "designs": designs,
        "target_pdb": result.get("target_pdb", target_pdb),
        "warnings": warnings,
    }

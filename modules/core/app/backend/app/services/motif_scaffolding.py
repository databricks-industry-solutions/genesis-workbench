"""Functional Motif Scaffolding pipeline. Proteina-Complexa-AME generates
protein scaffolds that preserve a given functional motif (active site,
binding loop, etc.). Optional ProteinMPNN sequence optimisation + ESMFold
validation per design.

Ported from modules/core/app/views/small_molecule_workflows/motif_scaffolding.py."""
from __future__ import annotations

import logging
from typing import Callable

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

from app.services.dataframe_endpoint import query_dataframe_endpoint
from app.services.endpoints import get_endpoint_name
from app.services.protein import hit_esmfold

logger = logging.getLogger(__name__)


def hit_proteina_complexa_ame(
    target_pdb: str,
    target_chain: str = "B",
    binder_length_min: int = 50,
    binder_length_max: int = 80,
    num_samples: int = 2,
) -> pd.DataFrame:
    """Proteina-Complexa-AME endpoint. Same dataframe_split shape as the
    other Proteina-Complexa variants (hotspot_residues is unused here)."""
    endpoint_name = get_endpoint_name("Proteina-Complexa AME")
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
                "",
                target_chain,
            ]
        ],
    )
    predictions = result.get("predictions", result)
    return pd.DataFrame(predictions)


def _hit_proteinmpnn_full_redesign(pdb_str: str) -> list[str]:
    """Two-column ProteinMPNN call (`pdb` + `fixed_positions=""`). The
    empty `fixed_positions` lets MPNN redesign every residue — fine for
    motif scaffolding because the AME endpoint already preserved the motif
    geometry; we're only optimising side-chains on the scaffold."""
    endpoint_name = get_endpoint_name("ProteinMPNN")
    # ProteinMPNN's PyFunc V8 needs dataframe_records, not dataframe_split.
    w = WorkspaceClient(config=Config(http_timeout_seconds=600))
    response = w.serving_endpoints.query(
        name=endpoint_name,
        dataframe_records=[{"pdb": pdb_str, "fixed_positions": ""}],
    )
    preds = response.predictions
    if isinstance(preds, list):
        return [str(s) for s in preds]
    if isinstance(preds, dict) and "predictions" in preds:
        return [str(s) for s in preds["predictions"]]
    raise RuntimeError(f"Unexpected ProteinMPNN response: {preds!r}")


def run_motif_scaffolding(
    motif_pdb: str,
    target_chain: str,
    scaffold_length_min: int,
    scaffold_length_max: int,
    num_samples: int,
    optimize_mpnn: bool,
    validate_esmfold: bool,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict:
    """End-to-end motif scaffolding. Caller owns the MLflow run."""
    def _p(pct: int, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    w = WorkspaceClient(config=Config(http_timeout_seconds=600))

    # Proteina-Complexa-AME → (ProteinMPNN) → (ESMFold) runs in the shared
    # executor chain; the UI keeps MLflow + SSE progress + viewer assembly.
    from genesis_workbench.executor import run_chain

    result = run_chain(
        "motif_scaffolding",
        {"motif_pdb": motif_pdb},
        {
            "target_chain": target_chain,
            "scaffold_length_min": scaffold_length_min,
            "scaffold_length_max": scaffold_length_max,
            "num_samples": num_samples,
            "optimize_mpnn": optimize_mpnn,
            "validate_esmfold": validate_esmfold,
        },
        w,
        progress=_p,
    )
    scaffolds = result.get("scaffolds", [])
    total = len(scaffolds)
    warnings: list[str] = []
    if optimize_mpnn and total:
        failed = sum(1 for s in scaffolds if not s.get("mpnn_sequence"))
        if failed:
            warnings.append(
                f"ProteinMPNN optimisation failed for {failed}/{total} scaffold(s); using original sequences."
            )
    if validate_esmfold and total:
        failed = sum(1 for s in scaffolds if not s.get("esmfold_validated"))
        if failed:
            warnings.append(
                f"ESMFold validation failed for {failed}/{total} scaffold(s); using CA backbone."
            )
    return {"scaffolds": scaffolds, "motif_pdb": result.get("motif_pdb", motif_pdb), "warnings": warnings}

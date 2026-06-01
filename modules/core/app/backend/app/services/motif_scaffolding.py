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
    w = WorkspaceClient()
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

    w = WorkspaceClient()

    # Step 1: Proteina-Complexa-AME scaffold generation -------------------
    _p(10, f"Generating {num_samples} scaffold(s) with Proteina-Complexa-AME")
    results_df = hit_proteina_complexa_ame(
        target_pdb=motif_pdb,
        target_chain=target_chain,
        binder_length_min=scaffold_length_min,
        binder_length_max=scaffold_length_max,
        num_samples=num_samples,
    )
    if results_df.empty:
        _p(100, "Proteina-Complexa-AME returned no scaffolds")
        return {"scaffolds": [], "motif_pdb": motif_pdb, "warnings": []}
    _p(35, f"Got {len(results_df)} scaffold(s)")

    warnings: list[str] = []
    total = len(results_df)

    # Step 2: ProteinMPNN sequence optimisation (optional) ----------------
    if optimize_mpnn:
        mpnn_seqs: list[str | None] = []
        for i, (_, row) in enumerate(results_df.iterrows()):
            pct = 35 + int(((i + 1) / total) * 25)
            _p(pct, f"Optimising sequence {i + 1}/{total} with ProteinMPNN")
            try:
                seqs = _hit_proteinmpnn_full_redesign(str(row["pdb_output"]))
                mpnn_seqs.append(seqs[0] if seqs else None)
            except Exception as e:
                logger.warning("ProteinMPNN failed for scaffold %d: %s", i + 1, e)
                mpnn_seqs.append(None)
        # Fall back to original sequence where MPNN failed.
        results_df["mpnn_sequence"] = [
            m if m else str(results_df.iloc[i]["sequence"])
            for i, m in enumerate(mpnn_seqs)
        ]
        failed = sum(1 for m in mpnn_seqs if m is None)
        if failed:
            warnings.append(
                f"ProteinMPNN optimisation failed for {failed}/{total} scaffold(s); using original sequences."
            )

    # Step 3: ESMFold validation (optional) -------------------------------
    if validate_esmfold:
        seq_col = "mpnn_sequence" if "mpnn_sequence" in results_df.columns else "sequence"
        esmfold_pdbs: list[str | None] = []
        validated: list[bool] = []
        for i, (_, row) in enumerate(results_df.iterrows()):
            pct = 60 + int(((i + 1) / total) * 30)
            _p(pct, f"Folding scaffold {i + 1}/{total} with ESMFold")
            try:
                esmfold_pdbs.append(hit_esmfold(w, str(row[seq_col])))
                validated.append(True)
            except Exception as e:
                logger.warning("ESMFold failed for scaffold %d: %s", i + 1, e)
                esmfold_pdbs.append(None)
                validated.append(False)
        results_df["esmfold_pdb"] = esmfold_pdbs
        results_df["esmfold_validated"] = validated
        failed = sum(1 for v in validated if not v)
        if failed:
            warnings.append(
                f"ESMFold validation failed for {failed}/{total} scaffold(s); using CA backbone."
            )

    scaffolds: list[dict] = []
    for _, row in results_df.iterrows():
        esmfold_pdb = row.get("esmfold_pdb") if "esmfold_pdb" in results_df.columns else None
        scaffolds.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "sequence": str(row.get("sequence", "")),
                "mpnn_sequence": (
                    str(row["mpnn_sequence"]) if "mpnn_sequence" in results_df.columns else None
                ),
                "rewards": float(row.get("rewards", 0.0) or 0.0),
                "pdb_output": str(row.get("pdb_output", "")),
                "esmfold_pdb": str(esmfold_pdb) if esmfold_pdb else None,
                "esmfold_validated": bool(row.get("esmfold_validated", False))
                if "esmfold_validated" in results_df.columns
                else False,
            }
        )

    _p(95, "Assembling response")
    return {"scaffolds": scaffolds, "motif_pdb": motif_pdb, "warnings": warnings}

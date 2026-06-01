"""DiffDock molecular docking pipeline. Ported from
modules/core/app/utils/small_molecule_tools.py.

Two-stage funnel:
  1. ESM-2 embed the target protein PDB (diffdock_esm_embeddings endpoint).
  2. DiffDock scoring with the pre-computed embeddings + ligand SMILES
     (diffdock endpoint). Returns N ranked poses, each with a confidence
     score and the docked-ligand SDF.

Synchronous; ~10-30s per request when both endpoints are warm. SSE
keepalives in the streaming route cover cold-starts."""
from __future__ import annotations

import logging
from typing import Callable

import pandas as pd
from databricks.sdk import WorkspaceClient

from app.services.endpoints import get_endpoint_name

logger = logging.getLogger(__name__)

EXAMPLE_SMILES = "COc(cc1)ccc1C#N"
_EXAMPLE_PDB_CACHE: dict[str, str] = {}


def get_example_pdb() -> str:
    """Return chain A of PDB 6agt, trimmed to the first 50 residues — same
    structure used in the DiffDock model registration tests. Cached after
    the first call. If the RCSB fetch fails (offline, network blocked),
    returns a placeholder so the form still has something to render."""
    cached = _EXAMPLE_PDB_CACHE.get("pdb")
    if cached:
        return cached
    try:
        import requests

        resp = requests.get("https://files.rcsb.org/view/6agt.pdb", timeout=10)
        resp.raise_for_status()
        seen: set[str] = set()
        trimmed: list[str] = []
        for line in resp.text.splitlines(keepends=True):
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[21] != "A":
                continue
            resseq = line[22:27].strip()
            seen.add(resseq)
            if len(seen) > 50:
                break
            trimmed.append(line)
        trimmed.append("END\n")
        _EXAMPLE_PDB_CACHE["pdb"] = "".join(trimmed)
    except Exception as e:
        logger.warning("get_example_pdb: RCSB fetch failed (%s); using placeholder", e)
        _EXAMPLE_PDB_CACHE["pdb"] = (
            "# Could not fetch example PDB from RCSB. Paste your own PDB content here.\nEND\n"
        )
    return _EXAMPLE_PDB_CACHE["pdb"]


def sdf_to_hetatm(sdf_content: str) -> str:
    """Convert the atom block of an SDF molecule to PDB-style HETATM records
    so it can be overlaid on the protein in Mol*. Returns "" if the SDF is
    malformed — caller should skip the ligand in the viewer."""
    lines = sdf_content.split("\n")
    counts_idx = None
    for i, line in enumerate(lines):
        if "V2000" in line or "V3000" in line:
            counts_idx = i
            break
    if counts_idx is None:
        logger.warning("sdf_to_hetatm: no V2000/V3000 counts line in %d lines", len(lines))
        return ""
    try:
        num_atoms = int(lines[counts_idx][:3].strip())
    except (ValueError, IndexError):
        logger.warning("sdf_to_hetatm: malformed counts line %r", lines[counts_idx][:30])
        return ""

    hetatm_lines: list[str] = []
    for i in range(num_atoms):
        atom_idx = counts_idx + 1 + i
        if atom_idx >= len(lines):
            break
        atom_line = lines[atom_idx]
        if len(atom_line) < 34:
            continue
        parts = atom_line.split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            element = parts[3]
        except (ValueError, IndexError):
            continue
        atom_name = element.ljust(2)
        serial = i + 1
        hetatm_lines.append(
            f"HETATM{serial:5d}  {atom_name:<2s}  LIG B   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
        )
    return "\n".join(hetatm_lines)


def _query_endpoint(endpoint_name: str, payload: dict) -> dict:
    """Send a request to a model-serving endpoint via the Databricks SDK.
    `dataframe_split` must be wrapped in `DataframeSplitInput` — the SDK
    calls `.as_dict()` on it and a plain dict raises AttributeError."""
    from databricks.sdk.service.serving import DataframeSplitInput

    w = WorkspaceClient()  # app SP — endpoints require model-serving scope
    ds = payload["dataframe_split"]
    response = w.serving_endpoints.query(
        name=endpoint_name,
        dataframe_split=DataframeSplitInput(
            columns=ds.get("columns"),
            data=ds.get("data"),
        ),
    )
    # The SDK normalises predictions onto response.predictions for non-chat
    # endpoints. Some legacy responses tunnel a dict-with-"predictions" — be
    # permissive about both shapes since this code originally targeted both.
    preds = response.predictions
    if isinstance(preds, dict) and "predictions" in preds:
        return preds
    return {"predictions": preds}


def hit_diffdock(
    protein_pdb: str,
    ligand_smiles: str,
    samples_per_complex: int = 10,
    progress_callback: Callable[[int, str], None] | None = None,
    pct_embed_start: int = 5,
    pct_embed_end: int = 25,
    pct_score_start: int = 25,
    pct_score_end: int = 85,
) -> pd.DataFrame:
    """Two-step DiffDock pipeline. `progress_callback(pct, msg)` fires at
    the two endpoint boundaries plus a final 'received' tick."""
    def _p(pct: int, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    # Step 1: ESM embeddings ------------------------------------------------
    esm_endpoint = get_endpoint_name("DiffDock ESM Embeddings")
    _p(pct_embed_start, f"Computing ESM-2 embeddings on {esm_endpoint}")
    esm_result = _query_endpoint(
        esm_endpoint,
        {"dataframe_split": {"columns": ["protein_pdb"], "data": [[protein_pdb]]}},
    )
    esm_predictions = esm_result.get("predictions", esm_result)
    if isinstance(esm_predictions, list) and esm_predictions:
        embeddings_b64 = esm_predictions[0].get("embeddings_b64", "{}")
    elif isinstance(esm_predictions, dict):
        embeddings_b64 = esm_predictions.get("embeddings_b64", "{}")
    else:
        embeddings_b64 = "{}"
    _p(pct_embed_end, "ESM embeddings ready")

    # Step 2: DiffDock scoring ---------------------------------------------
    scoring_endpoint = get_endpoint_name("DiffDock")
    _p(pct_score_start, f"Generating {samples_per_complex} pose(s) on {scoring_endpoint}")
    score_result = _query_endpoint(
        scoring_endpoint,
        {
            "dataframe_split": {
                "columns": [
                    "protein_pdb",
                    "ligand_smiles",
                    "samples_per_complex",
                    "esm_embeddings_b64",
                ],
                "data": [
                    [protein_pdb, ligand_smiles, samples_per_complex, embeddings_b64]
                ],
            }
        },
    )
    predictions = score_result.get("predictions", score_result)
    df = pd.DataFrame(predictions)
    _p(pct_score_end, f"DiffDock returned {len(df)} pose(s)")
    return df

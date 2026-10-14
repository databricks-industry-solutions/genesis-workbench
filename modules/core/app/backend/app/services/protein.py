"""Synchronous protein-structure inference helpers. Both call serving
endpoints with the app SP via the supplied WorkspaceClient (user OBO
tokens don't carry the model-serving scope)."""
from __future__ import annotations

import logging

from databricks.sdk import WorkspaceClient

from app.services.endpoints import get_endpoint_name

logger = logging.getLogger(__name__)


def hit_esmfold(w: WorkspaceClient, sequence: str) -> str:
    endpoint_name = get_endpoint_name("ESMFold")
    logger.info("Hitting ESMFold endpoint: %s", endpoint_name)
    response = w.serving_endpoints.query(name=endpoint_name, inputs=[sequence])
    predictions = response.predictions
    if not predictions:
        raise RuntimeError("ESMFold returned empty predictions")
    return predictions[0]


def hit_proteinmpnn(w: WorkspaceClient, pdb_str: str) -> list[str]:
    endpoint_name = get_endpoint_name("ProteinMPNN")
    logger.info("Hitting ProteinMPNN endpoint: %s", endpoint_name)
    response = w.serving_endpoints.query(
        name=endpoint_name,
        dataframe_records=[{"pdb": pdb_str, "fixed_positions": ""}],
    )
    predictions = response.predictions
    if isinstance(predictions, list):
        return [str(s) for s in predictions]
    raise RuntimeError(f"ProteinMPNN returned unexpected payload: {predictions!r}")


def hit_rfdiffusion(w: WorkspaceClient, payload: dict) -> str:
    endpoint_name = get_endpoint_name("RFDiffusion")
    logger.info("Hitting RFDiffusion endpoint: %s", endpoint_name)
    response = w.serving_endpoints.query(name=endpoint_name, inputs=[payload])
    predictions = response.predictions
    if isinstance(predictions, list) and predictions:
        return predictions[0]
    raise RuntimeError(f"RFDiffusion returned unexpected payload: {predictions!r}")


def hit_boltz(
    w: WorkspaceClient,
    sequence: str,
    msa: str = "no_msa",
    use_msa_server: str = "True",
) -> str:
    if not any(prefix in sequence for prefix in ("protein_", "rna_", "dna_", "smiles_", ":")):
        boltz_input = f"protein_A:{sequence}"
    else:
        boltz_input = sequence

    payload = [{"input": boltz_input, "msa": msa, "use_msa_server": use_msa_server}]

    endpoint_name = get_endpoint_name("Boltz")
    logger.info("Hitting Boltz endpoint: %s", endpoint_name)
    response = w.serving_endpoints.query(name=endpoint_name, inputs=payload)

    result = response.predictions
    if isinstance(result, list) and result:
        entry = result[0]
        if isinstance(entry, dict) and "pdb" in entry:
            return entry["pdb"]
        return entry
    raise RuntimeError(f"Boltz returned unexpected payload: {result!r}")


def hit_hlaiipred_windows(w: WorkspaceClient, windows: list[str]) -> list[float]:
    """Query HLAIIPred (MHC-II presentation) for a batch of equal-length peptide windows and return
    each window's `max_presentation_score`. Used to build a per-residue immunogenicity (ADA) burden
    heat map: the caller slides a fixed-width window over the sequence, so each query row is one
    window and its score is that window's presentation. Uses the model's default DRB1 allele panel."""
    endpoint_name = get_endpoint_name("HLAIIPred")
    logger.info("Hitting HLAIIPred endpoint: %s (%d windows)", endpoint_name, len(windows))
    response = w.serving_endpoints.query(
        name=endpoint_name, inputs=[{"sequence": win} for win in windows]
    )
    preds = response.predictions
    # predictions may be a dict-of-columns or a list-of-rows depending on the serving wrapper.
    if isinstance(preds, dict) and "max_presentation_score" in preds:
        return [float(v) for v in preds["max_presentation_score"]]
    if isinstance(preds, list):
        return [float(r.get("max_presentation_score", 0.0)) if isinstance(r, dict) else float(r)
                for r in preds]
    raise RuntimeError(f"HLAIIPred returned unexpected payload: {preds!r}")

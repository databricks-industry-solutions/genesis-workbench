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

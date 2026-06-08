"""GenMol generative small-molecule design — query the GenMol serving endpoint.

GenMol's PyFunc takes a `fragment` column (empty string => de novo; a SMILES
fragment => scaffold decoration) plus MLflow generation `params`. The SDK's
`serving_endpoints.query()` can't pass MLflow `params`, so POST the raw
`{dataframe_split, params}` body via the app service principal (these endpoints
need the model-serving scope user OBO tokens lack). Returns one row per
generated molecule: `{seed, smiles, score}`."""
from __future__ import annotations

import logging

from databricks.sdk import WorkspaceClient

from app.services.endpoints import get_endpoint_name

logger = logging.getLogger(__name__)


def generate(
    seeds: list[str],
    num_molecules: int = 20,
    temperature: float = 1.0,
    randomness: float = 1.0,
    scoring: str = "qed",
    unique: bool = True,
) -> list[dict]:
    """Generate molecules for each seed. An empty seed => de novo generation;
    a SMILES fragment => fragment_completion. Returns a list of dicts with
    keys seed/smiles/score (already ranked by the endpoint)."""
    endpoint_name = get_endpoint_name("GenMol Molecule Generator")
    w = WorkspaceClient()
    body = {
        "dataframe_split": {
            "columns": ["fragment"],
            "data": [[s] for s in seeds],
        },
        "params": {
            "num_molecules": int(num_molecules),
            "temperature": float(temperature),
            "randomness": float(randomness),
            "scoring": scoring,
            "unique": bool(unique),
        },
    }
    resp = w.api_client.do(
        "POST", f"/serving-endpoints/{endpoint_name}/invocations", body=body
    )
    preds = resp.get("predictions", resp) if isinstance(resp, dict) else resp
    return list(preds or [])

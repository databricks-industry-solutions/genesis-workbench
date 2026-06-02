"""scGPT perturbation endpoint wrapper. Mirrors the input shape used by the
Earlier `_hit_perturbation_endpoint` in
modules/core/app/views/single_cell_workflows/perturbation.py."""
from __future__ import annotations

import json
import logging

import pandas as pd
from databricks.sdk import WorkspaceClient

from app.services.endpoints import get_endpoint_name

logger = logging.getLogger(__name__)


def predict_perturbation(
    expression: list[float],
    gene_names: list[str],
    genes_to_perturb: list[str],
    perturbation_type: str,
) -> pd.DataFrame:
    endpoint_name = get_endpoint_name("scGPT Perturbation")
    payload = [
        {
            "expression": expression,
            "gene_names": json.dumps(gene_names),
            "genes_to_perturb": ",".join(genes_to_perturb),
            "perturbation_type": perturbation_type,
        }
    ]
    logger.info(
        "scGPT Perturbation: %d genes, perturbing %s (%s)",
        len(expression),
        genes_to_perturb,
        perturbation_type,
    )

    w = WorkspaceClient()
    response = w.serving_endpoints.query(name=endpoint_name, inputs=payload)
    result = response.predictions
    if result is None:
        raise RuntimeError(f"scGPT Perturbation endpoint returned no predictions")

    # The endpoint returns either a column-oriented dict
    # {gene_name: [...], original_expression: [...], ...} or a list with one
    # such dict. Normalize to DataFrame.
    if isinstance(result, list) and result:
        if isinstance(result[0], dict):
            result = result[0]
    if not isinstance(result, dict) or "gene_name" not in result:
        raise RuntimeError(
            f"scGPT Perturbation returned unexpected payload (no 'gene_name' column): {type(result)}"
        )

    df = pd.DataFrame(result)
    for col in ("original_expression", "predicted_expression", "delta", "abs_delta"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

"""Shared helper for Databricks serving endpoints that take a
`dataframe_split` payload — the SDK requires a `DataframeSplitInput` wrapper
(plain dicts raise `AttributeError: 'dict' object has no attribute 'as_dict'`
when the SDK serialises the body).

Used by the small-molecules endpoints (DiffDock, Proteina-Complexa, etc.)."""
from __future__ import annotations

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import DataframeSplitInput


def query_dataframe_endpoint(name: str, columns: list[str], data: list[list]) -> dict:
    """Query a serving endpoint with a `dataframe_split` body. Returns a
    dict with `predictions`. App SP is used implicitly (these endpoints
    need the model-serving scope that user OBO tokens lack)."""
    w = WorkspaceClient()
    response = w.serving_endpoints.query(
        name=name,
        dataframe_split=DataframeSplitInput(columns=columns, data=data),
    )
    preds = response.predictions
    if isinstance(preds, dict) and "predictions" in preds:
        return preds
    return {"predictions": preds}

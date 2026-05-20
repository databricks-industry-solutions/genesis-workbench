"""Client helpers for TEDDY-driven joint cell-type + disease annotation.

Pipeline (per cell):
  1. Call gwb_teddy_endpoint → 768-d embedding (or 512/1024 depending on variant).
  2. Query teddy_cell_index Vector Search for k nearest neighbors.
  3. Fetch (cell_type, disease) for those neighbors from the teddy_cells Delta.
  4. Majority-vote both columns per source cluster.

Mirrors scimilarity_tools.annotate_clusters but does both vote columns in one pass.
"""
import os
import json
import logging
from collections import Counter

import numpy as np
import pandas as pd
from databricks.sdk import WorkspaceClient

from genesis_workbench.workbench import execute_select_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

workspace_client = WorkspaceClient()

# Reference table + index (built by teddy module notebooks 03 + 04)
TEDDY_CELLS_TABLE = "teddy_cells"
TEDDY_CELL_INDEX = "teddy_cell_index"

# Embedding requests are heavy (each cell carries ~36k expression values when
# the source data uses the full Census gene vocab). Stay under the 16 MB cap.
_CELLS_PER_EMBED_REQUEST = 32


def _get_endpoint_name() -> str:
    dev_user_prefix = os.environ.get("DEV_USER_PREFIX", "")
    if dev_user_prefix and dev_user_prefix.lower() not in ("none", ""):
        return f"gwb_{dev_user_prefix}_teddy_endpoint"
    return "gwb_teddy_endpoint"


def _query_endpoint(payload, params=None):
    endpoint = _get_endpoint_name()
    return workspace_client.serving_endpoints.query(
        name=endpoint,
        inputs=payload,
        params=params or {},
    ).predictions


def embed_cells(
    expr_df: pd.DataFrame,
    gene_names: list,
    max_seq_len: int = 2048,
    progress_callback=None,
) -> pd.DataFrame:
    """Get a TEDDY embedding for every row of expr_df.

    Returns a DataFrame indexed identically to expr_df with a single column
    `embedding` (list[float] of d_model).
    """
    def _progress(pct, text):
        if progress_callback:
            progress_callback(pct, text)

    n = len(expr_df)
    if n == 0:
        return pd.DataFrame(columns=["embedding"])

    rows = []
    batches = [(s, min(s + _CELLS_PER_EMBED_REQUEST, n)) for s in range(0, n, _CELLS_PER_EMBED_REQUEST)]
    for bi, (start, end) in enumerate(batches):
        batch = expr_df.iloc[start:end]
        _progress(
            5 + int(45 * bi / max(1, len(batches))),
            f"Embedding cells {start + 1}-{end} of {n}…",
        )
        var_df = pd.DataFrame({"index": list(gene_names)})
        payload = [{
            "adata_sparsematrix": batch.values.tolist(),
            "adata_obs": pd.DataFrame(index=batch.index).to_json(orient="split"),
            "adata_var": var_df.to_json(orient="split"),
        }]
        try:
            result = _query_endpoint(payload, params={"max_seq_len": max_seq_len, "pooling": "mean"})
        except Exception as e:
            raise RuntimeError(f"TEDDY endpoint call failed for cells {start}-{end}: {e}") from e

        records = result if isinstance(result, list) else result.get("predictions", [])
        if len(records) != end - start:
            raise RuntimeError(f"Expected {end - start} embeddings, got {len(records)}")

        for cell_id, rec in zip(batch.index, records):
            rows.append({"cell_id": cell_id, "embedding": rec["embedding"]})

    out = pd.DataFrame(rows).set_index("cell_id")
    out.index = expr_df.index  # preserve original index ordering
    return out


def _fetch_neighbor_metadata(cell_ids: list, catalog: str, schema: str) -> pd.DataFrame:
    """Pull cell_type, disease, tissue from teddy_cells for a list of neighbor cell_ids."""
    if not cell_ids:
        return pd.DataFrame()
    ids_str = ", ".join(f"'{cid}'" for cid in cell_ids)
    df = execute_select_query(f"""
        SELECT cell_id, cell_type, disease, tissue, tissue_general, dataset_id
        FROM {catalog}.{schema}.{TEDDY_CELLS_TABLE}
        WHERE cell_id IN ({ids_str})
    """)
    if not df.empty and "cell_id" in df.columns:
        df = df.set_index("cell_id").reindex(cell_ids).reset_index()
    return df


def search_nearest_cells(embedding, k: int = 100,
                         catalog: str = None, schema: str = None) -> pd.DataFrame:
    """KNN search against the TEDDY reference index. Returns a DataFrame of neighbor metadata."""
    if catalog is None:
        catalog = os.environ.get("CORE_CATALOG_NAME", "genesis_workbench")
    if schema is None:
        schema = os.environ.get("CORE_SCHEMA_NAME", "genesis_schema")

    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    if isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], (list, tuple)):
        embedding = embedding[0]
    embedding = [float(x) for x in embedding]

    index_name = f"{catalog}.{schema}.{TEDDY_CELL_INDEX}"
    result = workspace_client.vector_search_indexes.query_index(
        index_name=index_name,
        columns=["cell_id"],
        query_vector=embedding,
        num_results=k,
    )
    cell_ids = []
    if result.result and result.result.data_array:
        for row in result.result.data_array:
            cell_ids.append(row[0])
    return _fetch_neighbor_metadata(cell_ids, catalog, schema)


def _summarize_topk(counter: Counter, total: int, k: int = 3) -> str:
    if not counter or total == 0:
        return ""
    top = counter.most_common(k)
    return "; ".join(f"{lbl} ({c / total:.0%})" for lbl, c in top)


def annotate_clusters(
    markers_df: pd.DataFrame,
    cluster_col: str = "cluster",
    cells_per_cluster: int = 20,
    k_neighbors: int = 50,
    catalog: str = None,
    schema: str = None,
    progress_callback=None,
):
    """End-to-end TEDDY annotation: embed sampled cells, KNN, majority-vote both heads.

    Returns:
        cluster_results_df with columns:
          Cluster, n_cells,
          Predicted Cell Type, Cell Type Confidence, Cell Type Top-3,
          Predicted Disease,   Disease Confidence,   Disease Top-3
    """
    def _progress(pct, text):
        if progress_callback:
            progress_callback(pct, text)

    if cluster_col not in markers_df.columns:
        raise ValueError(f"cluster column '{cluster_col}' not found")
    expr_cols = [c for c in markers_df.columns if c.startswith("expr_")]
    if not expr_cols:
        raise ValueError("markers_df has no expr_* columns")
    gene_names = [c.replace("expr_", "") for c in expr_cols]

    _progress(3, "Sampling cells per cluster…")
    sampled_idx = []
    clusters = sorted(markers_df[cluster_col].unique(), key=lambda x: str(x))
    for cl in clusters:
        cl_idx = markers_df.index[markers_df[cluster_col] == cl]
        sampled_idx.extend(cl_idx[: min(cells_per_cluster, len(cl_idx))].tolist())

    expr_df = markers_df.loc[sampled_idx, expr_cols].copy()
    expr_df.columns = gene_names
    sampled_clusters = markers_df.loc[sampled_idx, cluster_col].tolist()

    emb_df = embed_cells(expr_df, gene_names, progress_callback=_progress)

    # KNN per sampled cell, aggregate per cluster
    cluster_to_celltype_counts = {cl: Counter() for cl in clusters}
    cluster_to_disease_counts = {cl: Counter() for cl in clusters}
    cluster_to_total = {cl: 0 for cl in clusters}

    n_total = len(emb_df)
    for i, (idx, row) in enumerate(emb_df.iterrows()):
        _progress(55 + int(40 * (i + 1) / max(1, n_total)),
                  f"KNN for cell {i + 1}/{n_total}…")
        cl = sampled_clusters[i]
        try:
            meta_df = search_nearest_cells(row["embedding"], k=k_neighbors,
                                            catalog=catalog, schema=schema)
        except Exception as e:
            logger.warning(f"KNN failed for sampled cell {i}: {e}")
            continue
        if meta_df.empty:
            continue
        if "cell_type" in meta_df.columns:
            cluster_to_celltype_counts[cl].update(meta_df["cell_type"].dropna().tolist())
        if "disease" in meta_df.columns:
            cluster_to_disease_counts[cl].update(meta_df["disease"].dropna().tolist())
        cluster_to_total[cl] += len(meta_df)

    rows = []
    for cl in clusters:
        ct_counts = cluster_to_celltype_counts[cl]
        ds_counts = cluster_to_disease_counts[cl]
        total = cluster_to_total[cl]
        n_cells = sum(1 for sc in sampled_clusters if str(sc) == str(cl))

        if ct_counts:
            top_ct, top_ct_n = ct_counts.most_common(1)[0]
            ct_conf = top_ct_n / total
        else:
            top_ct, ct_conf = "Unknown", 0.0
        if ds_counts:
            top_ds, top_ds_n = ds_counts.most_common(1)[0]
            ds_conf = top_ds_n / total
        else:
            top_ds, ds_conf = "Unknown", 0.0

        rows.append({
            "Cluster": cl,
            "n_cells": n_cells,
            "Predicted Cell Type": top_ct,
            "Cell Type Confidence": f"{ct_conf:.0%}",
            "Cell Type Top-3": _summarize_topk(ct_counts, total),
            "Predicted Disease": top_ds,
            "Disease Confidence": f"{ds_conf:.0%}",
            "Disease Top-3": _summarize_topk(ds_counts, total),
        })

    _progress(100, "TEDDY annotation complete")
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Cluster", key=lambda s: s.astype(str)).reset_index(drop=True)
    return out

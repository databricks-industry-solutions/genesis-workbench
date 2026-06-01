"""SCimilarity annotation pipeline ported from
modules/core/app/utils/scimilarity_tools.py.

Talks to three SCimilarity endpoints (Gene_Order, Get_Embedding) plus the
Vector Search index over the externalized scimilarity_cells reference. All
calls use the app SP (OBO tokens lack model-serving scope)."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from databricks.sdk import WorkspaceClient

from app.config import get_settings
from app.services.endpoints import get_endpoint_name as _endpoint_for_display
from app.services.workbench import execute_select_query

logger = logging.getLogger(__name__)

SCIMILARITY_CELLS_TABLE = "scimilarity_cells"
SCIMILARITY_CELL_INDEX = "scimilarity_cell_index"


def _endpoint_for(suffix_display: str) -> str:
    """suffix_display is the *Display* name from DISPLAY_TO_UC
    (e.g. 'SCimilarity Gene Order' → 'scimilarity_gene_order')."""
    return _endpoint_for_display(suffix_display)


def _query_endpoint(endpoint_name: str, payload):
    w = WorkspaceClient()
    response = w.serving_endpoints.query(name=endpoint_name, inputs=payload)
    return response.predictions


def align_to_gene_order(expression_df: pd.DataFrame, gene_order: list[str]) -> pd.DataFrame:
    missing = [g for g in gene_order if g not in expression_df.columns]
    if missing:
        zeros = pd.DataFrame(0.0, index=expression_df.index, columns=missing)
        expression_df = pd.concat([expression_df, zeros], axis=1)
    return expression_df[gene_order]


def lognorm_counts(expression_df: pd.DataFrame, target_sum: float = 1e4) -> pd.DataFrame:
    row_sums = expression_df.sum(axis=1).replace(0, 1)
    normed = expression_df.div(row_sums, axis=0) * target_sum
    return np.log1p(normed)


def get_gene_order() -> list[str]:
    endpoint = _endpoint_for("SCimilarity Gene Order")
    result = _query_endpoint(endpoint, {"input": ["get_gene_order"]})
    if isinstance(result, list) and result:
        genes = result[0] if isinstance(result[0], list) else result
    else:
        genes = result
    return [str(g) for g in genes]


def get_cell_embeddings(normed_df: pd.DataFrame) -> pd.DataFrame:
    endpoint = _endpoint_for("SCimilarity Get Embedding")
    dense_rows = normed_df.values.tolist()
    celltype_subsample_pdf = pd.DataFrame(
        [{"celltype_subsample": row} for row in dense_rows],
        index=normed_df.index,
    )
    celltype_sample_json = celltype_subsample_pdf.to_json(orient="split")
    obs_json = pd.DataFrame(index=normed_df.index).to_json(orient="split")
    payload = [{"celltype_sample": celltype_sample_json, "celltype_sample_obs": obs_json}]
    result = _query_endpoint(endpoint, payload)
    if result is None:
        raise RuntimeError(f"GetEmbedding endpoint returned None for {len(dense_rows)} cells")
    if isinstance(result, (dict, list)):
        return pd.DataFrame(result)
    raise RuntimeError(f"GetEmbedding unexpected response type: {type(result)}")


def search_nearest_cells(embedding, k: int = 100) -> pd.DataFrame:
    s = get_settings()
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    if isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], (list, tuple)):
        embedding = embedding[0]
    embedding = [float(x) for x in embedding]

    index_name = f"{s.catalog}.{s.schema}.{SCIMILARITY_CELL_INDEX}"
    w = WorkspaceClient()
    results = w.vector_search_indexes.query_index(
        index_name=index_name,
        columns=["cell_id"],
        query_vector=embedding,
        num_results=k,
    )

    cell_ids: list[str] = []
    if results.result and results.result.data_array:
        for row in results.result.data_array:
            cell_ids.append(row[0])
    return _fetch_cell_metadata(cell_ids, s.catalog, s.schema)


def _fetch_cell_metadata(cell_ids: list[str], catalog: str, schema: str) -> pd.DataFrame:
    if not cell_ids:
        return pd.DataFrame()
    ids_str = ", ".join(f"'{cid}'" for cid in cell_ids)
    query = (
        f"SELECT cell_id, prediction, disease, tissue, study "
        f"FROM {catalog}.{schema}.{SCIMILARITY_CELLS_TABLE} "
        f"WHERE cell_id IN ({ids_str})"
    )
    df = execute_select_query(query)
    if not df.empty and "cell_id" in df.columns:
        df = df.set_index("cell_id").reindex(cell_ids).reset_index()
    return df


def annotate_clusters(
    markers_df: pd.DataFrame,
    cluster_col: str = "cluster",
    cells_per_cluster: int = 30,
    k_neighbors: int = 100,
    progress_callback=None,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Full SCimilarity annotation pipeline. Returns (cluster_results_df,
    cluster_to_type, warnings) — the second is a flat map for UMAP overlay;
    `warnings` is a per-batch failure summary that callers should surface in
    their response so the user can tell when annotation succeeded with
    partial data.

    `progress_callback(pct: int, msg: str)` is invoked between phases with
    real progress (0-100). Used by the SSE route for live updates."""
    def _p(pct: int, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    _p(2, "Fetching SCimilarity gene order")
    gene_order = get_gene_order()

    expr_cols = [c for c in markers_df.columns if c.startswith("expr_")]
    gene_names = [c.replace("expr_", "") for c in expr_cols]

    expr_df = markers_df[expr_cols].copy()
    expr_df.columns = gene_names
    aligned = align_to_gene_order(expr_df, gene_order)
    normed = lognorm_counts(aligned)

    clusters = sorted(
        markers_df[cluster_col].unique(),
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )

    sampled_indices: list = []
    for cl in clusters:
        cl_idx = markers_df.index[markers_df[cluster_col] == cl]
        n = min(cells_per_cluster, len(cl_idx))
        sampled_indices.extend(cl_idx[:n].tolist())

    sampled_normed = normed.loc[sampled_indices]
    sampled_clusters = markers_df.loc[sampled_indices, cluster_col].values

    EMBEDDING_BATCH_SIZE = 5
    total_sampled = len(sampled_indices)
    _p(8, f"Sampled {total_sampled} cells across {len(clusters)} clusters")
    embedding_frames: list[pd.DataFrame] = []
    # Parallel to embedding_frames after concat. We can't reuse sampled_indices /
    # sampled_clusters once a batch is skipped — positions in `embeddings_result`
    # no longer align with their slot in the original sampled list.
    successful_indices: list = []
    successful_clusters: list = []
    skipped_batches: list[tuple[int, int, str]] = []
    for batch_start in range(0, total_sampled, EMBEDDING_BATCH_SIZE):
        batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, total_sampled)
        batch_normed = sampled_normed.iloc[batch_start:batch_end]
        try:
            batch_result = get_cell_embeddings(batch_normed)
            if batch_result.empty:
                raise RuntimeError(f"Empty result for batch {batch_start}-{batch_end}")
            embedding_frames.append(batch_result)
            successful_indices.extend(sampled_indices[batch_start:batch_end])
            successful_clusters.extend(sampled_clusters[batch_start:batch_end].tolist())
        except Exception as e:
            # Per-batch skip: log + continue. Downstream cluster voting tolerates
            # missing cells — one bad batch just removes those cells from the
            # vote, not the whole annotation. If EVERY batch fails we still
            # raise below.
            msg = f"Embedding failed for cells {batch_start}-{batch_end}: {e}"
            logger.warning(msg)
            skipped_batches.append((batch_start, batch_end, str(e).splitlines()[0][:200]))
        # 10% → 55% spread across embedding batches.
        pct = 10 + int((batch_end / total_sampled) * 45)
        _p(pct, f"Embedded {batch_end}/{total_sampled} cells")
    if not embedding_frames:
        first_err = skipped_batches[0][2] if skipped_batches else "no batches"
        raise RuntimeError(
            f"All {len(skipped_batches)} embedding batch(es) failed; cannot annotate. "
            f"First error: {first_err}"
        )

    embeddings_result = pd.concat(embedding_frames, ignore_index=True)

    annotations: list[dict] = []
    n_emb = len(embeddings_result)
    for i, row in embeddings_result.iterrows():
        embedding = row["embedding"]
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        try:
            nn_meta = search_nearest_cells(embedding, k=k_neighbors)
        except Exception as e:
            logger.warning("search_nearest_cells failed for cell %d: %s", i, e)
            nn_meta = pd.DataFrame()
        annotations.append(
            {
                "cell_index": successful_indices[i] if i < len(successful_indices) else i,
                "cluster": successful_clusters[i] if i < len(successful_clusters) else "?",
                "nn_metadata": nn_meta,
            }
        )
        # 55% → 95% spread across per-cell KNN lookups.
        pct = 55 + int(((i + 1) / max(n_emb, 1)) * 40)
        _p(pct, f"Vector Search neighbours {i + 1}/{n_emb}")

    _p(96, "Aggregating votes per cluster")
    cluster_results: list[dict] = []
    cluster_to_type: dict[str, str] = {}
    for cl in clusters:
        cl_cells = [a for a in annotations if str(a["cluster"]) == str(cl)]
        all_predictions: list[str] = []
        for cell in cl_cells:
            meta = cell["nn_metadata"]
            if not meta.empty and "prediction" in meta.columns:
                all_predictions.extend(meta["prediction"].dropna().tolist())

        if all_predictions:
            type_counts = pd.Series(all_predictions).value_counts()
            total = len(all_predictions)
            top_type = str(type_counts.index[0])
            confidence = type_counts.iloc[0] / total
            top_3 = type_counts.head(3)
            top_types_str = "; ".join(f"{t} ({c / total:.0%})" for t, c in top_3.items())
        else:
            top_type = "Unknown"
            confidence = 0.0
            top_types_str = "No neighbors found"

        cluster_results.append(
            {
                "cluster": str(cl),
                "predicted_cell_type": top_type,
                "confidence_pct": round(confidence * 100, 1),
                "top_predictions": top_types_str,
            }
        )
        cluster_to_type[str(cl)] = top_type

    _p(100, f"Annotated {len(cluster_results)} clusters")
    warnings: list[str] = [
        f"Skipped cells {s}-{e}: {err}" for s, e, err in skipped_batches
    ]
    return pd.DataFrame(cluster_results), cluster_to_type, warnings

import os
import json
import logging
import numpy as np
import pandas as pd
from databricks.sdk import WorkspaceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

workspace_client = WorkspaceClient()


def _get_endpoint_name(model_suffix: str) -> str:
    """Resolve endpoint name with optional dev_user prefix."""
    dev_user_prefix = os.environ.get("DEV_USER_PREFIX", "")
    if dev_user_prefix and dev_user_prefix.lower() not in ("none", ""):
        return f"gwb_{dev_user_prefix}_{model_suffix}_endpoint"
    return f"gwb_{model_suffix}_endpoint"


def _query_endpoint(endpoint_name: str, payload):
    """Query a serving endpoint and return the raw response.

    The Databricks SDK query() only supports the `inputs=` kwarg.
    Pass the payload directly as a list or dict — the SDK handles serialization.
    """
    logger.info(f"Querying endpoint: {endpoint_name}")
    response = workspace_client.serving_endpoints.query(
        name=endpoint_name,
        inputs=payload,
    )
    return response.predictions


# ---------------------------------------------------------------------------
# Lightweight reimplementations of scimilarity preprocessing so we don't need
# the full scimilarity library in the Streamlit app.
# ---------------------------------------------------------------------------

def align_to_gene_order(expression_df: pd.DataFrame, gene_order: list) -> pd.DataFrame:
    """Reorder columns of expression_df to match gene_order, filling missing genes with 0."""
    missing = [g for g in gene_order if g not in expression_df.columns]
    if missing:
        zeros = pd.DataFrame(0.0, index=expression_df.index, columns=missing)
        expression_df = pd.concat([expression_df, zeros], axis=1)
    return expression_df[gene_order]


def lognorm_counts(expression_df: pd.DataFrame, target_sum: float = 1e4) -> pd.DataFrame:
    """Library-size normalize and log1p transform an expression matrix."""
    row_sums = expression_df.sum(axis=1).replace(0, 1)
    normed = expression_df.div(row_sums, axis=0) * target_sum
    return np.log1p(normed)


# ---------------------------------------------------------------------------
# SCimilarity endpoint wrappers
# ---------------------------------------------------------------------------

def get_gene_order() -> list:
    """Fetch the canonical gene order list from the SCimilarity GeneOrder endpoint."""
    endpoint = _get_endpoint_name("scimilarity_gene_order")
    # GeneOrder expects a simple dict with an "input" key
    result = _query_endpoint(endpoint, {"input": ["get_gene_order"]})
    if isinstance(result, list) and len(result) > 0:
        genes = result[0] if isinstance(result[0], list) else result
    else:
        genes = result
    return genes


def get_cell_embeddings(normed_df: pd.DataFrame, obs_df: pd.DataFrame = None) -> pd.DataFrame:
    """Generate 128-D cell embeddings via the SCimilarity GetEmbedding endpoint.

    The endpoint expects celltype_sample as a JSON DataFrame where each row has a
    'celltype_subsample' column containing a list of expression values (one per gene).

    Args:
        normed_df: Aligned and log-normalized expression DataFrame (cells x genes)
        obs_df: Optional cell metadata DataFrame with same index as normed_df

    Returns:
        DataFrame with columns: input_index, celltype_sample_index, embedding, + metadata
    """
    endpoint = _get_endpoint_name("scimilarity_get_embedding")

    # Build the celltype_subsample wrapper: each row is a list of expression values
    dense_rows = normed_df.values.tolist()
    celltype_subsample_pdf = pd.DataFrame(
        [{"celltype_subsample": row} for row in dense_rows],
        index=normed_df.index,
    )
    celltype_sample_json = celltype_subsample_pdf.to_json(orient="split")

    # The model's predict() calls pd.read_json(obs_json, orient='split') — passing
    # "null" causes NoneType.items() error. Build a minimal obs DataFrame with the
    # same index so the model can merge correctly.
    if obs_df is not None:
        obs_json = obs_df.to_json(orient="split")
    else:
        empty_obs = pd.DataFrame(index=normed_df.index)
        obs_json = empty_obs.to_json(orient="split")

    payload = [{
        "celltype_sample": celltype_sample_json,
        "celltype_sample_obs": obs_json,
    }]
    logger.info(f"GetEmbedding request: {len(dense_rows)} cells, payload size ~{len(celltype_sample_json)} chars")
    result = _query_endpoint(endpoint, payload)
    logger.info(f"GetEmbedding response type: {type(result)}, preview: {str(result)[:200]}")
    if result is None:
        raise RuntimeError(f"GetEmbedding endpoint returned None for {len(dense_rows)} cells")
    if isinstance(result, dict):
        return pd.DataFrame(result)
    if isinstance(result, list):
        return pd.DataFrame(result)
    raise RuntimeError(f"GetEmbedding unexpected response type: {type(result)}")


def search_nearest_cells(embedding: list, k: int = 100) -> dict:
    """Search the 23M-cell reference database for k nearest neighbors.

    Args:
        embedding: 128-element list of floats
        k: Number of nearest neighbors

    Returns:
        dict with nn_idxs, nn_dists, results_metadata (JSON string)
    """
    endpoint = _get_endpoint_name("scimilarity_search_nearest")
    # Flatten embedding if needed
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    if isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], (list, tuple)):
        embedding = embedding[0]
    embedding = [float(x) for x in embedding]

    payload = {
        "embedding": embedding,
        "params": json.dumps({"k": k}),
    }
    result = _query_endpoint(endpoint, payload)
    # Result is typically a list with one dict; extract the first
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        return result[0]
    return result


def annotate_clusters(
    markers_df: pd.DataFrame,
    cluster_col: str = "cluster",
    cells_per_cluster: int = 30,
    k_neighbors: int = 100,
    progress_callback=None,
) -> pd.DataFrame:
    """Run the full SCimilarity annotation pipeline on a markers_flat DataFrame.

    Steps:
      1. Get gene order from GeneOrder endpoint
      2. Build expression matrix from expr_ columns, align to gene order, lognorm
      3. Sample cells per cluster
      4. Get embeddings for sampled cells
      5. Search nearest neighbors for each embedding
      6. Majority-vote cell type per cluster

    Args:
        markers_df: markers_flat.parquet DataFrame (must have cluster col and expr_ cols)
        cluster_col: Name of the cluster column
        cells_per_cluster: Number of cells to sample per cluster
        k_neighbors: Number of nearest neighbors per cell
        progress_callback: Optional callable(pct, text) for progress updates

    Returns:
        DataFrame with columns: cluster, predicted_cell_type, confidence, top_types
    """
    def _progress(pct, text):
        if progress_callback:
            progress_callback(pct, text)

    # Step 1: Gene order
    _progress(5, "Fetching gene order from SCimilarity...")
    gene_order = get_gene_order()

    # Step 2: Build expression matrix from expr_ columns
    _progress(10, "Preparing expression matrix...")
    expr_cols = [c for c in markers_df.columns if c.startswith("expr_")]
    gene_names = [c.replace("expr_", "") for c in expr_cols]

    expr_df = markers_df[expr_cols].copy()
    expr_df.columns = gene_names

    # Align to gene order and lognorm
    aligned = align_to_gene_order(expr_df, gene_order)
    normed = lognorm_counts(aligned)

    # Step 3: Sample cells per cluster
    clusters = sorted(markers_df[cluster_col].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
    sampled_indices = []
    for cl in clusters:
        cl_idx = markers_df.index[markers_df[cluster_col] == cl]
        n = min(cells_per_cluster, len(cl_idx))
        sampled_indices.extend(cl_idx[:n].tolist())

    sampled_normed = normed.loc[sampled_indices]
    sampled_clusters = markers_df.loc[sampled_indices, cluster_col].values

    # Step 4: Get embeddings in batches to stay under the 16MB request limit.
    # Each cell has ~18K gene values; at ~10 bytes per float in JSON, one cell is ~180KB.
    # Batch size of 5 keeps each request well under 16MB.
    EMBEDDING_BATCH_SIZE = 5
    total_sampled = len(sampled_indices)
    _progress(25, f"Generating embeddings for {total_sampled} sampled cells...")

    all_embedding_dfs = []
    for batch_start in range(0, total_sampled, EMBEDDING_BATCH_SIZE):
        batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, total_sampled)
        batch_normed = sampled_normed.iloc[batch_start:batch_end]
        batch_pct = 25 + int((batch_end / total_sampled) * 20)
        _progress(batch_pct, f"Generating embeddings ({batch_end}/{total_sampled} cells)...")
        try:
            batch_result = get_cell_embeddings(batch_normed)
            if batch_result.empty:
                raise RuntimeError(f"Empty result for batch {batch_start}-{batch_end}")
            all_embedding_dfs.append(batch_result)
        except Exception as e:
            raise RuntimeError(f"Embedding failed for cells {batch_start}-{batch_end}: {e}") from e

    if not all_embedding_dfs:
        raise RuntimeError("No embeddings were generated for any batch")
    embeddings_result = pd.concat(all_embedding_dfs, ignore_index=True)

    # Step 5: Search nearest neighbors for each cell
    _progress(50, "Searching reference database...")
    all_annotations = []
    total_cells = len(embeddings_result)

    for i, row in embeddings_result.iterrows():
        embedding = row["embedding"]
        if isinstance(embedding, str):
            embedding = json.loads(embedding)

        pct = 50 + int((i + 1) / total_cells * 40)
        _progress(pct, f"Searching neighbors for cell {i + 1}/{total_cells}...")

        try:
            search_result = search_nearest_cells(embedding, k=k_neighbors)

            # Parse metadata
            metadata_json = None
            if isinstance(search_result, dict):
                metadata_json = search_result.get("results_metadata")
            elif isinstance(search_result, list) and len(search_result) > 0:
                first = search_result[0] if isinstance(search_result[0], dict) else search_result
                if isinstance(first, dict):
                    metadata_json = first.get("results_metadata")

            if metadata_json and isinstance(metadata_json, str):
                nn_meta = pd.read_json(metadata_json, orient="split") if "columns" in metadata_json else pd.read_json(metadata_json)
            elif isinstance(metadata_json, (dict, list)):
                nn_meta = pd.DataFrame(metadata_json)
            else:
                nn_meta = pd.DataFrame()

            all_annotations.append({
                "cell_index": sampled_indices[i] if i < len(sampled_indices) else i,
                "cluster": sampled_clusters[i] if i < len(sampled_clusters) else "?",
                "nn_metadata": nn_meta,
            })
        except Exception as e:
            logger.warning(f"Search failed for cell {i}: {e}")
            all_annotations.append({
                "cell_index": sampled_indices[i] if i < len(sampled_indices) else i,
                "cluster": sampled_clusters[i] if i < len(sampled_clusters) else "?",
                "nn_metadata": pd.DataFrame(),
            })

    # Step 6: Majority vote per cluster
    _progress(95, "Computing cluster annotations...")
    cluster_results = []
    for cl in clusters:
        cl_cells = [a for a in all_annotations if str(a["cluster"]) == str(cl)]
        # Collect all neighbor predictions
        all_predictions = []
        for cell in cl_cells:
            meta = cell["nn_metadata"]
            if not meta.empty and "prediction" in meta.columns:
                all_predictions.extend(meta["prediction"].dropna().tolist())

        if all_predictions:
            type_counts = pd.Series(all_predictions).value_counts()
            total = len(all_predictions)
            top_type = type_counts.index[0]
            confidence = type_counts.iloc[0] / total
            top_3 = type_counts.head(3)
            top_types_str = "; ".join(
                f"{t} ({c / total:.0%})" for t, c in top_3.items()
            )
        else:
            top_type = "Unknown"
            confidence = 0.0
            top_types_str = "No neighbors found"

        cluster_results.append({
            "Cluster": cl,
            "Predicted Cell Type": top_type,
            "Confidence": f"{confidence:.0%}",
            "Top Predictions": top_types_str,
        })

    _progress(100, "Annotation complete!")
    return pd.DataFrame(cluster_results)

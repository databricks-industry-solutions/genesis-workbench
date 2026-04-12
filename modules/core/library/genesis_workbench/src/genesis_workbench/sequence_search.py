"""Hybrid Funnel Sequence Search: embedding → vector search → Smith-Waterman alignment.

Provides BLAST-like sequence similarity search on Databricks using:
  1. ESM-2 embedding via Model Serving endpoint
  2. Databricks Vector Search for ANN candidate retrieval
  3. parasail Smith-Waterman for exact alignment scoring
"""

import logging
from typing import List, Optional
from dataclasses import dataclass

from databricks.sdk import WorkspaceClient

from genesis_workbench.workbench import execute_select_query

logger = logging.getLogger(__name__)

# Must match the model used in both the batch embedding pipeline
# and the ESM2 Embeddings serving endpoint
MODEL_TAG = "nvidia/esm2_t33_650M_UR50D"
EMBEDDING_DIM = 1280

# Default resource names
VS_INDEX_NAME_SUFFIX = "sequence_embedding_index"
SEQUENCE_TABLE_SUFFIX = "sequence_db"


@dataclass
class AlignmentHit:
    seq_id: str
    description: str
    seq_length: int
    identity_pct: float
    sw_score: int
    alignment_length: int
    vector_distance: float
    aligned_query: str
    aligned_comp: str
    aligned_target: str


def _get_embedding(sequence: str, endpoint_name: str) -> List[float]:
    """Get 1280d mean-pooled embedding from the ESM2 Embeddings serving endpoint."""
    w = WorkspaceClient()
    response = w.serving_endpoints.query(
        name=endpoint_name,
        inputs=[sequence],
    )
    predictions = response.predictions
    if isinstance(predictions, list) and len(predictions) > 0:
        embedding = predictions[0]
        if isinstance(embedding, dict):
            # Handle nested response format
            embedding = list(embedding.values())[0]
        return embedding
    raise ValueError(f"Unexpected response from embedding endpoint: {predictions}")


def _vector_search(embedding: List[float],
                   index_name: str,
                   num_results: int = 500) -> list:
    """Query the Vector Search index for nearest neighbors."""
    w = WorkspaceClient()
    results = w.vector_search_indexes.query_index(
        index_name=index_name,
        columns=["seq_id"],
        query_vector=embedding,
        num_results=num_results,
    )
    hits = []
    if results.result and results.result.data_array:
        for row in results.result.data_array:
            hits.append({
                "seq_id": row[0],
                "distance": float(row[-1]) if len(row) > 1 else 0.0,
            })
    return hits


def _fetch_sequences(seq_ids: List[str], catalog: str, schema: str) -> dict:
    """Fetch full sequence records from the sequence_db Delta table."""
    if not seq_ids:
        return {}

    # Build IN clause with quoted IDs
    ids_str = ", ".join(f"'{sid}'" for sid in seq_ids)
    query = f"""
        SELECT seq_id, sequence, description, seq_length
        FROM {catalog}.{schema}.{SEQUENCE_TABLE_SUFFIX}
        WHERE seq_id IN ({ids_str})
    """
    df = execute_select_query(query)
    return {row["seq_id"]: row for _, row in df.iterrows()}


def align_candidates(query_seq: str, candidates: list) -> List[AlignmentHit]:
    """Run Smith-Waterman alignment on candidate sequences using parasail.

    Args:
        query_seq: The query amino acid sequence.
        candidates: List of dicts with keys: seq_id, sequence, description, seq_length, distance.

    Returns:
        List of AlignmentHit sorted by SW score descending.
    """
    import parasail

    matrix = parasail.blosum62
    gap_open = 11
    gap_extend = 1

    results = []
    for cand in candidates:
        target_seq = cand["sequence"]
        result = parasail.sw_trace_striped_32(
            query_seq, target_seq, gap_open, gap_extend, matrix
        )

        traceback = result.traceback
        aligned_query = traceback.query
        aligned_ref = traceback.ref
        aligned_comp = traceback.comp

        # Calculate identity percentage
        matches = sum(1 for a, b in zip(aligned_query, aligned_ref) if a == b)
        alignment_length = len(aligned_query)
        identity_pct = (matches / alignment_length * 100) if alignment_length > 0 else 0.0

        results.append(AlignmentHit(
            seq_id=cand["seq_id"],
            description=cand.get("description", ""),
            seq_length=cand.get("seq_length", len(target_seq)),
            identity_pct=round(identity_pct, 1),
            sw_score=result.score,
            alignment_length=alignment_length,
            vector_distance=cand.get("distance", 0.0),
            aligned_query=aligned_query,
            aligned_comp=aligned_comp,
            aligned_target=aligned_ref,
        ))

    results.sort(key=lambda x: x.sw_score, reverse=True)
    return results


def search_sequences(
    query_sequence: str,
    catalog: str,
    schema: str,
    embedding_endpoint_name: str,
    vs_index_name: Optional[str] = None,
    top_k_vector: int = 500,
    top_k_final: int = 50,
    progress_callback=None,
) -> List[AlignmentHit]:
    """Full hybrid funnel: embed → vector search → fetch → align.

    Args:
        query_sequence: Amino acid sequence to search.
        catalog: Unity Catalog name.
        schema: Schema name.
        embedding_endpoint_name: Name of the ESM2 Embeddings serving endpoint.
        vs_index_name: Full name of the Vector Search index (defaults to {catalog}.{schema}.sequence_embedding_index).
        top_k_vector: Number of ANN candidates to retrieve from Vector Search.
        top_k_final: Number of final results to return after alignment.
        progress_callback: Optional callable(progress_pct, status_text) for UI updates.

    Returns:
        List of AlignmentHit sorted by SW score descending.
    """
    if vs_index_name is None:
        vs_index_name = f"{catalog}.{schema}.{VS_INDEX_NAME_SUFFIX}"

    # Stage 1: Embed query
    if progress_callback:
        progress_callback(10, "Embedding sequence...")
    logger.info(f"Embedding query sequence ({len(query_sequence)} residues) via {embedding_endpoint_name}")
    embedding = _get_embedding(query_sequence, embedding_endpoint_name)

    # Stage 2: Vector Search
    if progress_callback:
        progress_callback(33, "Searching sequence database...")
    logger.info(f"Querying Vector Search index {vs_index_name} for top {top_k_vector} candidates")
    vs_hits = _vector_search(embedding, vs_index_name, num_results=top_k_vector)
    logger.info(f"Vector Search returned {len(vs_hits)} candidates")

    if not vs_hits:
        if progress_callback:
            progress_callback(100, "No results found")
        return []

    # Stage 3: Fetch full sequences
    if progress_callback:
        progress_callback(50, "Fetching candidate sequences...")
    seq_ids = [h["seq_id"] for h in vs_hits]
    seq_records = _fetch_sequences(seq_ids, catalog, schema)
    logger.info(f"Fetched {len(seq_records)} sequences from Delta table")

    # Build candidate list with full data
    candidates = []
    for hit in vs_hits:
        sid = hit["seq_id"]
        if sid in seq_records:
            rec = seq_records[sid]
            candidates.append({
                "seq_id": sid,
                "sequence": rec["sequence"],
                "description": rec["description"],
                "seq_length": rec["seq_length"],
                "distance": hit["distance"],
            })

    # Stage 4: Smith-Waterman alignment
    if progress_callback:
        progress_callback(70, f"Aligning {len(candidates)} candidates...")
    logger.info(f"Running Smith-Waterman alignment on {len(candidates)} candidates")
    aligned_results = align_candidates(query_sequence, candidates)

    if progress_callback:
        progress_callback(100, "Search complete")

    return aligned_results[:top_k_final]

"""Hybrid Funnel Sequence Search utilities for Streamlit.

Provides BLAST-like sequence similarity search on Databricks using:
  1. ESM-2 embedding via Model Serving endpoint
  2. Databricks Vector Search for ANN candidate retrieval
  3. parasail Smith-Waterman for exact alignment scoring
"""

import os
import logging
from typing import List, Optional
from dataclasses import dataclass

from databricks.sdk import WorkspaceClient

from genesis_workbench.workbench import execute_select_query
from .streamlit_helper import get_endpoint_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1280
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

    ids_str = ", ".join(f"'{sid}'" for sid in seq_ids)
    query = f"""
        SELECT seq_id, sequence, description, seq_length
        FROM {catalog}.{schema}.{SEQUENCE_TABLE_SUFFIX}
        WHERE seq_id IN ({ids_str})
    """
    df = execute_select_query(query)
    return {row["seq_id"]: row for _, row in df.iterrows()}


def align_candidates(query_seq: str, candidates: list) -> List[AlignmentHit]:
    """Run Smith-Waterman alignment on candidate sequences using parasail."""
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
    """Full hybrid funnel: embed -> vector search -> fetch -> align."""
    if vs_index_name is None:
        vs_index_name = f"{catalog}.{schema}.{VS_INDEX_NAME_SUFFIX}"

    if progress_callback:
        progress_callback(10, "Embedding sequence...")
    logger.info(f"Embedding query sequence ({len(query_sequence)} residues) via {embedding_endpoint_name}")
    embedding = _get_embedding(query_sequence, embedding_endpoint_name)

    if progress_callback:
        progress_callback(33, "Searching sequence database...")
    logger.info(f"Querying Vector Search index {vs_index_name} for top {top_k_vector} candidates")
    vs_hits = _vector_search(embedding, vs_index_name, num_results=top_k_vector)
    logger.info(f"Vector Search returned {len(vs_hits)} candidates")

    if not vs_hits:
        if progress_callback:
            progress_callback(100, "No results found")
        return []

    if progress_callback:
        progress_callback(50, "Fetching candidate sequences...")
    seq_ids = [h["seq_id"] for h in vs_hits]
    seq_records = _fetch_sequences(seq_ids, catalog, schema)
    logger.info(f"Fetched {len(seq_records)} sequences from Delta table")

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

    if progress_callback:
        progress_callback(70, f"Aligning {len(candidates)} candidates...")
    logger.info(f"Running Smith-Waterman alignment on {len(candidates)} candidates")
    aligned_results = align_candidates(query_sequence, candidates)

    if progress_callback:
        progress_callback(100, "Search complete")

    return aligned_results[:top_k_final]


def run_sequence_search(query_sequence: str,
                        top_k: int = 50,
                        progress_callback=None) -> List[AlignmentHit]:
    """Run a full hybrid funnel sequence search (convenience wrapper for Streamlit)."""
    catalog = os.environ.get("CORE_CATALOG_NAME", "genesis_workbench")
    schema = os.environ.get("CORE_SCHEMA_NAME", "genesis_schema")
    embedding_endpoint = get_endpoint_name("ESM2 Embeddings")

    logger.info(f"Starting sequence search: query length={len(query_sequence)}, "
                f"top_k={top_k}, endpoint={embedding_endpoint}")

    results = search_sequences(
        query_sequence=query_sequence,
        catalog=catalog,
        schema=schema,
        embedding_endpoint_name=embedding_endpoint,
        top_k_vector=min(top_k * 10, 1000),
        top_k_final=top_k,
        progress_callback=progress_callback,
    )

    logger.info(f"Search complete: {len(results)} results")
    return results

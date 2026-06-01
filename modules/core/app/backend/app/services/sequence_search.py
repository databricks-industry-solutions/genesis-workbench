"""Hybrid-funnel sequence search ported from
modules/core/app/utils/sequence_search_tools.py.

5-stage funnel: ESM-2 embed → Vector Search ANN → Delta fetch → parasail
Smith-Waterman align → rank. Runs synchronously in the FastAPI threadpool;
total wall time ~3-5s for top_k=50, well under the Databricks Apps proxy
timeout."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from databricks.sdk import WorkspaceClient

from app.config import get_settings
from app.services.endpoints import get_endpoint_name
from app.services.workbench import execute_select_query

logger = logging.getLogger(__name__)

VS_INDEX_SUFFIX = "sequence_embedding_index"
SEQUENCE_TABLE_SUFFIX = "sequence_db"


@dataclass(frozen=True)
class AlignmentHit:
    seq_id: str
    description: str
    seq_length: int
    identity_pct: float
    sw_score: int
    alignment_length: int
    # Fraction of the query that was actually aligned, 0-100. Lets us rank
    # by "global-ish" similarity rather than raw local SW score — a 30aa
    # target with a perfect 30/30 match against a 200aa query is a low-
    # coverage hit and shouldn't outrank a 180/200 medium-quality hit.
    query_coverage_pct: float
    # identity_pct * (clamped coverage) — single composite used as the
    # primary sort key. Higher = better.
    similarity_score: float
    vector_distance: float
    aligned_query: str
    aligned_comp: str
    aligned_target: str


def _embed(sequence: str, endpoint_name: str) -> list[float]:
    w = WorkspaceClient()
    response = w.serving_endpoints.query(name=endpoint_name, inputs=[sequence])
    predictions = response.predictions
    if isinstance(predictions, list) and predictions:
        embedding = predictions[0]
        if isinstance(embedding, dict):
            embedding = list(embedding.values())[0]
        return embedding
    raise ValueError(f"Unexpected response from embedding endpoint: {predictions!r}")


def _ann(embedding: list[float], index_name: str, num_results: int) -> list[dict]:
    w = WorkspaceClient()
    result = w.vector_search_indexes.query_index(
        index_name=index_name,
        columns=["seq_id"],
        query_vector=embedding,
        num_results=num_results,
    )
    out: list[dict] = []
    if result.result and result.result.data_array:
        for row in result.result.data_array:
            out.append({"seq_id": row[0], "distance": float(row[-1]) if len(row) > 1 else 0.0})
    return out


def _fetch(seq_ids: list[str], catalog: str, schema: str) -> dict[str, dict]:
    if not seq_ids:
        return {}
    ids_str = ", ".join(f"'{sid}'" for sid in seq_ids)
    query = (
        f"SELECT seq_id, sequence, description, seq_length "
        f"FROM {catalog}.{schema}.{SEQUENCE_TABLE_SUFFIX} "
        f"WHERE seq_id IN ({ids_str})"
    )
    df = execute_select_query(query)
    return {row["seq_id"]: row for _, row in df.iterrows()}


def _align(
    query_seq: str,
    candidates: list[dict],
    progress_callback=None,
    pct_start: int = 60,
    pct_end: int = 95,
) -> list[AlignmentHit]:
    import parasail

    matrix = parasail.blosum62
    gap_open = 11
    gap_extend = 1
    query_len = max(len(query_seq), 1)
    out: list[AlignmentHit] = []
    n = max(len(candidates), 1)
    for i, cand in enumerate(candidates):
        target_seq = cand["sequence"]
        res = parasail.sw_trace_striped_32(query_seq, target_seq, gap_open, gap_extend, matrix)
        tb = res.traceback
        aligned_query = tb.query
        aligned_target = tb.ref
        aligned_comp = tb.comp
        matches = sum(1 for a, b in zip(aligned_query, aligned_target) if a == b)
        alignment_length = len(aligned_query)
        identity_pct = (matches / alignment_length * 100) if alignment_length else 0.0
        # Aligned-query characters excluding gaps — better signal than the
        # raw alignment_length for coverage, since SW pads the local match
        # with gap characters when the target is short.
        aligned_query_no_gaps = sum(1 for c in aligned_query if c != "-")
        coverage_pct = min(aligned_query_no_gaps / query_len, 1.0) * 100.0
        similarity = identity_pct * (coverage_pct / 100.0)
        out.append(
            AlignmentHit(
                seq_id=cand["seq_id"],
                description=cand.get("description", ""),
                seq_length=cand.get("seq_length", len(target_seq)),
                identity_pct=round(identity_pct, 1),
                sw_score=res.score,
                alignment_length=alignment_length,
                query_coverage_pct=round(coverage_pct, 1),
                similarity_score=round(similarity, 2),
                vector_distance=cand.get("distance", 0.0),
                aligned_query=aligned_query,
                aligned_comp=aligned_comp,
                aligned_target=aligned_target,
            )
        )
        if progress_callback:
            pct = pct_start + int(((i + 1) / n) * (pct_end - pct_start))
            progress_callback(pct, f"Smith-Waterman alignment {i + 1}/{n}")
    # Primary sort by composite similarity (identity × coverage). SW score
    # is tiebreaker so two equally-good hits with the same identity and
    # coverage still order by absolute alignment strength.
    out.sort(key=lambda h: (h.similarity_score, h.sw_score), reverse=True)
    return out


def run_sequence_search(
    query_sequence: str,
    top_k: int = 50,
    min_coverage_pct: float = 0.0,
    progress_callback=None,
) -> list[AlignmentHit]:
    """`min_coverage_pct` (0-100) drops candidates whose aligned-query
    coverage is below the threshold — useful when the query is long and the
    user doesn't want tiny perfect-match fragments cluttering the top-K."""
    s = get_settings()
    embedding_endpoint = get_endpoint_name("ESM2 Embeddings")
    vs_index = f"{s.catalog}.{s.schema}.{VS_INDEX_SUFFIX}"
    top_k_vector = min(top_k * 10, 1000)

    def _p(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    logger.info(
        "sequence_search: query_len=%d top_k=%d vs_index=%s endpoint=%s",
        len(query_sequence),
        top_k,
        vs_index,
        embedding_endpoint,
    )

    _p(5, f"Embedding query sequence ({len(query_sequence)} aa)")
    embedding = _embed(query_sequence, embedding_endpoint)
    _p(35, f"Vector Search ({top_k_vector} candidates)")
    vs_hits = _ann(embedding, vs_index, num_results=top_k_vector)
    if not vs_hits:
        _p(100, "No vector-search hits")
        return []

    _p(50, f"Fetching {len(vs_hits)} candidate sequences")
    seq_records = _fetch([h["seq_id"] for h in vs_hits], s.catalog, s.schema)
    candidates: list[dict] = []
    for hit in vs_hits:
        sid = hit["seq_id"]
        if sid in seq_records:
            rec = seq_records[sid]
            candidates.append(
                {
                    "seq_id": sid,
                    "sequence": rec["sequence"],
                    "description": rec["description"],
                    "seq_length": rec["seq_length"],
                    "distance": hit["distance"],
                }
            )
    _p(60, f"Aligning {len(candidates)} sequences (Smith-Waterman)")
    aligned = _align(query_sequence, candidates, progress_callback=progress_callback)
    if min_coverage_pct > 0:
        before = len(aligned)
        aligned = [h for h in aligned if h.query_coverage_pct >= min_coverage_pct]
        logger.info(
            "sequence_search: coverage filter %d%% kept %d/%d hits",
            int(min_coverage_pct), len(aligned), before,
        )
    aligned = aligned[:top_k]
    _p(100, f"Returned top {len(aligned)} hits")
    return aligned


def extract_organism(description: str, llm_endpoint_name: str) -> str:
    if not description.strip():
        return "Unknown"
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    w = WorkspaceClient()
    try:
        response = w.serving_endpoints.query(
            name=llm_endpoint_name,
            messages=[
                ChatMessage(
                    role=ChatMessageRole.SYSTEM,
                    content=(
                        "Extract the organism name from the protein sequence description. "
                        "Return ONLY the organism name (e.g. 'Homo sapiens', 'Escherichia coli'). "
                        "If you cannot determine the organism, return 'Unknown'."
                    ),
                ),
                ChatMessage(role=ChatMessageRole.USER, content=description),
            ],
            max_tokens=50,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("extract_organism failed: %s", e)
        return "Unknown"

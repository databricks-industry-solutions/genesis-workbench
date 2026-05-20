"""Client helpers for TEDDY-driven joint cell-type + disease annotation.

Pipeline (per cell):
  1. Call gwb_teddy_endpoint → 768-d embedding (or 512/1024 depending on variant).
  2. Query teddy_cell_index Vector Search for k nearest neighbors.
  3. Fetch (cell_type, disease) for those neighbors from the teddy_cells Delta.
  4. Majority-vote both columns per source cluster.

Mirrors scimilarity_tools.annotate_clusters but does both vote columns in one pass.
"""
import math
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

# Embedding requests are heavy in two dimensions:
#   - JSON payload size (each cell carries hundreds-to-thousands of expression
#     values); stay under the 16 MB serving request cap.
#   - GPU attention memory inside the endpoint (T4/16GB hosts an attention
#     matrix of (batch × heads × seq² × 4B); for HVG inputs of ~2000 genes,
#     batch=32 → ~4 GB just for attention, which OOMs alongside the model
#     weights + activations.
# 8 cells/request keeps both bounded: ~125 KB payload, ~1 GB GPU peak.
_CELLS_PER_EMBED_REQUEST = 8

# HGNC → ENSG mapping is loaded lazily on first use. Built once via notebook
# `06_extract_gene_mapping.py` from CELLxGENE Census var and stored at
# /Volumes/{catalog}/{schema}/teddy/gene_mapping.json.
_GENE_MAPPING_CACHE = {"loaded": False, "data": {}}

# Per-label reference frequencies for inverse-frequency vote weighting (IDF).
# Loaded lazily from teddy_cells on first annotate_clusters call with
# bias_correct=True. Keyed by catalog.schema so multi-workspace deployments
# don't share caches.
_LABEL_FREQS_CACHE = {}


def _gene_mapping_path() -> str:
    catalog = os.environ.get("CORE_CATALOG_NAME", "genesis_workbench")
    schema = os.environ.get("CORE_SCHEMA_NAME", "genesis_schema")
    return f"/Volumes/{catalog}/{schema}/teddy/gene_mapping.json"


def _load_gene_mapping() -> dict:
    """Lazily load the HGNC → ENSG mapping from the Volume. Returns {} on miss.

    Uses the Databricks SDK Files API because Databricks Apps run in a
    sandboxed container without POSIX/FUSE access to /Volumes paths —
    plain `open("/Volumes/...")` fails silently with FileNotFoundError,
    leaves the mapping empty, and every input gene becomes <unk> at the
    endpoint's tokenizer, which collapses all embeddings to a constant.
    (This is the same pattern `processing._load_gmt` uses for GMT files.)
    """
    if _GENE_MAPPING_CACHE["loaded"]:
        return _GENE_MAPPING_CACHE["data"]
    path = _gene_mapping_path()
    try:
        ws = workspace_client
        response = ws.files.download(path)
        mapping = json.loads(response.contents.read().decode("utf-8"))
        logger.info(f"Loaded TEDDY gene mapping: {len(mapping):,} entries from {path}")
        _GENE_MAPPING_CACHE["data"] = mapping
    except Exception as e:  # noqa: BLE001 — includes NotFound and any SDK errors
        logger.warning(
            f"Failed to load TEDDY gene mapping from {path}: {e}. Inputs that "
            f"aren't already ENSG IDs will become <unk> at query time → noisy "
            f"embeddings. Run notebooks/06_extract_gene_mapping.py to populate it."
        )
    _GENE_MAPPING_CACHE["loaded"] = True
    return _GENE_MAPPING_CACHE["data"]


def _translate_gene_names(names: list) -> tuple:
    """Translate HGNC symbols → ENSG IDs where possible.

    Returns (translated_names, stats_dict). Names already in ENSG form
    (`ENSGxxxxxxxxxxx`) pass through unchanged. Unknown names pass through
    unchanged too — the endpoint's tokenizer turns them into <unk>.
    """
    mapping = _load_gene_mapping()
    translated = []
    n_already_ensg = 0
    n_mapped = 0
    n_unmapped = 0
    for n in names:
        s = str(n)
        if s.startswith("ENSG"):
            translated.append(s); n_already_ensg += 1
        else:
            ensg = mapping.get(s)
            if ensg is not None:
                translated.append(ensg); n_mapped += 1
            else:
                translated.append(s); n_unmapped += 1
    stats = {
        "n_total": len(names),
        "n_already_ensg": n_already_ensg,
        "n_mapped_hgnc_to_ensg": n_mapped,
        "n_unmapped": n_unmapped,
    }
    return translated, stats


def _get_endpoint_name() -> str:
    """Look up the TEDDY serving endpoint name from the model_deployments table.

    Source of truth: `deploy_model_endpoint()` writes the actual deployed
    endpoint name into the table at deploy time. We read it back here, rather
    than constructing it from `DEV_USER_PREFIX` + a hardcoded suffix — that
    pattern silently 404s when the env var isn't plumbed (see the May 20
    incident: app-side `_translate_gene_names` silently returned HGNC because
    the env var wasn't bound, embeddings collapsed, every cluster predicted
    glutamatergic neuron).
    """
    from genesis_workbench.models import get_endpoint_name_for_uc_model
    return get_endpoint_name_for_uc_model("teddy")


def _query_endpoint(payload):
    """Query the TEDDY embedding endpoint.

    The Databricks SDK's serving_endpoints.query() only takes `inputs=`; the
    PyFunc's `params` (max_seq_len, pooling) cannot be overridden through this
    code path, so the wrapper's defaults (max_seq_len=2048, pooling="mean")
    are what the batch annotation always gets. Those match the values we
    want — overriding would require a raw HTTP call to /invocations.
    """
    endpoint = _get_endpoint_name()
    return workspace_client.serving_endpoints.query(
        name=endpoint,
        inputs=payload,
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

    # Translate HGNC symbols (TSPAN6, ...) to ENSG IDs once before the loop —
    # TEDDY's vocab is ENSG. Untranslated names pass through and become <unk>
    # at the endpoint's tokenizer (gracefully, not a crash).
    translated_genes, stats = _translate_gene_names(gene_names)
    if stats["n_unmapped"] > 0:
        logger.warning(
            f"TEDDY: {stats['n_unmapped']:,}/{stats['n_total']:,} input gene names "
            f"could not be translated to ENSG (mapped {stats['n_mapped_hgnc_to_ensg']:,} "
            f"from HGNC, {stats['n_already_ensg']:,} were already ENSG). Unmapped "
            f"genes become <unk> and degrade embedding quality."
        )
    _progress(2, f"Translated {stats['n_mapped_hgnc_to_ensg']:,}/{n} HGNC→ENSG")

    rows = []
    batches = [(s, min(s + _CELLS_PER_EMBED_REQUEST, n)) for s in range(0, n, _CELLS_PER_EMBED_REQUEST)]
    for bi, (start, end) in enumerate(batches):
        batch = expr_df.iloc[start:end]
        _progress(
            5 + int(45 * bi / max(1, len(batches))),
            f"Embedding cells {start + 1}-{end} of {n}…",
        )
        var_df = pd.DataFrame({"index": translated_genes})
        payload = [{
            "adata_sparsematrix": batch.values.tolist(),
            "adata_obs": pd.DataFrame(index=batch.index).to_json(orient="split"),
            "adata_var": var_df.to_json(orient="split"),
        }]
        try:
            # Note: max_seq_len + pooling come from the wrapper's defaults (see _query_endpoint docstring).
            result = _query_endpoint(payload)
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


def _summarize_topk_weighted(weights: dict, total_weight: float, k: int = 3) -> str:
    if not weights or total_weight <= 0:
        return ""
    top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return "; ".join(f"{lbl} ({w / total_weight:.0%})" for lbl, w in top)


def _load_label_freqs(catalog: str, schema: str) -> tuple:
    """Lazily load (cell_type → count, disease → count) from teddy_cells.

    Cached per (catalog, schema). Used for inverse-frequency vote weighting in
    annotate_clusters when bias_correct=True. Fails gracefully (returns empty
    dicts) if teddy_cells doesn't exist — caller treats absent counts as 1.
    """
    key = f"{catalog}.{schema}"
    if key in _LABEL_FREQS_CACHE:
        return _LABEL_FREQS_CACHE[key]
    table = f"{catalog}.{schema}.{TEDDY_CELLS_TABLE}"
    try:
        ct_df = execute_select_query(
            f"SELECT cell_type, COUNT(*) AS n FROM {table} GROUP BY cell_type"
        )
        ds_df = execute_select_query(
            f"SELECT disease, COUNT(*) AS n FROM {table} GROUP BY disease"
        )
        ct_freqs = dict(zip(ct_df["cell_type"].astype(str), ct_df["n"].astype(int)))
        ds_freqs = dict(zip(ds_df["disease"].astype(str), ds_df["n"].astype(int)))
        logger.info(
            f"Loaded TEDDY label frequencies: {len(ct_freqs)} cell_types, "
            f"{len(ds_freqs)} diseases from {table}"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not load TEDDY label frequencies from {table}: {e}")
        ct_freqs, ds_freqs = {}, {}
    _LABEL_FREQS_CACHE[key] = (ct_freqs, ds_freqs)
    return ct_freqs, ds_freqs


def _idf_weight(label: str, freqs: dict) -> float:
    """Inverse-document-frequency-style weight: 1 / log(1 + count).

    Common labels in the reference get smaller weights; rare labels get larger.
    Unknown labels (missing from freqs dict) get weight=1 (treated as singleton).
    """
    count = freqs.get(label, 1)
    return 1.0 / math.log1p(max(1, count))


def annotate_clusters(
    markers_df: pd.DataFrame,
    cluster_col: str = "cluster",
    cells_per_cluster: int = 20,
    k_neighbors: int = 50,
    catalog: str = None,
    schema: str = None,
    bias_correct: bool = True,
    progress_callback=None,
):
    """End-to-end TEDDY annotation: embed sampled cells, KNN, majority-vote both heads.

    When `bias_correct=True` (default), each neighbor's vote is weighted by
    1/log(1 + freq_in_reference) so over-represented labels in `teddy_cells`
    (e.g. plasma cells in a disease-only reference) don't dominate.

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

    cat = catalog or os.environ.get("CORE_CATALOG_NAME", "genesis_workbench")
    sch = schema or os.environ.get("CORE_SCHEMA_NAME", "genesis_schema")

    if bias_correct:
        ct_freqs, ds_freqs = _load_label_freqs(cat, sch)
        if ct_freqs:
            sample_ct_weights = [_idf_weight(l, ct_freqs) for l in list(ct_freqs)[:8]]
            logger.info(
                f"TEDDY IDF on: cell_type weights min={min(sample_ct_weights):.3f} "
                f"max={max(sample_ct_weights):.3f} (sample of {len(sample_ct_weights)})"
            )
        else:
            logger.info("TEDDY IDF requested but no reference frequencies loaded; "
                        "falling back to unweighted voting.")
            bias_correct = False
    else:
        ct_freqs, ds_freqs = {}, {}

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

    # KNN per sampled cell, aggregate per cluster. With bias_correct, we store
    # weighted dict[label, float]; without, we keep Counters (cheaper).
    if bias_correct:
        cluster_to_celltype = {cl: {} for cl in clusters}
        cluster_to_disease = {cl: {} for cl in clusters}
    else:
        cluster_to_celltype = {cl: Counter() for cl in clusters}
        cluster_to_disease = {cl: Counter() for cl in clusters}
    cluster_to_total = {cl: 0.0 for cl in clusters}

    n_total = len(emb_df)
    for i, (idx, row) in enumerate(emb_df.iterrows()):
        _progress(55 + int(40 * (i + 1) / max(1, n_total)),
                  f"KNN for cell {i + 1}/{n_total}…")
        cl = sampled_clusters[i]
        try:
            meta_df = search_nearest_cells(row["embedding"], k=k_neighbors,
                                            catalog=cat, schema=sch)
        except Exception as e:
            logger.warning(f"KNN failed for sampled cell {i}: {e}")
            continue
        if meta_df.empty:
            continue

        # Cell type
        if "cell_type" in meta_df.columns:
            for lbl in meta_df["cell_type"].dropna().tolist():
                if bias_correct:
                    w = _idf_weight(lbl, ct_freqs)
                    cluster_to_celltype[cl][lbl] = cluster_to_celltype[cl].get(lbl, 0.0) + w
                    cluster_to_total[cl] += w
                else:
                    cluster_to_celltype[cl][lbl] += 1
                    cluster_to_total[cl] += 1
        # Disease
        if "disease" in meta_df.columns:
            for lbl in meta_df["disease"].dropna().tolist():
                if bias_correct:
                    w = _idf_weight(lbl, ds_freqs)
                    cluster_to_disease[cl][lbl] = cluster_to_disease[cl].get(lbl, 0.0) + w
                else:
                    cluster_to_disease[cl][lbl] += 1

    rows = []
    for cl in clusters:
        ct = cluster_to_celltype[cl]
        ds = cluster_to_disease[cl]
        n_cells = sum(1 for sc in sampled_clusters if str(sc) == str(cl))

        # Cell-type pick — bias_correct decides which path; Counter ⊂ dict so
        # we can't discriminate on isinstance.
        if not ct:
            top_ct, ct_conf, ct_top3 = "Unknown", 0.0, ""
        elif bias_correct:
            ct_total = sum(ct.values())
            top_ct, top_ct_w = max(ct.items(), key=lambda kv: kv[1])
            ct_conf = top_ct_w / ct_total if ct_total > 0 else 0.0
            ct_top3 = _summarize_topk_weighted(ct, ct_total)
        else:
            ct_total = sum(ct.values())
            top_ct, top_ct_n = ct.most_common(1)[0]
            ct_conf = top_ct_n / ct_total if ct_total > 0 else 0.0
            ct_top3 = _summarize_topk(ct, ct_total)

        # Disease pick
        if not ds:
            top_ds, ds_conf, ds_top3 = "Unknown", 0.0, ""
        elif bias_correct:
            ds_total = sum(ds.values())
            top_ds, top_ds_w = max(ds.items(), key=lambda kv: kv[1])
            ds_conf = top_ds_w / ds_total if ds_total > 0 else 0.0
            ds_top3 = _summarize_topk_weighted(ds, ds_total)
        else:
            ds_total = sum(ds.values())
            top_ds, top_ds_n = ds.most_common(1)[0]
            ds_conf = top_ds_n / ds_total if ds_total > 0 else 0.0
            ds_top3 = _summarize_topk(ds, ds_total)

        rows.append({
            "Cluster": cl,
            "n_cells": n_cells,
            "Predicted Cell Type": top_ct,
            "Cell Type Confidence": f"{ct_conf:.0%}",
            "Cell Type Top-3": ct_top3,
            "Predicted Disease": top_ds,
            "Disease Confidence": f"{ds_conf:.0%}",
            "Disease Top-3": ds_top3,
        })

    _progress(100, "TEDDY annotation complete")
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Cluster", key=lambda s: s.astype(str)).reset_index(drop=True)
    return out

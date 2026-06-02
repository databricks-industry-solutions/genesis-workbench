"""TEDDY-driven joint cell-type + disease annotation. Ported from
modules/core/app/utils/teddy_tools.py.

Pipeline per sampled cell:
  1. POST to TEDDY embedding endpoint → d_model embedding
  2. Query teddy_cell_index VS for k nearest neighbours
  3. Fetch (cell_type, disease, tissue, dataset_id) for those neighbours
  4. Majority-vote per cluster, optionally IDF-bias-corrected against the
     reference label frequencies."""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from databricks.sdk import WorkspaceClient

from app.config import get_settings
from app.services.endpoints import get_endpoint_name as _endpoint_for_display
from app.services.workbench import execute_select_query

logger = logging.getLogger(__name__)

TEDDY_CELLS_TABLE = "teddy_cells"
TEDDY_CELL_INDEX = "teddy_cell_index"
_CELLS_PER_EMBED_REQUEST = 8

_GENE_MAPPING_CACHE = {"loaded": False, "data": {}}
_LABEL_FREQS_CACHE: dict[str, tuple[dict[str, int], dict[str, int]]] = {}


def _gene_mapping_path() -> str:
    s = get_settings()
    return f"/Volumes/{s.catalog}/{s.schema}/teddy/gene_mapping.json"


def _load_gene_mapping() -> dict:
    if _GENE_MAPPING_CACHE["loaded"]:
        return _GENE_MAPPING_CACHE["data"]
    path = _gene_mapping_path()
    try:
        w = WorkspaceClient()
        response = w.files.download(path)
        mapping = json.loads(response.contents.read().decode("utf-8"))
        logger.info("Loaded TEDDY gene mapping: %d entries from %s", len(mapping), path)
        _GENE_MAPPING_CACHE["data"] = mapping
    except Exception as e:
        logger.warning(
            "Failed to load TEDDY gene mapping from %s: %s. Unmapped genes will become <unk>.",
            path,
            e,
        )
    _GENE_MAPPING_CACHE["loaded"] = True
    return _GENE_MAPPING_CACHE["data"]


def _translate_gene_names(names: list[str]) -> tuple[list[str], dict[str, int]]:
    mapping = _load_gene_mapping()
    translated: list[str] = []
    n_already = 0
    n_mapped = 0
    n_unmapped = 0
    for n in names:
        s = str(n)
        if s.startswith("ENSG"):
            translated.append(s)
            n_already += 1
        else:
            ensg = mapping.get(s)
            if ensg is not None:
                translated.append(ensg)
                n_mapped += 1
            else:
                translated.append(s)
                n_unmapped += 1
    return translated, {
        "n_total": len(names),
        "n_already_ensg": n_already,
        "n_mapped_hgnc_to_ensg": n_mapped,
        "n_unmapped": n_unmapped,
    }


def _embed(payload: list) -> list:
    endpoint = _endpoint_for_display("TEDDY Annotation")
    w = WorkspaceClient()
    return w.serving_endpoints.query(name=endpoint, inputs=payload).predictions


def embed_cells(
    expr_df: pd.DataFrame,
    gene_names: list[str],
    progress_callback=None,
    pct_start: int = 15,
    pct_end: int = 55,
) -> pd.DataFrame:
    """Embed cells via the TEDDY serving endpoint, batched.

    `progress_callback(pct, msg)` fires after each batch finishes. The bar
    spans `pct_start` → `pct_end` so the caller can budget the remaining
    range for downstream KNN/aggregation phases."""
    if len(expr_df) == 0:
        return pd.DataFrame(columns=["embedding"])
    translated, _stats = _translate_gene_names(gene_names)
    rows: list[dict] = []
    n = len(expr_df)
    for start in range(0, n, _CELLS_PER_EMBED_REQUEST):
        end = min(start + _CELLS_PER_EMBED_REQUEST, n)
        batch = expr_df.iloc[start:end]
        var_df = pd.DataFrame({"index": translated})
        payload = [
            {
                "adata_sparsematrix": batch.values.tolist(),
                "adata_obs": pd.DataFrame(index=batch.index).to_json(orient="split"),
                "adata_var": var_df.to_json(orient="split"),
            }
        ]
        try:
            result = _embed(payload)
        except Exception as e:
            raise RuntimeError(f"TEDDY endpoint call failed for cells {start}-{end}: {e}") from e
        records = result if isinstance(result, list) else result.get("predictions", [])
        if len(records) != end - start:
            raise RuntimeError(f"Expected {end - start} embeddings, got {len(records)}")
        for cell_id, rec in zip(batch.index, records):
            rows.append({"cell_id": cell_id, "embedding": rec["embedding"]})
        if progress_callback:
            pct = pct_start + int((end / n) * (pct_end - pct_start))
            progress_callback(pct, f"Embedded {end}/{n} cells")
    out = pd.DataFrame(rows).set_index("cell_id")
    out.index = expr_df.index
    return out


def _fetch_neighbor_metadata(cell_ids: list, catalog: str, schema: str) -> pd.DataFrame:
    if not cell_ids:
        return pd.DataFrame()
    ids_str = ", ".join(f"'{cid}'" for cid in cell_ids)
    df = execute_select_query(
        f"SELECT cell_id, cell_type, disease, tissue, tissue_general, dataset_id "
        f"FROM {catalog}.{schema}.{TEDDY_CELLS_TABLE} WHERE cell_id IN ({ids_str})"
    )
    if not df.empty and "cell_id" in df.columns:
        df = df.set_index("cell_id").reindex(cell_ids).reset_index()
    return df


def search_nearest_cells(embedding, k: int = 100) -> pd.DataFrame:
    s = get_settings()
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    if isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], (list, tuple)):
        embedding = embedding[0]
    embedding = [float(x) for x in embedding]
    index_name = f"{s.catalog}.{s.schema}.{TEDDY_CELL_INDEX}"
    w = WorkspaceClient()
    result = w.vector_search_indexes.query_index(
        index_name=index_name,
        columns=["cell_id"],
        query_vector=embedding,
        num_results=k,
    )
    cell_ids = []
    if result.result and result.result.data_array:
        for row in result.result.data_array:
            cell_ids.append(row[0])
    return _fetch_neighbor_metadata(cell_ids, s.catalog, s.schema)


def _load_label_freqs(catalog: str, schema: str) -> tuple[dict[str, int], dict[str, int]]:
    key = f"{catalog}.{schema}"
    cached = _LABEL_FREQS_CACHE.get(key)
    if cached is not None:
        return cached
    table = f"{catalog}.{schema}.{TEDDY_CELLS_TABLE}"
    try:
        ct = execute_select_query(
            f"SELECT cell_type, COUNT(*) AS n FROM {table} GROUP BY cell_type"
        )
        ds = execute_select_query(
            f"SELECT disease, COUNT(*) AS n FROM {table} GROUP BY disease"
        )
        ct_freqs = dict(zip(ct["cell_type"].astype(str), ct["n"].astype(int)))
        ds_freqs = dict(zip(ds["disease"].astype(str), ds["n"].astype(int)))
        logger.info(
            "Loaded TEDDY label frequencies: %d cell_types, %d diseases from %s",
            len(ct_freqs),
            len(ds_freqs),
            table,
        )
    except Exception as e:
        logger.warning("Could not load TEDDY label frequencies from %s: %s", table, e)
        ct_freqs, ds_freqs = {}, {}
    _LABEL_FREQS_CACHE[key] = (ct_freqs, ds_freqs)
    return ct_freqs, ds_freqs


def _idf_weight(label: str, freqs: dict[str, int]) -> float:
    count = freqs.get(label, 1)
    return 1.0 / math.log1p(max(1, count))


def _summarize_topk(counter: Counter, total: float, k: int = 3) -> str:
    if not counter or total == 0:
        return ""
    top = counter.most_common(k)
    return "; ".join(f"{lbl} ({c / total:.0%})" for lbl, c in top)


def _summarize_topk_weighted(weights: dict, total_weight: float, k: int = 3) -> str:
    if not weights or total_weight <= 0:
        return ""
    top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return "; ".join(f"{lbl} ({w / total_weight:.0%})" for lbl, w in top)


@dataclass(frozen=True)
class TeddyClusterAnnotation:
    cluster: str
    n_cells: int
    predicted_cell_type: str
    cell_type_confidence_pct: float
    cell_type_top3: str
    predicted_disease: str
    disease_confidence_pct: float
    disease_top3: str


def _check_teddy_assets_available() -> None:
    """Pre-flight the TEDDY reference assets. Raises a single human-readable
    error if either the cells table or the VS index is missing — much better
    UX than per-cell KNN failures that surface as "Unknown / 0%" rows."""
    s = get_settings()
    table = f"{s.catalog}.{s.schema}.{TEDDY_CELLS_TABLE}"
    index_name = f"{s.catalog}.{s.schema}.{TEDDY_CELL_INDEX}"
    w = WorkspaceClient()
    missing: list[str] = []
    try:
        w.tables.get(full_name=table)
    except Exception:
        missing.append(f"table {table}")
    try:
        w.vector_search_indexes.get_index(index_name=index_name)
    except Exception:
        missing.append(f"vector-search index {index_name}")
    if missing:
        raise RuntimeError(
            "TEDDY reference assets are not deployed in this workspace: "
            + ", ".join(missing)
            + ". Run the TEDDY reference setup notebooks (06_extract_gene_mapping.py + the "
            "teddy_cells loader + VS index sync) on this workspace, then retry."
        )


def annotate_clusters(
    markers_df: pd.DataFrame,
    cluster_col: str = "cluster",
    cells_per_cluster: int = 20,
    k_neighbors: int = 50,
    bias_correct: bool = True,
    progress_callback=None,
) -> tuple[list[TeddyClusterAnnotation], dict[str, str], dict[str, str]]:
    """Returns (rows, cluster_to_cell_type, cluster_to_disease).

    `progress_callback(pct, msg)` is invoked between phases. Used by the
    SSE route for live updates."""
    def _p(pct: int, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    _p(2, "Checking TEDDY reference assets")
    _check_teddy_assets_available()
    if cluster_col not in markers_df.columns:
        raise ValueError(f"cluster column '{cluster_col}' not found")
    expr_cols = [c for c in markers_df.columns if c.startswith("expr_")]
    if not expr_cols:
        raise ValueError("markers_df has no expr_* columns")
    gene_names = [c.replace("expr_", "") for c in expr_cols]

    s = get_settings()
    if bias_correct:
        _p(5, "Loading reference label frequencies for IDF weighting")
        ct_freqs, ds_freqs = _load_label_freqs(s.catalog, s.schema)
        if not ct_freqs:
            bias_correct = False
    else:
        ct_freqs, ds_freqs = {}, {}

    sampled_idx: list = []
    clusters = sorted(
        [str(c) for c in markers_df[cluster_col].unique()],
        key=lambda x: int(x) if x.isdigit() else x,
    )
    for cl in clusters:
        cl_idx = markers_df.index[markers_df[cluster_col].astype(str) == cl]
        sampled_idx.extend(cl_idx[: min(cells_per_cluster, len(cl_idx))].tolist())

    expr_df = markers_df.loc[sampled_idx, expr_cols].copy()
    expr_df.columns = gene_names
    sampled_clusters = markers_df.loc[sampled_idx, cluster_col].astype(str).tolist()

    _p(10, f"Sampled {len(sampled_idx)} cells across {len(clusters)} clusters")
    emb_df = embed_cells(expr_df, gene_names, progress_callback=progress_callback)

    if bias_correct:
        cluster_to_celltype: dict[str, dict[str, float]] = {cl: {} for cl in clusters}
        cluster_to_disease: dict[str, dict[str, float]] = {cl: {} for cl in clusters}
    else:
        cluster_to_celltype = {cl: Counter() for cl in clusters}  # type: ignore[assignment]
        cluster_to_disease = {cl: Counter() for cl in clusters}  # type: ignore[assignment]

    n_emb = len(emb_df)
    for i, (_idx, row) in enumerate(emb_df.iterrows()):
        cl = sampled_clusters[i]
        try:
            meta = search_nearest_cells(row["embedding"], k=k_neighbors)
        except Exception as e:
            logger.warning("KNN failed for cell %d: %s", i, e)
            continue
        # 55% → 95% spread across per-cell KNN lookups.
        _p(55 + int(((i + 1) / max(n_emb, 1)) * 40),
           f"Vector Search neighbours {i + 1}/{n_emb}")
        if meta.empty:
            continue
        if "cell_type" in meta.columns:
            for lbl in meta["cell_type"].dropna().tolist():
                if bias_correct:
                    w = _idf_weight(lbl, ct_freqs)
                    d = cluster_to_celltype[cl]
                    d[lbl] = d.get(lbl, 0.0) + w
                else:
                    cluster_to_celltype[cl][lbl] += 1
        if "disease" in meta.columns:
            for lbl in meta["disease"].dropna().tolist():
                if bias_correct:
                    w = _idf_weight(lbl, ds_freqs)
                    d = cluster_to_disease[cl]
                    d[lbl] = d.get(lbl, 0.0) + w
                else:
                    cluster_to_disease[cl][lbl] += 1

    _p(96, "Aggregating votes per cluster")
    rows: list[TeddyClusterAnnotation] = []
    cluster_to_type: dict[str, str] = {}
    cluster_to_dis: dict[str, str] = {}
    for cl in clusters:
        ct = cluster_to_celltype[cl]
        ds = cluster_to_disease[cl]
        n_cells = sum(1 for sc in sampled_clusters if sc == cl)

        if not ct:
            top_ct, ct_conf, ct_top3 = "Unknown", 0.0, ""
        elif bias_correct:
            ct_total = sum(ct.values())
            top_ct, top_w = max(ct.items(), key=lambda kv: kv[1])
            ct_conf = top_w / ct_total if ct_total > 0 else 0.0
            ct_top3 = _summarize_topk_weighted(ct, ct_total)
        else:
            ct_total = sum(ct.values())
            top_ct, top_n = ct.most_common(1)[0]
            ct_conf = top_n / ct_total if ct_total > 0 else 0.0
            ct_top3 = _summarize_topk(ct, ct_total)

        if not ds:
            top_ds, ds_conf, ds_top3 = "Unknown", 0.0, ""
        elif bias_correct:
            ds_total = sum(ds.values())
            top_ds, top_w = max(ds.items(), key=lambda kv: kv[1])
            ds_conf = top_w / ds_total if ds_total > 0 else 0.0
            ds_top3 = _summarize_topk_weighted(ds, ds_total)
        else:
            ds_total = sum(ds.values())
            top_ds, top_n = ds.most_common(1)[0]
            ds_conf = top_n / ds_total if ds_total > 0 else 0.0
            ds_top3 = _summarize_topk(ds, ds_total)

        rows.append(
            TeddyClusterAnnotation(
                cluster=cl,
                n_cells=n_cells,
                predicted_cell_type=str(top_ct),
                cell_type_confidence_pct=round(ct_conf * 100, 1),
                cell_type_top3=ct_top3,
                predicted_disease=str(top_ds),
                disease_confidence_pct=round(ds_conf * 100, 1),
                disease_top3=ds_top3,
            )
        )
        cluster_to_type[cl] = str(top_ct)
        cluster_to_dis[cl] = str(top_ds)

    _p(100, f"Annotated {len(rows)} clusters")
    return rows, cluster_to_type, cluster_to_dis

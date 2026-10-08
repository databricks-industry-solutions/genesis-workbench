---
name: gwb-vector-search
description: Query a Genesis Workbench Vector Search index from a notebook for a life-sciences similarity or annotation workflow — protein sequence similarity search and single-cell nearest-neighbor cell-type annotation. Use when a user wants to find similar proteins or cells, annotate single cells against a reference by KNN, or query/build a GWB vector index. Covers the embed-then-query pattern and all GWB indices (UniRef + human-gene protein indices, SCimilarity cells, TEDDY cells). Triggers on "find similar proteins/cells", "nearest neighbor", "annotate cell types", "vector search", "similarity search", "semantic search over sequences", "KNN".
---

# Query Genesis Workbench Vector Search indices

GWB builds Delta-Sync Vector Search indices over model embeddings so you can do similarity search and
reference KNN annotation. You query them with the Databricks SDK — compute a **query embedding** with the
same model that built the index, then `query_index`.

## The indices
| Index (`{catalog}.{schema}.<name>`) | VS endpoint | Primary key | Dim | Source table | Use for |
|---|---|---|---|---|---|
| `sequence_embedding_index` | `gwb_sequence_search_vs_endpoint` | `seq_id` | 1280 | `sequence_embeddings` (UniRef90, ESM-2) | protein similarity (broad organisms) |
| `gene_sequence_embedding_index` | `gwb_sequence_search_vs_endpoint` | `seq_id` | 1280 | `gene_sequence_embeddings` (human SwissProt, ESM-2) | protein similarity (human targets, e.g. PARP1) |
| `scimilarity_cell_index` | `gwb_scimilarity_vs_endpoint` | `cell_id` | 128 | `scimilarity_cells` (~23M ref cells) | single-cell KNN (SCimilarity 128-d) |
| `teddy_cell_index` | `gwb_teddy_vs_endpoint` | `cell_id` | 1024 | `teddy_cells` (~2M ref cells) | single-cell cell-type annotation (TEDDY 1024-d) |

Query these alongside, not instead of: GWB searches the UniRef **and** human-gene protein indices together
so one query returns both broad-organism and human hits.

## Embed-then-query (the core pattern)
```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# 1) query vector — embed the query with the SAME model that built the index.
#    protein → the ESM-2 embeddings endpoint; cell → the TEDDY/SCimilarity encoder endpoint.
#    (exact per-model payloads: see gwb-serving-endpoints / gwb-single-cell)
qvec = embed_query(...)            # list[float] of the index's dim (1280 proteins / 1024 teddy / 128 scimilarity)

# 2) query the index
res = w.vector_search_indexes.query_index(
    index_name="main.genesis_workbench.sequence_embedding_index",
    columns=["seq_id"],            # metadata columns to return (plus the score)
    query_vector=qvec,
    num_results=10,
)
for row in (res.result.data_array or []):
    print(row)                     # [<seq_id>, ..., <similarity_score>]
```

## Worked example — protein similarity search (query both protein indices)
```python
query_seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"   # your protein
# embed via the ESM-2 endpoint → a 1280-d vector (see gwb-serving-endpoints for the exact payload)
qvec = esm2_embed(query_seq)

hits = []
for idx in ("sequence_embedding_index", "gene_sequence_embedding_index"):
    r = w.vector_search_indexes.query_index(
        index_name=f"main.genesis_workbench.{idx}", columns=["seq_id"],
        query_vector=qvec, num_results=10)
    hits += [(idx, *row) for row in (r.result.data_array or [])]
# merge + sort by score (last element of each row), dedupe by seq_id, take top-k
```

## Worked example — annotate single cells by reference KNN
```python
# encode each cell with the TEDDY (1024-d) or SCimilarity (128-d) encoder endpoint, then KNN-vote
cell_vec = teddy_encode(cell_expression)         # see gwb-single-cell for the encode payload
r = w.vector_search_indexes.query_index(
    index_name="main.genesis_workbench.teddy_cell_index",
    columns=["cell_id", "cell_type", "tissue", "disease"],
    query_vector=cell_vec, num_results=25)
# majority-vote cell_type over the returned neighbors → the cell's predicted annotation
```

## Build your own index over your embeddings
After producing an embedding Delta table (one row per id, an `ARRAY<FLOAT>` `embedding` column — e.g. via
**gwb-ray-batch-inference**), enable Change Data Feed and create a Delta-Sync index:
```python
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest, EmbeddingVectorColumn, VectorIndexType, PipelineType, EndpointType)
spark.sql("ALTER TABLE main.genesis_workbench.my_embeddings SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
# w.vector_search_endpoints.create_endpoint(name="my_vs_endpoint", endpoint_type=EndpointType.STANDARD)  # if needed
w.vector_search_indexes.create_index(
    name="main.genesis_workbench.my_index", endpoint_name="my_vs_endpoint",
    primary_key="id", index_type=VectorIndexType.DELTA_SYNC,
    delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
        source_table="main.genesis_workbench.my_embeddings",
        embedding_vector_columns=[EmbeddingVectorColumn(name="embedding", embedding_dimension=1280)],
        pipeline_type=PipelineType.TRIGGERED))
```

## Notes
- **Warm it first.** GWB reaps idle VS **endpoints** (and serving endpoints) — if `query_index` or the
  embed call fails, the endpoint may be asleep; (re)deploy/start it before running. The indices, tables,
  and the `gwb_*_vs_endpoint` names are preserved across GWB destroys.
- **Dim must match.** The query vector length must equal the index dim (1280 / 1024 / 128). A dim mismatch
  means you embedded with the wrong model.
- Scores are similarity (higher = closer) for the index's metric; use them to rank, not as probabilities.

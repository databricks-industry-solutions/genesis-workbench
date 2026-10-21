---
name: gwb-sequence-search
description: Protein sequence similarity search in a Databricks notebook with Genesis Workbench — embed a query protein with ESM-2 and search the UniRef + human-protein Vector Search indices to find functionally/structurally similar proteins (broad organisms and human targets like PARP1). Use for life-sciences homology/analog discovery by embedding. Triggers on "find proteins similar to this sequence", "what is this protein", "homologs / analogs by embedding", "search my sequence against the reference", "protein similarity search".
---

# Protein sequence similarity search (ESM-2 + Vector Search)

Embedding-space protein search: embed a query sequence with the **ESM-2 embeddings** endpoint (1280-d),
then query both GWB protein indices and merge. This captures functional/structural similarity (complements
alignment-based search). See **gwb-serving-endpoints** (esm2) and **gwb-vector-search** for the primitives.

```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

query_seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"   # your protein

esm2_ep = next(e.name for e in w.serving_endpoints.list() if "esm2_embeddings" in e.name)
qvec = w.serving_endpoints.query(name=esm2_ep, inputs=[query_seq]).predictions[0]   # 1280-d

hits = []
for idx in ("sequence_embedding_index",          # UniRef90 — broad organisms
            "gene_sequence_embedding_index"):     # human SwissProt — targets like PARP1
    r = w.vector_search_indexes.query_index(
        index_name=f"main.genesis_workbench.{idx}",
        columns=["seq_id"], query_vector=qvec, num_results=10)
    hits += [{"index": idx, "seq_id": row[0], "score": row[-1]} for row in (r.result.data_array or [])]

import pandas as pd
df = (pd.DataFrame(hits).sort_values("score", ascending=False)
        .drop_duplicates("seq_id").head(10).reset_index(drop=True))
display(df)   # seq_id = UniRef90 cluster id or human UniProt accession; map accessions via gene_sequences
```

## Notes
- **Search both indices** (as GWB does) so a human query also surfaces human targets, not only
  microorganism homologs. Dedupe by `seq_id`, rank by score.
- `seq_id` from the gene index is a UniProt accession — join to `main.genesis_workbench.gene_sequences`
  (`accession`→`gene`,`sequence`) to get the gene symbol and sequence.
- This is **embedding similarity**, not alignment — it finds functional/structural analogs even at low
  sequence identity. Warm the ESM-2 endpoint + the VS endpoint first (GWB reaps idle ones).
- Build a similar index over your own sequences with **gwb-ray-batch-inference** + **gwb-vector-search**.

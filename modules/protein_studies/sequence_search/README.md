# Sequence Search Module

Provides BLAST-like protein sequence similarity search on Databricks using a
hybrid funnel approach: ESM-2 embeddings for fast approximate retrieval,
followed by Smith-Waterman alignment for exact scoring.

## Architecture

```
         DEPLOYMENT PIPELINE (offline, run once + monthly refresh)
┌──────────────────────────────────────────────────────────────────┐
│  Download UniRef90 FASTA (~150M sequences)                       │
│       ↓                                                          │
│  Parse to Delta table (sequence_db)                              │
│       ↓                                                          │
│  Batch embed with ESM-2 650M via predict_batch_udf (GPU cluster) │
│       ↓                                                          │
│  Create Databricks Vector Search Delta Sync index                │
└──────────────────────────────────────────────────────────────────┘

         SEARCH (runtime, <5 seconds)
┌──────────────────────────────────────────────────────────────────┐
│  User pastes amino acid sequence in Streamlit UI                 │
│       ↓                                                          │
│  [Stage 1] Call ESM2 Embedding endpoint → 1280d vector  (~200ms) │
│       ↓                                                          │
│  [Stage 2] Vector Search ANN → top 500 candidates       (~500ms) │
│       ↓                                                          │
│  [Stage 3] Fetch sequences from Delta table               (~1s)  │
│       ↓                                                          │
│  [Stage 4] parasail Smith-Waterman alignment              (~2s)  │
│       ↓                                                          │
│  [Stage 5] Rank by SW score, return top 50 with alignments       │
└──────────────────────────────────────────────────────────────────┘
```

## Search Algorithm

### Stage 1 — Embed Query

The user's amino acid sequence is sent to the `gwb_esm2_embeddings_endpoint`,
which runs Facebook's ESM-2 650M model (`facebook/esm2_t33_650M_UR50D`). The
model produces per-residue representations that are mean-pooled (excluding
BOS/EOS tokens) into a single 1280-dimensional vector.

### Stage 2 — Approximate Nearest Neighbor Search

The query embedding is sent to a Databricks Vector Search Delta Sync index
built over ~150M pre-embedded UniRef90 sequences. The index uses approximate
nearest neighbor (ANN) algorithms to retrieve the top-K candidates (~500) by
vector distance.

### Stage 3 — Sequence Fetch

Full amino acid sequences for the candidate hits are fetched from the
`sequence_db` Delta table in Unity Catalog via SQL.

### Stage 4 — Smith-Waterman Alignment

Each candidate is aligned against the query using parasail's SIMD-accelerated
Smith-Waterman implementation with BLOSUM62 scoring matrix, gap open penalty
of 11, and gap extension penalty of 1. This produces:

- Identity percentage
- Smith-Waterman alignment score
- Alignment length
- Visual pairwise alignment (query, match line, target)

### Stage 5 — Rank and Return

Results are sorted by Smith-Waterman score descending. The top-K final results
(default 50) are returned to the Streamlit UI with full alignment details.

## How This Differs from BLAST

This is **not a BLAST replacement**. It is a complementary tool that trades
some of BLAST's properties for massive speed and scalability. Users should
understand the differences:

### Determinism

| | BLAST | This approach |
|---|---|---|
| **Deterministic?** | Yes — same query + same database = same results every time | No — Stage 2 uses approximate nearest neighbor (ANN) algorithms that may return slightly different candidate sets across runs |
| **Why** | Seeding, extension, and statistical scoring are all exact operations | ANN probes a subset of index partitions; index rebuilds can change internal cluster assignments |

Note: Stage 4 (Smith-Waterman) is fully deterministic. The non-determinism
comes only from Stage 2's ANN pre-filtering.

### What Each Algorithm Measures

| | BLAST | This approach |
|---|---|---|
| **Core metric** | Local sequence alignment — finds shared subsequences using k-mer seeding + gapped extension | Semantic similarity in ESM-2's learned embedding space, refined by local alignment |
| **Scoring** | Substitution matrix + gap penalties + Karlin-Altschul statistics producing E-values | Embedding distance (L2/cosine) for filtering, then SW score for ranking |
| **Statistical framework** | E-values with rigorous meaning ("expected number of hits by chance in a database of this size") | No E-values — SW scores on a pre-filtered subset have no comparable statistical interpretation |

### Where This Approach Excels

- **Speed**: Sub-5-second searches across 150M sequences vs. minutes for BLAST
- **Functional similarity**: ESM-2 embeddings capture structural and functional
  relationships that pure sequence alignment misses. Two proteins with different
  sequences but similar folds will score well here but may be missed by BLAST.
- **Scalability**: Vector Search scales to billions of sequences with consistent
  latency. BLAST scales linearly with database size.

### Where BLAST is Better

- **Short conserved motifs**: BLAST excels at finding proteins sharing a short
  conserved region (e.g., a 20-residue binding motif). Mean-pooled embeddings
  dilute short motifs across the full-length representation.
- **Multi-domain proteins**: Mean pooling blends all domains into one vector.
  BLAST finds hits to individual domains; this approach may miss domain-level
  matches when other domains are very different.
- **Remote homologs (20-30% identity)**: BLAST's statistical model (E-values,
  gapped alignment) is specifically tuned for the twilight zone. Embedding
  distance does not have the same sensitivity here.
- **Reproducibility**: BLAST results are fully deterministic and carry
  well-understood statistical significance (E-values). This matters for
  publications and regulatory submissions.

### Result Overlap

A user who runs the same query on NCBI BLAST and on this system will get
**overlapping but different result sets**:

- **Top hits** (high identity homologs, >40%) will likely agree
- **Mid-range hits** will diverge — this system may surface functionally related
  proteins that BLAST misses, while BLAST may find distant sequence matches that
  the embedding filter discards
- **Edge cases** (twilight zone, short motifs) will differ significantly

## Technology Stack

| Component | Technology |
|---|---|
| Embedding model | `facebook/esm2_t33_650M_UR50D` (ESM-2 650M, 1280d) |
| Embedding serving | Databricks Model Serving (MLflow PyFunc, GPU) |
| Vector index | Databricks Vector Search (Delta Sync, TRIGGERED) |
| Sequence database | UniRef90 (~150M sequences) in Delta table |
| Alignment | parasail (SIMD-accelerated Smith-Waterman) |
| Batch embedding | Spark `predict_batch_udf` on GPU cluster |
| UI | Streamlit (Protein Studies → Sequence Search tab) |

## Deployment

### Prerequisites

- ESM2 Embeddings model must be deployed first
  (`modules/protein_studies/esm2_embeddings/`)
- Core module must be deployed (provides libraries volume, workbench tables)

### Deploy

```bash
cd modules/protein_studies/sequence_search/sequence_search_v1
./deploy.sh --var="<params>"
```

This runs a 4-step workflow:

1. **01_download_sequences.py** — Downloads UniRef90 FASTA to UC Volume
2. **02_create_delta_tables.py** — Parses FASTA into `sequence_db` Delta table
3. **03_batch_embed_sequences.py** — Generates embeddings with ESM-2 on GPU
4. **04_create_vector_index.py** — Creates Vector Search endpoint and index

### Incremental Updates

UniRef90 is updated approximately every 8 weeks. To refresh:

1. Re-run notebook 01 to download the latest FASTA
2. Re-run notebook 02 with `MERGE INTO` logic for new/changed sequences
3. Re-run notebook 03 only on rows where `embedding IS NULL`
4. Trigger a Vector Search index sync

## Potential Improvements

- **Increase ANN candidates** (`top_k_vector`) from 500 to 2000-5000 to widen
  the funnel and improve recall (parasail is fast enough to handle it)
- **Per-residue embeddings** instead of mean-pooling — would capture local
  motif matches better, but requires a more sophisticated similarity metric
  and significantly more storage
- **Nucleotide support** (Phase 2) — requires a separate embedding model
  (e.g., Nucleotide Transformer) and a second Vector Search index
- **NVIDIA NIM** — if NVIDIA releases an ESM-2 NIM with optimized CUDA kernels,
  it could accelerate both batch embedding and query-time inference
- **E-value approximation** — compute Karlin-Altschul statistics from the SW
  scores to provide users with familiar BLAST-like significance measures

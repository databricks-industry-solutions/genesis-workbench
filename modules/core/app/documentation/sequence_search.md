# Sequence Similarity Search

## Introduction

Sequence Similarity Search provides fast, BLAST-like protein sequence searching across ~150 million UniRef90 sequences. It uses a hybrid approach combining ESM-2 neural embeddings for approximate filtering with Smith-Waterman alignment for exact scoring, achieving sub-5-second latency compared to minutes for traditional BLAST.

## What It Achieves

- Searches a query protein sequence against ~150M reference sequences
- Returns top-K similar sequences ranked by alignment score
- Provides pairwise alignments with identity percentages
- Achieves comparable sensitivity to BLAST for most use cases at dramatically faster speed

## How to Use

1. Navigate to **Protein Studies > Sequence Search** tab
2. Paste a protein sequence or upload a FASTA file
3. Set the maximum number of results (25-500)
4. Click **Search**
5. Browse results: sequence IDs, identity %, alignment scores
6. Click any result to view the detailed pairwise alignment

### Inputs

| Field | Description | Default |
|-------|-------------|---------|
| Query sequence | Amino acid sequence to search | Required |
| Max results | Number of top hits to return | 50 |

### Outputs

- **Results table**: Sequence ID, identity %, alignment length, vector distance, alignment score
- **Pairwise alignment**: Visual alignment showing matches, mismatches, and gaps

## How It's Implemented

### 5-Stage Funnel Architecture

```
Stage 1: Query Embedding (~200ms)
  └→ ESM-2 endpoint → 1280-dimensional vector

Stage 2: Approximate Nearest Neighbor (~500ms)
  └→ Databricks Vector Search (FAISS) → top 500 candidates

Stage 3: Sequence Fetch (~1s)
  └→ Delta table query → full sequences for candidates

Stage 4: Smith-Waterman Alignment (~2s)
  └→ parasail SIMD alignment → exact scores + alignments

Stage 5: Ranking & Return
  └→ Sort by SW score → return top-K results
```

### Comparison with BLAST

| Property | BLAST | ESM-2 + Smith-Waterman |
|----------|-------|----------------------|
| Speed | Minutes | < 5 seconds |
| Database size | Millions | 150M+ sequences |
| Remote homologs (< 30% identity) | Better | Less sensitive |
| Functional similarity (different sequences) | Misses | Better at detecting |
| Statistical significance (E-values) | Yes | No |

### Setup Pipeline (One-Time)

1. **Download sequences** (`01_download_sequences.py`): UniRef90 FASTA (~40GB)
2. **Create Delta tables** (`02_create_delta_tables.py`): Parse FASTA into `sequence_db` table
3. **Batch embed** (`03_batch_embed_sequences.py`): Compute ESM-2 embeddings for all sequences using GPU Spark UDF
4. **Create vector index** (`04_create_vector_index.py`): Build Databricks Vector Search Delta Sync index

### Key Files

- `modules/core/app/views/protein_studies.py` — UI (Sequence Search tab)
- `modules/protein_studies/sequence_search/sequence_search_v1/notebooks/` — Setup pipeline
- `modules/protein_studies/esm2_embeddings/esm2_embeddings_v1/` — ESM-2 embedding model

### Dependencies

- ESM-2 Embeddings module must be deployed first
- Databricks Vector Search index must be created
- parasail library for Smith-Waterman alignment

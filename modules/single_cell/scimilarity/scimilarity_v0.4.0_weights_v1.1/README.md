# SCimilarity Module

Cell search and annotation using [SCimilarity](https://genentech.github.io/scimilarity/) (Genentech) — a pretrained model for querying ~22.7M single-cell reference profiles.

## What it does

1. **Downloads** SCimilarity model weights + Adams et al. sample dataset
2. **Registers** MLflow custom pyfunc models in Unity Catalog (gene order + cell embedding)
3. **Deploys** model serving endpoints for those pyfuncs
4. **Extracts** the 23M-cell reference corpus to a Delta table and builds a Databricks Vector Search index for nearest-neighbor search (mirrors the Protein Sequence Search pattern)
5. **Extracts** disease/celltype sample data for endpoint testing

## Serving Endpoints

| Endpoint | Model | What it does | Workload | Size |
|---|---|---|---|---|
| `gwb_mmt_scimilarity_gene_order_endpoint` | `scimilarity_gene_order` | Returns gene ordering for dataset alignment | CPU | Small |
| `gwb_mmt_scimilarity_get_embedding_endpoint` | `scimilarity_get_embedding` | Neural network inference — generates cell embeddings from gene expression | GPU Medium (4xA10G) | Small |

Nearest-neighbor cell search now runs against a **Databricks Vector Search index** (`scimilarity_cell_index`) that is Delta-synced from `scimilarity_cells`. The app-side `search_nearest_cells` in `modules/core/app/utils/scimilarity_tools.py` queries the index directly — no per-request model serving endpoint in the loop. This matches the pattern used by the Protein Sequence Search feature.

**Why GPU for the embedder endpoint?**
`scimilarity==0.4.0` transitively pulls in `torch` + `pytorch-lightning`. GPU serving environments have torch pre-cached in the base image, so container builds are fast. CPU serving works functionally but triggers a full torch install from scratch (~slow build).

Workload type and size are configurable per endpoint via job parameters (see below).

## Registration Job Flow

```
01_wget_scimilarity (download model + sample data)
    ├── 02_register_GeneOrder ─────────────────────┐
    ├── 03_register_GetEmbedding ──────────────────┤
    │                                               └── 05_importNserve_model_gwb
    ├── 06a_extractNsave_DiseaseCellTypeSamples
    └── 06b_extract_reference_to_delta
            └── 06c_create_cell_vector_index
```

All tasks run on `14.3.x-gpu-ml-scala2.12` A10 GPU clusters (ON_DEMAND).

## Job Parameters

| Parameter | Default | Description |
|---|---|---|
| `catalog` | `${var.core_catalog_name}` | Unity Catalog name |
| `schema` | `${var.core_schema_name}` | Schema name |
| `model_name` | `SCimilarity` | Model name prefix |
| `cache_dir` | `scimilarity` | Volume name for model/data storage |
| `experiment_name` | `gwb_modules_scimilarity` | MLflow experiment for registration runs |
| `sql_warehouse_id` | `${var.sql_warehouse_id}` | SQL warehouse for metadata queries |
| `user_email` | `${var.current_user}` | Deploying user |
| `gene_order_workload_type` | `CPU` | Endpoint workload type for gene_order (lightweight, no torch dependency) |
| `gene_order_workload_size` | `Small` | Endpoint workload size for gene_order |
| `get_embedding_workload_type` | `MULTIGPU_MEDIUM` | Endpoint workload type for get_embedding |
| `get_embedding_workload_size` | `Small` | Endpoint workload size for get_embedding |

## MLflow Experiments

Registration runs log to: `/Shared/dbx_genesis_workbench_models/gwb_modules_scimilarity`

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_wget_scimilarity.py` | Download model weights + Adams et al. sample dataset to UC Volume |
| `02_register_GeneOrder.py` | Register gene order pyfunc model |
| `03_register_GetEmbedding.py` | Register cell embedding pyfunc model |
| `04_register_SearchNearest.py` | **DEPRECATED** — no longer in the deploy DAG. Kept for reference; superseded by 06b + 06c. |
| `05_importNserve_model_gwb.py` | Import models into GWB catalog + deploy serving endpoints |
| `06a_extractNsave_DiseaseCellTypeSamples.py` | Extract IPF myofibroblast samples for testing |
| `06b_extract_reference_to_delta.py` | Write the 23M-cell reference corpus to Delta table `scimilarity_cells` |
| `06c_create_cell_vector_index.py` | Create VS endpoint + Delta Sync index `scimilarity_cell_index` |
| `utils.py` | Shared setup: pip installs, config variables, data preprocessing |

## Data

Stored in `/Volumes/<catalog>/<schema>/scimilarity/`:

```
scimilarity/
├── model/
│   └── model_v1.1/          # SCimilarity pretrained model + gene_order.tsv
├── data/
│   └── adams_etal_2020/      # Adams et al. lung disease dataset
│       ├── adams.h5ad                                    # Full dataset (gzipped)
│       ├── adams0_alignedNlognormed_Xscim_umap.h5ad     # Pre-embedded for viz
│       ├── GSE136831_subsample.h5ad                      # Original subsample
│       └── IPF/myofibroblast-cell/                       # Extracted samples per disease/celltype
```

## Dependencies

Key pinned versions (in `utils.py` and `06b`):
- `scimilarity==0.4.0`
- `scanpy==1.11.2`
- `numpy==1.26.4`
- `pandas==1.5.3`
- `mlflow==2.22.0`
- `numcodecs[crc32c]==0.13.1`

Requires DBR `14.3 LTS` (Python 3.10) for SCimilarity compatibility.

## References

- [SCimilarity documentation](https://genentech.github.io/scimilarity/)
- [SCimilarity GitHub](https://github.com/Genentech/scimilarity)
- [Cell search tutorial](https://genentech.github.io/scimilarity/notebooks/cell_search_tutorial_1.html)
- [Adams et al. 2020](https://www.science.org/doi/10.1126/sciadv.aba1983) — single-cell atlas of IPF, COPD, and healthy lung

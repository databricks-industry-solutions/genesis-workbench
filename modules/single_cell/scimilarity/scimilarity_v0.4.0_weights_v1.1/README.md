# SCimilarity Module

Cell search and annotation using [SCimilarity](https://genentech.github.io/scimilarity/) (Genentech) — a pretrained model for querying ~22.7M single-cell reference profiles.

## What it does

1. **Downloads** SCimilarity model weights + Adams et al. sample dataset
2. **Registers** three MLflow custom pyfunc models in Unity Catalog
3. **Deploys** three model serving endpoints
4. **Extracts** disease/celltype sample data for endpoint testing

## Serving Endpoints

| Endpoint | Model | What it does | Workload | Size |
|---|---|---|---|---|
| `gwb_mmt_scimilarity_gene_order_endpoint` | `scimilarity_gene_order` | Returns gene ordering for dataset alignment | CPU | Small |
| `gwb_mmt_scimilarity_get_embedding_endpoint` | `scimilarity_get_embedding` | Neural network inference — generates cell embeddings from gene expression | GPU Medium (4xA10G) | Small |
| `gwb_mmt_scimilarity_search_nearest_endpoint` | `scimilarity_search_nearest` | FAISS nearest-neighbor search across ~23M cell reference | GPU Medium (4xA10G) | Small |

**Why GPU for all endpoints?**
All models depend on `scimilarity==0.4.0` which transitively pulls in `torch` + `pytorch-lightning`. GPU serving environments have torch pre-cached in the base image, so container builds are fast. CPU serving works functionally but triggers a full torch install from scratch (~slow build). Keep all endpoints on GPU for fast, consistent deployments.

**Why Small concurrency for search_nearest?**
`search_nearest` loads ~23M cell reference into RAM (~12GB per worker). Small concurrency (0-4 workers) keeps total memory within the node's limits. Medium concurrency (0-16) causes OOM.

Workload type and size are configurable per endpoint via job parameters (see below).

## Registration Job Flow

```
01_wget_scimilarity (download model + sample data)
    ├── 02_register_GeneOrder
    ├── 03_register_GetEmbedding
    ├── 04_register_SearchNearest
    │       └── 05_importNserve_model_gwb (deploy endpoints after all register tasks)
    └── 06a_extractNsave_DiseaseCellTypeSamples (extract sample data)
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
| `search_nearest_workload_type` | `MULTIGPU_MEDIUM` | Endpoint workload type for search_nearest |
| `search_nearest_workload_size` | `Small` | Endpoint workload size for search_nearest |

## MLflow Experiments

Registration runs log to: `/Shared/dbx_genesis_workbench_models/gwb_modules_scimilarity`

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_wget_scimilarity.py` | Download model weights + Adams et al. sample dataset to UC Volume |
| `02_register_GeneOrder.py` | Register gene order pyfunc model |
| `03_register_GetEmbedding.py` | Register cell embedding pyfunc model |
| `04_register_SearchNearest.py` | Register nearest-neighbor search pyfunc model |
| `05_importNserve_model_gwb.py` | Import models into GWB catalog + deploy serving endpoints |
| `06a_extractNsave_DiseaseCellTypeSamples.py` | Extract IPF myofibroblast samples for testing |
| `06b_checkNuse_SCimilarityEndpoints.ipynb` | Guided walkthrough of using served endpoints |
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

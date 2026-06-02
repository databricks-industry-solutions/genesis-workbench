# Genesis Workbench
### **A Blueprint for a Life Sciences application on Databricks**

## $${\color{orange}Opportunity}$$
<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/opportunity.png" alt="Generative AI reshaping life sciences" width="800"/>

Generative AI is reshaping the life sciences. Foundation models trained on genomics, protein structures, molecular interactions, and cellular behaviors are unlocking capabilities that weren't possible a few years ago:

- **Foundation models across every domain** — From DNA to proteins to single cells to small molecules, there's now a pretrained model for almost every biological modality. Teams can start from these and fine-tune on their own data instead of training from scratch.
- **Drug discovery acceleration** — Generative AI compresses months of wet-lab iteration into hours of in-silico screening, candidate ranking, and hypothesis generation. Lead time on early-stage discovery drops by an order of magnitude.
- **Higher-accuracy structure prediction** — Tools like AlphaFold, ESMFold, and Boltz predict 3D structure from sequence with previously unattainable accuracy, opening up targets that were structurally inaccessible.
- **Personalized medicine** — Patient-specific modeling of therapeutic responses is moving from research curiosity to clinical reality, enabling precision treatment decisions at the individual level.
- **Knowledge synthesis at scale** — LLMs specialized in biomedical literature surface insights buried in millions of papers, accelerating hypothesis generation and reducing duplicated work.

## $${\color{orange}Challenges}$$
<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/challenges.png" alt="Scientists overwhelmed by AI/infrastructure challenges" width="800"/>

Despite the breakthroughs, the experts who can apply these models — biologists, geneticists, biochemists — spend most of their time on tasks far outside their training:

- **GPU and CUDA infrastructure** — Days spent configuring drivers, toolkits, and CUDA-compatible PyTorch builds before a single inference can run.
- **Workflow orchestration** — Stitching together data prep, training, evaluation, and serving pipelines requires software-engineering skills the biology curriculum doesn't cover.
- **Data engineering and governance** — Collecting, cleaning, and integrating heterogeneous biological datasets under privacy and reproducibility constraints is a full-time job in itself.
- **Model serving and lifecycle** — Even after training, productionizing a model — registry, endpoints, monitoring, retraining — is yet another discipline the scientist has to pick up.
- **Time stolen from the science** — Every hour spent on infrastructure is an hour not spent on biology. The slower the loop, the slower the innovation.

## $${\color{orange}Genesis}$$ $${\color{orange}Workbench}$$

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/genesis_workbench.png" alt="Genesis Workbench: Blueprint for life sciences application on Databricks" width="800"/>

Genesis Workbench is an open-source, Databricks-native blueprint that packages biological foundation models behind an intuitive UI — so scientists can run them without managing GPU clusters, CUDA, model registries, or serving endpoints.

- **Pre-packaged biological models** ready to deploy: ESMFold, AlphaFold2, ProteinMPNN, RFDiffusion, scGPT, SCimilarity, Scanpy, rapids-singlecell (part of scverse), ChemProp, DiffDock, Boltz, NVIDIA Parabricks, NVIDIA BioNeMo and more.
- **Tailored workflows** for protein design, drug discovery, single-cell analysis, and variant analysis — each surfaced as a UI tab with sane defaults.
- **Built on Databricks primitives**: Asset Bundles, Workflows, Model Serving, MLflow, Unity Catalog, and Databricks Apps — so everything you run is governed, reproducible, and traceable.
- **Modular and extensible**: each capability is an independent module that can be deployed and destroyed independently.

## $${\color{orange}Modules}$$

Each module is independently deployable with `./deploy.sh <module> <cloud>`. Click through to the workflow deep-dives for inputs, outputs, models, and example runs.

### Single Cell
Single-cell RNA-seq at scale. Run end-to-end pipelines on millions of cells with Scanpy or GPU-accelerated rapids-singlecell (part of scverse). Annotate cell types per cluster against SCimilarity's 23M-cell reference database, search published studies for similar cells, and predict the effect of gene knockouts or overexpression with scGPT. The interactive results viewer offers UMAP exploration, differential expression, pathway enrichment, and trajectory analysis on the same run.

**Models bundled:** scGPT, SCimilarity, Scanpy, rapids-singlecell (part of scverse), Merck TEDDY-G 400M (joint cell-type + disease annotation)

📖 [Single Cell Analysis](modules/core/app/backend/documentation/single_cell_analysis.md) · [Cell Type Annotation](modules/core/app/backend/documentation/cell_type_annotation.md) · [Cell Similarity Search](modules/core/app/backend/documentation/cell_similarity.md) · [Gene Perturbation Prediction](modules/core/app/backend/documentation/perturbation_prediction.md)

### Large Molecule
Protein structure prediction, design, and engineering. Fold proteins in seconds with ESMFold or at high accuracy with AlphaFold2; design novel sequences for a target backbone with RFDiffusion + ProteinMPNN; run BLAST-like searches across 150M+ sequences using ESM-2 embeddings. The **Guided Enzyme Optimization** workflow chains Proteina-Complexa, ProteinMPNN, ESMFold, Boltz, NetSolP, PLTNUM, DeepSTABp, and MHCflurry into a reward-weighted scaffold-and-score loop that surfaces variants ranked on motif fidelity, fold confidence, optional substrate complex, and four developability axes (solubility, anchor-relative half-life, melting temperature, immunogenic burden).

**Models bundled:** ESMFold, ESM-2 Embeddings, AlphaFold2, ProteinMPNN, RFDiffusion, Boltz

📖 [Protein Structure Prediction](modules/core/app/backend/documentation/protein_structure_prediction.md) · [Protein Design](modules/core/app/backend/documentation/protein_design.md) · [Sequence Similarity Search](modules/core/app/backend/documentation/sequence_search.md) · [Guided Enzyme Optimization](modules/core/app/backend/documentation/enzyme_optimization.md)

### Small Molecule
Drug-discovery essentials. Profile candidates for drug-like properties and toxicity with ChemProp, predict protein-ligand binding poses with DiffDock, design protein binders to a target protein or small molecule with Proteina-Complexa, and transplant functional motifs into new scaffolds. Each generated candidate can be scored on developability through NetSolP (solubility), PLTNUM-ESM2 (relative half-life), DeepSTABp (melting temperature), and MHCflurry (immunogenic burden).

**Models bundled:** ChemProp, DiffDock, Proteina-Complexa, NetSolP-1.0, PLTNUM-ESM2, DeepSTABp, MHCflurry 2.x

📖 [Molecular Docking](modules/core/app/backend/documentation/molecular_docking.md) · [Protein Binder Design](modules/core/app/backend/documentation/protein_binder_design.md) · [Ligand Binder Design](modules/core/app/backend/documentation/ligand_binder_design.md) · [Motif Scaffolding](modules/core/app/backend/documentation/motif_scaffolding.md) · [ADMET & Safety](modules/core/app/backend/documentation/admet_safety.md)

### Genomics
Variant analysis at population scale. Call germline variants from FASTQ files with GPU-accelerated NVIDIA Parabricks, ingest VCFs into Delta tables for fast SQL/Spark queries, run genome-wide association studies (GWAS) using Glow, and annotate variants with ClinVar clinical-significance data. Inline interactive charts in the UI break down findings by gene, ACMG category, clinical significance, and zygosity.

**Components:** NVIDIA Parabricks variant calling, Glow GWAS, Spark VCF→Delta ingestion, ClinVar annotation

📖 [Variant Calling](modules/core/app/backend/documentation/variant_calling.md) · [VCF Ingestion](modules/core/app/backend/documentation/vcf_ingestion.md) · [GWAS Analysis](modules/core/app/backend/documentation/gwas_analysis.md) · [Variant Annotation](modules/core/app/backend/documentation/variant_annotation.md)

### NVIDIA BioNeMo
NVIDIA's generative AI framework for digital biology. The optional BioNeMo module ships container definitions and workflows that expose pre-trained BioNeMo models, starting with ESM-2 fine-tuning and inference. Containers are optimized for NVIDIA hardware and integrated into Genesis Workbench's job system, MLflow registry, and model-serving infrastructure.

📖 [ESM2 Fine-tuning & Inference](modules/core/app/backend/documentation/bionemo_esm2.md)

📚 **Full workflow catalog:** [documentation index](modules/core/app/backend/documentation/index.md)

## $${\color{orange}Architecture}$$ $${\color{orange}Diagram}$$

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/architecture.png" alt="Architecture" width="700"/>

## $${\color{orange}Important}$$ $${\color{orange}Disclaimer}$$

**NVIDIA, the NVIDIA logo, and NVIDIA BioNeMo are trademarks or registered trademarks of NVIDIA Corporation in the United States and other countries. All other product names, trademarks, and registered trademarks are the property of their respective owners.**

**References to third-party products or services, including NVIDIA BioNeMo, are for informational purposes only and do not constitute an endorsement or affiliation. This material is not sponsored or endorsed by NVIDIA Corporation. The information provided here is for general informational purposes and should not be interpreted as specific advice or a warranty of suitability for any particular use.**

**Use of NVIDIA BioNeMo and related technologies should comply with all relevant licensing terms, trademarks, and applicable regulations.**

## $${\color{orange}Installation}$$

Full step-by-step setup is in the [Installation Guide](Installation.md). Quick path (assumes Databricks CLI authenticated to the target workspace as DEFAULT profile):

```bash
# 1. Deploy core first (UI, infrastructure, settings tables)
./deploy.sh core <aws|azure|gcp>

# 2. Deploy each module — one at a time, wait for jobs to finish before the next
./deploy.sh large_molecule  <aws|azure|gcp>
./deploy.sh single_cell     <aws|azure|gcp>
./deploy.sh small_molecule  <aws|azure|gcp>
./deploy.sh genomics        <aws|azure|gcp>
./deploy.sh bionemo         <aws|azure|gcp>   # optional, requires BioNeMo container build
```

For UI-only redeploys after the initial install (preserves all settings tables — never use `deploy.sh core` on a populated install):

```bash
cd modules/core
./update.sh <cloud> --ui-only   # rebuilds frontend, redeploys app, skips secret refresh / grants / UC volume copy
```

## $${\color{orange}Changelog}$$
See [CHANGELOG.md](CHANGELOG.md) for deployment fixes, known issues, and configuration notes.

## $${\color{orange}Troubleshooting}$$
The repo ships [Claude Code skills](claude_skills/) covering installation, deployment, troubleshooting, workflows, and development. These are designed to be loaded into Claude Code; they also serve as the canonical reference for common deployment failures and fixes.

## $${\color{orange}License}$$
Please see LICENSE for the details of the license.

Some packages, tools, and code used inside individual tutorials are under their own licenses as described therein. Please ensure you read the details and licensing of individual tools. Other third party packages are used in tutorials within this accelerator and have their own licensing, as laid out in the table below.

We are adding a script to build your own Databricks compatible container for NVIDIA BioNeMo. If you want to use NVIDIA BioNeMo in Genesis Workbench, please follow the instructions to build the container and push the image to your container repository.

NVIDIA GPUs and cudatoolkit may be used in multiple places so you should consider the NVIDIA EULA(link) when using code in this package.

Module | Package | License | Source
-------- | ------- | ------- | --------
core | fastapi==0.115.5 | MIT | https://github.com/tiangolo/fastapi
core | uvicorn[standard]==0.32.1 | BSD-3 | https://github.com/encode/uvicorn
core | react==19.x | MIT | https://github.com/facebook/react
core | vite | MIT | https://github.com/vitejs/vite
core | tailwindcss | MIT | https://github.com/tailwindlabs/tailwindcss
core | tanstack/react-query | MIT | https://github.com/tanstack/query
core | tanstack/react-table | MIT | https://github.com/tanstack/table
core | zustand | MIT | https://github.com/pmndrs/zustand
core | molstar | MIT | https://github.com/molstar/molstar
core | plotly.js | MIT | https://github.com/plotly/plotly.js
core | databricks-sdk==0.50.0 | Apache2.0 | https://pypi.org/project/databricks-sdk/
core | databricks-sql-connector | Apache2.0 | https://github.com/databricks/databricks-sql-python
core | py3Dmol==2.4.0 | MIT | TOBEREMOVED (https://pypi.org/project/py3Dmol/)
core | biopython |	[BioPython License Agreement](https://github.com/biopython/biopython/blob/master/LICENSE.rst) | https://github.com/biopython/biopython
core | Mlflow	| Apache2.0 | https://github.com/mlflow/mlflow
core | plotly | MIT | https://github.com/plotly/plotly.py
core | openai | MIT | https://github.com/openai/openai-python
core | parasail | BSD | https://github.com/jeffdaily/parasail-python
core | requests | Apache2.0 | https://github.com/psf/requests
core | pandas | BSD-3 | https://github.com/pandas-dev/pandas
core | numpy | BSD-3 | https://github.com/numpy/numpy
core | rdkit | BSD-3 | https://github.com/rdkit/rdkit
scGPT | numpy==1.26.4 | BSD-3-Clause | https://github.com/numpy/numpy
scGPT | gdown==5.2.0 | MIT | https://github.com/wkentaro/gdown
scGPT | wget==3.2 | MIT / Public Domain | https://pypi.org/project/wget/
scGPT | ipython==8.15.0 | BSD | https://github.com/ipython/ipython
scGPT | cloudpickle==2.2.1 | BSD-3-Clause | https://github.com/cloudpipe/cloudpickle
scGPT | torch==2.0.1+cu118 | BSD | https://github.com/pytorch/pytorch
scGPT | torchvision==0.15.2+cu118 | BSD | https://github.com/pytorch/vision
scGPT | flash-attn==2.5.8 | Apache-2.0 | https://github.com/Dao-AILab/flash-attention
scGPT | scgpt==0.2.4 | MIT | https://github.com/bowang-lab/scGPT
scGPT | wandb==0.19.11 | MIT | https://github.com/wandb/wandb
SCimilarity | SCimilarity v0.4.0 |	Apache2.0 | https://github.com/Genentech/scimilarity
SCimilarity | Mlflow	| Apache2.0 | https://github.com/mlflow/mlflow
SCimilarity | 'torch' / PyTorch | BSD-3-Clause | https://pypi.org/project/torch/2.7.1/
SCimilarity | scanpy |	BSD 3-Clause | https://github.com/scverse/scanpy
SCimilarity | numcodecs |	MIT | https://github.com/zarr-developers/numcodecs
SCimilarity | tbb	| Apache2.0 | https://github.com/uxlfoundation/oneTBB
SCimilarity | typing_extensions	| PSF | https://github.com/python/typing_extensions
SCimilarity | numpy  |	BSD | https://github.com/numpy/numpy
SCimilarity | pandas | BSD 3-Clause | https://github.com/pandas-dev/pandas
SCimilarity | uv | Apache2.0 | https://github.com/astral-sh/uv
SCimilarity | cloudpickle==2.0.0 | BSD-3 | https://github.com/cloudpipe/cloudpickle
RFDiffusion | RFDiffusion |	BSD-3 | https://github.com/RosettaCommons/RFdiffusion
RFDiffusion | Hydra	| MIT | https://github.com/facebookresearch/hydra
RFDiffusion | OmegaConf |	BSD-3 | https://github.com/omry/omegaconf
RFDiffusion | Biopython |	[BioPython License Agreement](https://github.com/biopython/biopython/blob/master/LICENSE.rst) | https://github.com/biopython/biopython
RFDiffusion | DGL	| Apache2.0 | https://github.com/dmlc/dgl
RFDiffusion | pyrsistent |	MIT | https://github.com/tobgu/pyrsistent
RFDiffusion | e3nn	| MIT | https://github.com/e3nn/e3nn
RFDiffusion | Wandb |	MIT | https://github.com/wandb/wandb
RFDiffusion | Pynvml	| BSD-3 | https://github.com/gpuopenanalytics/pynvml
RFDiffusion | Decorator	| BSD-2 | https://github.com/micheles/decorator
RFDiffusion | Torch |	BSD-3 | https://github.com/pytorch/pytorch
RFDiffusion | Torchvision |	BSD-3 | https://github.com/pytorch/vision
RFDiffusion | torchaudio==0.11.0 |	BSD-2 | https://github.com/pytorch/audio
RFDiffusion | cloudpickle==2.2.1	| BSD-3 | https://github.com/cloudpipe/cloudpickle
RFDiffusion | dllogger 	| Apache2.0 | https://github.com/NVIDIA/dllogger
RFDiffusion | SE3Transformer |	MIT | https://github.com/RosettaCommons/RFdiffusion/tree/main/env/SE3Transformer
RFDiffusion | mlflow==2.15.1 | Apache2.0 | https://github.com/mlflow/mlflow
RFDiffusion | MODEL WEIGHTS |	BSD | https://github.com/RosettaCommons/RFdiffusion
ProteinMPNN | ProteinMPNN 	| MIT | https://github.com/dauparas/ProteinMPNN
ProteinMPNN | Numpy |	BSD-3 | https://github.com/numpy/numpy
ProteinMPNN | torch==1.11.0+cu113 |	BSD-3 | https://github.com/pytorch/pytorch
ProteinMPNN | torchvision==0.12.0+cu113 |	BSD-3 |  https://github.com/pytorch/vision
ProteinMPNN | torchaudio==0.11.0 | BSD-2 | https://github.com/pytorch/audio
ProteinMPNN | mlflow==2.15.1 | Apache2.0 | https://github.com/mlflow/mlflow
ProteinMPNN | cloudpickle==2.2.1 | BSD-3 | https://github.com/cloudpipe/cloudpickle
ProteinMPNN | biopython==1.79 | [BioPython License Agreement](https://github.com/biopython/biopython/blob/master/LICENSE.rst) |  https://github.com/biopython/biopython
ProteinMPNN | MODEL WEIGHTS | MIT | https://github.com/dauparas/ProteinMPNN
Alphafold | AlphaFold (2.3.2) | Apache2.0 | https://github.com/google-deepmind/alphafold
Alphafold | other dependencies | we provide a file of requirements per alphafold's own [repo](https://github.com/google-deepmind/alphafold), see [yml file](https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/modules/large_molecule/alphafold/alphafold_v2.3.2/envs/alphafold_env.yml) for further details |
Alphafold | MODEL WEIGHTS | CC BY 4.0
ESMfold | ESMFold |	MIT | https://github.com/facebookresearch/esm
ESMfold | torch | BSD-3 | https://github.com/pytorch/pytorch
ESMfold | transformers | Apache2.0 | https://github.com/huggingface/transformers
ESMfold | accelerate | Apache2.0 | https://github.com/huggingface/transformers
ESMfold | hf_transfer==0.1.9 | Apache2.0 | https://github.com/huggingface/hf_transfer
ESMfold | MODEL WEIGHTS | MIT
Boltz-1 | Boltz-1 |	MIT | https://github.com/jwohlwend/boltz
Boltz-1 | packaging |Apache2.0 | https://github.com/pypa/packaging
Boltz-1 | ninja | Apache2.0 | https://github.com/scikit-build/ninja-python-distributions
Boltz-1 | torch==2.3.1+cu121 | BSD-3 | https://github.com/pytorch/pytorch
Boltz-1 | torchvision==0.18.1+cu121 | BSD-3 | https://github.com/pytorch/vision
Boltz-1 | mlflow==2.15.1 | Apache2.0 | https://github.com/mlflow/mlflow
Boltz-1 | cloudpickle==2.2.1 | BSD-3 | https://github.com/cloudpipe/cloudpickle
Boltz-1 | requests>=2.25.1 | Apache2.0 | https://github.com/psf/requests
Boltz-1 | boltz==0.4.0 | MIT | https://github.com/jwohlwend/boltz
Boltz-1 | rdkit | BSD-3 | https://github.com/rdkit/rdkit
Boltz-1 | absl-py==1.0.0 |	Apache2.0 | https://github.com/abseil/abseil-py
Boltz-1 | transformers>=4.41 | 	Apache2.0 | https://github.com/huggingface/transformers
Boltz-1 | sentence-transformers>=2.7 |	Apache2.0 | https://github.com/UKPLab/sentence-transformers/
Boltz-1 | pyspark |	Apache2.0 | https://github.com/apache/spark
Boltz-1 | pandas |	BSD-3 | https://github.com/pandas-dev/pandas
Boltz-1 | flash_attn==1.0.9 (optional GPU) | Apache2.0 | https://github.com/Dao-AILab/flash-attention
Boltz-1 | MODEL WEIGHTS |	MIT | https://github.com/jwohlwend/boltz
Scanpy | scanpy==1.11.4 | BSD-3 | https://github.com/scverse/scanpy
Scanpy | anndata | BSD-3 | https://github.com/scverse/anndata
Scanpy | scikit-network | BSD-3 | https://github.com/sknetwork-team/scikit-network
Scanpy | pybiomart | BSD-3 | https://github.com/jrderuiter/pybiomart
Scanpy | Numpy | BSD-3 | https://github.com/numpy/numpy
Scanpy | pandas | BSD-3 | https://github.com/pandas-dev/pandas
Scanpy | scipy | BSD-3 | https://github.com/scipy/scipy
rapids-singlecell (part of scverse) | rapids-singlecell | MIT | https://github.com/scverse/rapids_singlecell
rapids-singlecell (part of scverse) | cudf-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
rapids-singlecell (part of scverse) | cuml-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
rapids-singlecell (part of scverse) | cugraph-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
rapids-singlecell (part of scverse) | cucim-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
rapids-singlecell (part of scverse) | dask-cudf-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
rapids-singlecell (part of scverse) | nx-cugraph-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
rapids-singlecell (part of scverse) | cuxfilter-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
rapids-singlecell (part of scverse) | pylibraft-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
rapids-singlecell (part of scverse) | raft-dask-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
rapids-singlecell (part of scverse) | cuvs-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
rapids-singlecell (part of scverse) | cupy-cuda12x | MIT | https://github.com/cupy/cupy/
rapids-singlecell (part of scverse) | scikit-learn==1.5.* | BSD-3 | https://github.com/scikit-learn/scikit-learn
rapids-singlecell (part of scverse) | rmm (RAPIDS Memory Manager) | Apache-2.0 | https://github.com/rapidsai/rmm
Chemprop | chemprop>=2.0.0 | MIT | https://github.com/chemprop/chemprop
Chemprop | lightning | Apache2.0 | https://github.com/Lightning-AI/pytorch-lightning
Chemprop | scikit-learn>=1.3 | BSD-3 | https://github.com/scikit-learn/scikit-learn
Chemprop | PyTDC | MIT | https://github.com/mims-harvard/TDC
Chemprop | torch>=2.0 | BSD-3 | https://github.com/pytorch/pytorch
Chemprop | rdkit | BSD-3 | https://github.com/rdkit/rdkit
Chemprop | mlflow>=2.15 | Apache2.0 | https://github.com/mlflow/mlflow
Chemprop | cloudpickle | BSD-3 | https://github.com/cloudpipe/cloudpickle
DiffDock | DiffDock | MIT | https://github.com/gcorso/DiffDock
DiffDock | pyyaml==6.0.1 | MIT | https://github.com/yaml/pyyaml
DiffDock | scipy==1.7.3 | BSD-3 | https://github.com/scipy/scipy
DiffDock | networkx==2.6.3 | BSD-3 | https://github.com/networkx/networkx
DiffDock | biopython==1.79 | [BioPython License Agreement](https://github.com/biopython/biopython/blob/master/LICENSE.rst) | https://github.com/biopython/biopython
DiffDock | rdkit-pypi==2022.03.5 | BSD-3 | https://github.com/rdkit/rdkit
DiffDock | e3nn==0.5.1 | MIT | https://github.com/e3nn/e3nn
DiffDock | spyrmsd==0.5.2 | MIT | https://github.com/RMeli/spyrmsd
DiffDock | biopandas==0.4.1 | BSD-3 | https://github.com/BioPandas/biopandas
DiffDock | prody==2.6.1 | MIT | https://github.com/prody/ProDy
DiffDock | fair-esm==2.0.0 | MIT | https://github.com/facebookresearch/esm
DiffDock | torch-geometric==2.2.0 | MIT | https://github.com/pyg-team/pytorch_geometric
DiffDock | torch-scatter==2.1.1 | MIT | https://github.com/rusty1s/pytorch_scatter
DiffDock | torch-sparse==0.6.17 | MIT | https://github.com/rusty1s/pytorch_sparse
DiffDock | torch-cluster==1.6.1 | MIT | https://github.com/rusty1s/pytorch_cluster
DiffDock | pandas==1.5.3 | BSD-3 | https://github.com/pandas-dev/pandas
Proteina-Complexa | Proteina-Complexa | MIT | https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa
Proteina-Complexa | torch==2.7.1 | BSD-3 | https://github.com/pytorch/pytorch
Proteina-Complexa | lightning==2.6.1 | Apache2.0 | https://github.com/Lightning-AI/pytorch-lightning
Proteina-Complexa | hydra-core==1.3.1 | MIT | https://github.com/facebookresearch/hydra
Proteina-Complexa | omegaconf==2.3.0 | BSD-3 | https://github.com/omry/omegaconf
Proteina-Complexa | torch_geometric==2.7.0 | MIT | https://github.com/pyg-team/pytorch_geometric
Proteina-Complexa | torch_scatter==2.1.2 | MIT | https://github.com/rusty1s/pytorch_scatter
Proteina-Complexa | torch_sparse==0.6.18 | MIT | https://github.com/rusty1s/pytorch_sparse
Proteina-Complexa | torch_cluster==1.6.3 | MIT | https://github.com/rusty1s/pytorch_cluster
Proteina-Complexa | biotite==1.6.0 | BSD-3 | https://github.com/biotite-dev/biotite
Proteina-Complexa | loralib==0.1.2 | MIT | https://github.com/microsoft/LoRA
Proteina-Complexa | einops==0.8.2 | MIT | https://github.com/arogozhnikov/einops
Proteina-Complexa | transformers==5.5.0 | Apache2.0 | https://github.com/huggingface/transformers
Proteina-Complexa | jaxtyping | MIT | https://github.com/patrick-kidger/jaxtyping
NetSolP | NetSolP-1.0 | BSD-3-Clause | https://github.com/tvinet/NetSolP-1.0
NetSolP | onnxruntime==1.20.1 | MIT | https://github.com/microsoft/onnxruntime
NetSolP | fair-esm==2.0.0 | MIT | https://github.com/facebookresearch/esm
NetSolP | torch==2.7.1 | BSD-3 | https://github.com/pytorch/pytorch
NetSolP | numpy==1.26.4 | BSD-3 | https://github.com/numpy/numpy
NetSolP | pandas==1.5.3 | BSD-3 | https://github.com/pandas-dev/pandas
NetSolP | mlflow==2.22.0 | Apache2.0 | https://github.com/mlflow/mlflow
NetSolP | cloudpickle==2.0.0 | BSD-3 | https://github.com/cloudpipe/cloudpickle
NetSolP | databricks-sdk==0.50.0 | Apache2.0 | https://pypi.org/project/databricks-sdk/
NetSolP | databricks-sql-connector==4.0.2 | Apache2.0 | https://github.com/databricks/databricks-sql-python
NetSolP | MODEL WEIGHTS — Solubility_ESM12_0_quantized.onnx (~85 MB, split 0 of the upstream 5-fold ESM-12 ensemble) + ESM12_alphabet.pkl, committed under modules/small_molecule/netsolp/netsolp_v1/weights/ | BSD-3-Clause | https://services.healthtech.dtu.dk/services/NetSolP-1.0/
PLTNUM | PLTNUM (vendored PLTNUM_PreTrainedModel class) | MIT | https://github.com/sagawatatsuya/PLTNUM
PLTNUM | torch==2.7.1 | BSD-3 | https://github.com/pytorch/pytorch
PLTNUM | transformers==4.46.3 | Apache2.0 | https://github.com/huggingface/transformers
PLTNUM | safetensors==0.4.5 | Apache2.0 | https://github.com/huggingface/safetensors
PLTNUM | huggingface-hub==0.26.2 | Apache2.0 | https://github.com/huggingface/huggingface_hub
PLTNUM | numpy==1.26.4 | BSD-3 | https://github.com/numpy/numpy
PLTNUM | pandas==1.5.3 | BSD-3 | https://github.com/pandas-dev/pandas
PLTNUM | mlflow==2.22.0 | Apache2.0 | https://github.com/mlflow/mlflow
PLTNUM | cloudpickle==2.0.0 | BSD-3 | https://github.com/cloudpipe/cloudpickle
PLTNUM | databricks-sdk==0.50.0 | Apache2.0 | https://pypi.org/project/databricks-sdk/
PLTNUM | databricks-sql-connector==4.0.2 | Apache2.0 | https://github.com/databricks/databricks-sql-python
PLTNUM | MODEL WEIGHTS (HuggingFace sagawa/PLTNUM-ESM2-NIH3T3) | MIT | https://huggingface.co/sagawa/PLTNUM-ESM2-NIH3T3
PLTNUM | ESM-2 650M backbone (facebook/esm2_t33_650M_UR50D) | MIT | https://github.com/facebookresearch/esm
DeepSTABp | DeepSTABp (vendored deepSTAPpMLP class) | MIT | https://github.com/CSBiology/deepStabP
DeepSTABp | torch==2.7.1 | BSD-3 | https://github.com/pytorch/pytorch
DeepSTABp | transformers==4.46.3 | Apache2.0 | https://github.com/huggingface/transformers
DeepSTABp | safetensors==0.4.5 | Apache2.0 | https://github.com/huggingface/safetensors
DeepSTABp | huggingface-hub==0.26.2 | Apache2.0 | https://github.com/huggingface/huggingface_hub
DeepSTABp | pytorch-lightning==2.5.5 | Apache2.0 | https://github.com/Lightning-AI/pytorch-lightning
DeepSTABp | sentencepiece==0.2.0 | Apache2.0 | https://github.com/google/sentencepiece
DeepSTABp | biopython==1.84 | [BioPython License Agreement](https://github.com/biopython/biopython/blob/master/LICENSE.rst) | https://github.com/biopython/biopython
DeepSTABp | numpy==1.26.4 | BSD-3 | https://github.com/numpy/numpy
DeepSTABp | pandas==1.5.3 | BSD-3 | https://github.com/pandas-dev/pandas
DeepSTABp | mlflow==2.22.0 | Apache2.0 | https://github.com/mlflow/mlflow
DeepSTABp | cloudpickle==2.0.0 | BSD-3 | https://github.com/cloudpipe/cloudpickle
DeepSTABp | databricks-sdk==0.50.0 | Apache2.0 | https://pypi.org/project/databricks-sdk/
DeepSTABp | databricks-sql-connector==4.0.2 | Apache2.0 | https://github.com/databricks/databricks-sql-python
DeepSTABp | MODEL WEIGHTS — MLP head (~80 MB, fetched from upstream raw URL at registration time) | MIT | https://github.com/CSBiology/deepStabP/raw/main/src/Api/trained_model/b25_sampled_10k_tuned_2_d01/checkpoints/
DeepSTABp | ProtT5-XL backbone (Rostlab/prot_t5_xl_uniref50) | MIT (verified at parent ProtTrans repo) | https://github.com/agemagician/ProtTrans
MHCflurry | mhcflurry==2.2.1 | Apache2.0 | https://github.com/openvax/mhcflurry
MHCflurry | torch==2.7.1 | BSD-3 | https://github.com/pytorch/pytorch
MHCflurry | numpy==1.26.4 | BSD-3 | https://github.com/numpy/numpy
MHCflurry | pandas==2.2.3 | BSD-3 | https://github.com/pandas-dev/pandas
MHCflurry | scikit-learn==1.5.2 | BSD-3 | https://github.com/scikit-learn/scikit-learn
MHCflurry | biopython==1.84 | [BioPython License Agreement](https://github.com/biopython/biopython/blob/master/LICENSE.rst) | https://github.com/biopython/biopython
MHCflurry | mlflow==2.22.0 | Apache2.0 | https://github.com/mlflow/mlflow
MHCflurry | cloudpickle==2.0.0 | BSD-3 | https://github.com/cloudpipe/cloudpickle
MHCflurry | databricks-sdk==0.50.0 | Apache2.0 | https://pypi.org/project/databricks-sdk/
MHCflurry | databricks-sql-connector==4.0.2 | Apache2.0 | https://github.com/databricks/databricks-sql-python
MHCflurry | MODEL WEIGHTS (auto-fetched via `mhcflurry-downloads fetch models_class1_presentation`, ~150 MB) | Apache2.0 | https://github.com/openvax/mhcflurry
Genomics | glow | Apache2.0 | https://github.com/projectglow/glow
Genomics | pyspark | Apache2.0 | https://github.com/apache/spark
BioNeMo | six==1.16.0 | MIT | https://github.com/benjaminp/six
BioNeMo | numpy==1.26.4 | BSD-3 | https://github.com/numpy/numpy
BioNeMo | pandas==2.2.3 | BSD-3 | https://github.com/pandas-dev/pandas
BioNeMo | pyarrow>=14.0.0 | Apache2.0 | https://github.com/apache/arrow
BioNeMo | matplotlib>=3.8.0 | PSF/BSD | https://github.com/matplotlib/matplotlib
BioNeMo | Jinja2>=3.1.2 | BSD-3 | https://github.com/pallets/jinja
BioNeMo | protobuf>=4.23.3 | BSD-3 | https://github.com/protocolbuffers/protobuf
BioNeMo | grpcio>=1.59.0 | Apache2.0 | https://github.com/grpc/grpc
BioNeMo | grpcio-status>=1.59.0 | Apache2.0 | https://github.com/grpc/grpc
BioNeMo | databricks-sdk>=0.1.6 | Apache2.0 | https://pypi.org/project/databricks-sdk/
BioNeMo | psutil | BSD-2 | https://github.com/giampaolo/psutil
BioNeMo | pynvml | BSD-3 | https://github.com/gpuopenanalytics/pynvml
Parabricks | see BioNeMo (same docker base dependencies) | |
ESM2 Embeddings | torch==2.3.1 | BSD-3 | https://github.com/pytorch/pytorch
ESM2 Embeddings | transformers==4.41.2 | Apache2.0 | https://github.com/huggingface/transformers

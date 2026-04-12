# Genesis Workbench
### **A Blue print for a Life Sciences application on Databricks**
## Introduction

### $${\color{orange}Opportunity}$$
<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/genai_ls.png" alt="Generative AI in Life Sciences" width="700"/>

Generative AI is revolutionizing the life sciences by harnessing multiple foundational models tailored to various biological domains. These models, trained on vast biological datasets including genomic sequences, protein structures, molecular interactions, and cellular behaviors, enable advanced capabilities such as predictive modeling, drug discovery, and synthetic biology design. By integrating diverse biological data into unified frameworks, foundational models can generate novel hypotheses, simulate complex biochemical pathways, and predict molecular folding with unprecedented accuracy. This accelerates the identification of therapeutic targets, optimizes compound screening, and personalizes medicine by modeling patient-specific responses. Additionally, the synergy of large-scale language models specialized in biological text mining facilitates the extraction and synthesis of biomedical knowledge from the growing scientific literature.

### $${\color{orange}Challenges}$$
<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/sad_scientists.png" alt="Generative AI in Life Sciences" width="700"/>

Despite their expertise in biology, many highly talented life science scientists find themselves struggling to set up advanced biological models due to the burden of non-biological tasks. These challenges include technical complexities such as configuring CUDA environments for GPU acceleration, which is essential for efficiently training large models. Additionally, scientists often need to create and manage complex workflows that automate data processing, model training, and validation—a task that requires skills outside traditional biological training. Data engineering also poses a significant hurdle, involving the collection, cleaning, and integration of diverse biological datasets while ensuring compliance with data governance policies to maintain privacy and reproducibility. These non-biological demands divert valuable time and focus away from the core scientific research, slowing down progress and innovation in applying generative AI models in life sciences. Addressing this gap requires interdisciplinary collaboration and improved tool accessibility that lowers the technical barriers for biological researchers.

### $${\color{orange}Genesis}$$ $${\color{orange}Workbench}$$
<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/gwb.png" alt="Generative AI in Life Sciences" width="700"/>

- Genesis Workbench offers a blueprint for using Databricks capabilities—like automated workflows, GPU clusters, model serving, and MLflow—to accelerate AI-driven life sciences research.

- It features an intuitive Databricks Apps interface with pre-packaged biological models and tailored workflows, enabling scientists to start quickly without complex setup.

- In collaboration with NVIDIA, BioNeMo—a generative AI framework for digital biology—is integrated for easy access to advanced pre-trained models.The BioNeMo models are optimized for NVIDIA hardware, delivering high performance and scalability for enterprise workloads.

- Being open source, Genesis Workbench provides extensible templates for AI engineers, reducing non-biological workload and promoting rapid innovation in generative AI for biology.

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/happy_scientists.png" alt="Generative AI in Life Sciences" width="700"/>

## $${\color{orange}Architecture}$$ $${\color{orange}Diagram}$$

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/architecture.png" alt="Architecture" width="700"/>

### $${\color{orange}Important}$$ $${\color{orange}Disclaimer}$$

**NVIDIA, the NVIDIA logo, and NVIDIA BioNeMo are trademarks or registered trademarks of NVIDIA Corporation in the United States and other countries. All other product names, trademarks, and registered trademarks are the property of their respective owners.**

**References to third-party products or services, including NVIDIA BioNeMo, are for informational purposes only and do not constitute an endorsement or affiliation. This material is not sponsored or endorsed by NVIDIA Corporation. The information provided here is for general informational purposes and should not be interpreted as specific advice or a warranty of suitability for any particular use.**

**Use of NVIDIA BioNeMo and related technologies should comply with all relevant licensing terms, trademarks, and applicable regulations.**

## $${\color{orange}Inside}$$ $${\color{orange}Genesis}$$ $${\color{orange}Workbench}$$

- Scripts to deploy Genesis Workbench core module in your workspace
- Scripts to deploy below modules:
	- Single Cell module that deploys and uses scGPT, SCimilarity, Scanpy and Rapids-SingleCell
	- Protein Studies module that deploys and uses ESMFold, ESM2 Embeddings, Alphafold2, ProteinMPNN, RFDiffusion and Boltz
	- Small Molecule module that deploys and uses Chemprop, DiffDock and Proteina-Complexa
	- Disease Biology module for VCF ingestion, variant annotation and GWAS analysis
	- BioNeMo container definitions and workflows
	- Parabricks container definitions and workflows
	- Access Management, Monitoring and Dashboards

## $${\color{orange}Installation}$$
Read the [Installation Guide](https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/Installation.md)

## $${\color{orange}Changelog}$$
See [CHANGELOG.md](CHANGELOG.md) for deployment fixes, known issues, and configuration notes.

## $${\color{orange}Troubleshooting}$$
Read the Troubleshooting guide

## $${\color{orange}License}$$
Please see LICENSE for the details of the license. 

Some packages, tools, and code used inside individual tutorials are under their own licenses as described therein. Please ensure you read the details and licensing of individual tools. Other thrid party packages are used in tutorials within this accelerator and have their own licensing, as laid out in the table below. 

We are adding a script to build your own Databricks compatible container for NVIDIA BioNeMo. If you want to use NVIDIA BioNeMo in Genesis Workbench, please follow the instructions to build the container and push the image to your container repository. 

NVIDIA GPUs and cudatoolkit may be used in multiple places so you should consider the NVIDIA EULA(link) when using code in this package.

Module | Package | License | Source
-------- | ------- | ------- | --------
core | streamlit | Apache2.0 | https://github.com/streamlit
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
Alphafold | other dependencies | we provide a file of requirements per alphafold's own [repo](https://github.com/google-deepmind/alphafold), see [yml file](https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/modules/protein_studies/alphafold/alphafold_v2.3.2/envs/alphafold_env.yml) for further details |
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
Rapids-SingleCell | rapids-singlecell | MIT | https://github.com/scverse/rapids_singlecell
Rapids-SingleCell | cudf-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
Rapids-SingleCell | cuml-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
Rapids-SingleCell | cugraph-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
Rapids-SingleCell | cucim-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
Rapids-SingleCell | dask-cudf-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
Rapids-SingleCell | nx-cugraph-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
Rapids-SingleCell | cuxfilter-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
Rapids-SingleCell | pylibraft-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
Rapids-SingleCell | raft-dask-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
Rapids-SingleCell | cuvs-cu12==25.10.* | Apache-2.0 | https://rapids.ai/
Rapids-SingleCell | cupy-cuda12x | MIT | https://github.com/cupy/cupy/
Rapids-SingleCell | scikit-learn==1.5.* | BSD-3 | https://github.com/scikit-learn/scikit-learn
Rapids-SingleCell | rmm (RAPIDS Memory Manager) | Apache-2.0 | https://github.com/rapidsai/rmm
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
Disease Biology | glow | Apache2.0 | https://github.com/projectglow/glow
Disease Biology | pyspark | Apache2.0 | https://github.com/apache/spark
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



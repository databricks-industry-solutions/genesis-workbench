# Genesis Workbench

## Introduction

Generative AI is revolutionizing the life sciences by harnessing multiple foundational models tailored to various biological domains. These models, trained on vast biological datasets including genomic sequences, protein structures, molecular interactions, and cellular behaviors, enable advanced capabilities such as predictive modeling, drug discovery, and synthetic biology design. By integrating diverse biological data into unified frameworks, foundational models can generate novel hypotheses, simulate complex biochemical pathways, and predict molecular folding with unprecedented accuracy. This accelerates the identification of therapeutic targets, optimizes compound screening, and personalizes medicine by modeling patient-specific responses. Additionally, the synergy of large-scale language models specialized in biological text mining facilitates the extraction and synthesis of biomedical knowledge from the growing scientific literature.

Despite their expertise in biology, many highly talented life science scientists find themselves struggling to set up advanced biological models due to the burden of non-biological tasks. These challenges include technical complexities such as configuring CUDA environments for GPU acceleration, which is essential for efficiently training large models. Additionally, scientists often need to create and manage complex workflows that automate data processing, model training, and validation—a task that requires skills outside traditional biological training. Data engineering also poses a significant hurdle, involving the collection, cleaning, and integration of diverse biological datasets while ensuring compliance with data governance policies to maintain privacy and reproducibility. These non-biological demands divert valuable time and focus away from the core scientific research, slowing down progress and innovation in applying generative AI models in life sciences. Addressing this gap requires interdisciplinary collaboration and improved tool accessibility that lowers the technical barriers for biological researchers.

Genesis Workbench from Databricks leverages the powerful capabilities of the Databricks platform—such as automated workflows, GPU-enabled clusters, model serving, and MLflow for experiment tracking and lifecycle management—to streamline and accelerate AI-driven research in life sciences. It builds an intuitive user interface using Databricks Apps to offer pre-packaged biological models alongside a curated set of workflows tailored specifically for life science scientists, enabling them to quickly get started without dealing with complex setup or infrastructure management.

Moreover, because Genesis Workbench is open source, it provides AI engineers with blueprints that serve as extensible templates, allowing easy customization and expansion to meet unique customer needs. This combination of robust platform features and flexible, ready-to-use components significantly reduces the non-biological workload on researchers and fosters faster innovation with generative AI in biology.

## Architecture Diagram

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/architecture.png" alt="Architecture" width="700"/>

#### Important Disclaimer

**NVIDIA, the NVIDIA logo, and NVIDIA BioNeMo are trademarks or registered trademarks of NVIDIA Corporation in the United States and other countries. All other product names, trademarks, and registered trademarks are the property of their respective owners.**

**References to third-party products or services, including NVIDIA BioNeMo, are for informational purposes only and do not constitute an endorsement or affiliation. This material is not sponsored or endorsed by NVIDIA Corporation. The information provided here is for general informational purposes and should not be interpreted as specific advice or a warranty of suitability for any particular use.**

**Use of NVIDIA BioNeMo and related technologies should comply with all relevant licensing terms, trademarks, and applicable regulations.**

## Inside Genesis Workbench

- Scripts to deploy Genesis Workbench core module in your workspace
- Scripts to deploy below modules:
	- Single Cell module that deploys and uses scGPT and SCimilarity
	- Protein Studies module that deploys and uses ESMFold, Aphafold2, ProteinMPNN, rfdiffusion and Boltz
	- BioNeMo container definitions and workflows
	- Access Management, Monitoring and Dashboards

## Installation
Read the Installation guide

## Troubleshooting
Read the Troubleshooting guide

## License
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
SCimilarity | SCimilarity |	Apache2.0 | https://github.com/Genentech/scimilarity
SCimilarity | Mlflow	| Apache2.0 | https://github.com/mlflow/mlflow
SCimilarity | scanpy |	BSD 3-Clause | https://github.com/scverse/scanpy
SCimilarity | numcodecs |	MIT | https://github.com/zarr-developers/numcodecs
SCimilarity | tbb	| Apache2.0 | https://github.com/uxlfoundation/oneTBB [i think this is it?]
SCimilarity | typing_extensions	| PSF | https://github.com/python/typing_extensions
SCimilarity | numpy |	BSD [https://numpy.org/] | https://github.com/numpy/numpy
SCimilarity | pandas | BSD 3-Clause | https://github.com/pandas-dev/pandas
SCimilarity | uv | Apache2.0 | https://github.com/astral-sh/uv
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
Alphafold | other dependencies | we provide a file of requirements per alphafold's own [repo](https://github.com/google-deepmind/alphafold), see [yml file](https://github.com/databricks-industry-solutions/hls-proteinfolding/blob/main/tutorials/alphafold/workflow/envs/alphafold_env.yml) for further details |
Alphafold | MODEL WEIGHTS | CC BY 4.0
ESMfold | ESMFold |	MIT | https://github.com/facebookresearch/esm
ESMfold | torch | BSD-3 | https://github.com/pytorch/pytorch
ESMfold | transformers | Apache2.0 | https://github.com/huggingface/transformers
ESMfold | accelerate | Apache2.0 | https://github.com/huggingface/transformers
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
Boltz-1 | MODEL WEIGHTS |	MIT | https://github.com/jwohlwend/boltz




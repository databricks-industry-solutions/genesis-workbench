# Databricks notebook source
# MAGIC %md
# MAGIC ### Installing dependencies

# COMMAND ----------

# DBTITLE 1,[gwb] pip install from requirements list
# MAGIC %cat ../requirements.txt
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_yyang_genesis_workbench", "Schema")
dbutils.widgets.text("model_name", "scgpt", "Model Name")
dbutils.widgets.text("experiment_name", "scgpt_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "yang.yang@databricks.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "scgpt_cache_dir", "Cache dir")


# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
experiment_name = dbutils.widgets.get("experiment_name")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
cache_dir = dbutils.widgets.get("cache_dir")

# cache dir is for Hugging Face models or artifacts
print(f"Cache dir: {cache_dir}")
cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"
print(f"Cache full path: {cache_full_path}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{cache_dir}")
print(f"Cache full path: {cache_full_path} established")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Download data and model into Volume/Folder
# MAGIC Pretrained scGPT Model Zoo
# MAGIC
# MAGIC Here is the list of pretrained models. Please find the links for downloading the checkpoint folders. We recommend using the `whole-human` model for most applications by default. If your fine-tuning dataset shares similar cell type context with the training data of the organ-specific models, these models can usually demonstrate competitive performance as well. A paired vocabulary file mapping gene names to ids is provided in each checkpoint folder. If ENSEMBL ids are needed, please find the conversion at [gene_info.csv](https://github.com/bowang-lab/scGPT/files/13243634/gene_info.csv).
# MAGIC
# MAGIC | Model name                | Description                                             | Download                                                                                     |
# MAGIC | :------------------------ | :------------------------------------------------------ | :------------------------------------------------------------------------------------------- |
# MAGIC | whole-human (recommended) | Pretrained on 33 million normal human cells.            | [link](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing) |
# MAGIC | continual pretrained      | For zero-shot cell embedding related tasks.             | [link](https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB?usp=sharing) |
# MAGIC | brain                     | Pretrained on 13.2 million brain cells.                 | [link](https://drive.google.com/drive/folders/1vf1ijfQSk7rGdDGpBntR5bi5g6gNt-Gx?usp=sharing) |
# MAGIC | blood                     | Pretrained on 10.3 million blood and bone marrow cells. | [link](https://drive.google.com/drive/folders/1kkug5C7NjvXIwQGGaGoqXTk_Lb_pDrBU?usp=sharing) |
# MAGIC | heart                     | Pretrained on 1.8 million heart cells                   | [link](https://drive.google.com/drive/folders/1GcgXrd7apn6y4Ze_iSCncskX3UsWPY2r?usp=sharing) |
# MAGIC | lung                      | Pretrained on 2.1 million lung cells                    | [link](https://drive.google.com/drive/folders/16A1DJ30PT6bodt4bWLa4hpS7gbWZQFBG?usp=sharing) |
# MAGIC | kidney                    | Pretrained on 814 thousand kidney cells                 | [link](https://drive.google.com/drive/folders/1S-1AR65DF120kNFpEbWCvRHPhpkGK3kK?usp=sharing) |
# MAGIC | pan-cancer                | Pretrained on 5.7 million cells of various cancer types | [link](https://drive.google.com/drive/folders/13QzLHilYUd0v3HTwa_9n4G4yEF-hdkqa?usp=sharing) |
# MAGIC

# COMMAND ----------

#: downnload model
import gdown
model_dir = f'{cache_full_path}/models/'
print(f"Model dir: {model_dir}")

# with the file ID, for example, the | blood                     | Pretrained on 10.3 million blood and bone marrow cells. |
# [link](https://drive.google.com/drive/folders/1kkug5C7NjvXIwQGGaGoqXTk_Lb_pDrBU?usp=sharing) |
########################################################
# Feb 19, 2026 update to use whole-human (recommended) model instead of blood model. | [link](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing) |
# ########################################################
id = "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
# gdown.download(id=id, )
gdown_return = gdown.download_folder(id=id, output=f"{model_dir}/")

model_dir = gdown_return[0].rsplit('/', 1)[0]
print(f"Model dir: {model_dir}")


# COMMAND ----------

#: downnload data
import wget
import os
# old url: https://figshare.com/ndownloader/files/25717328, which will download a 0kb size file due to 
# This usually happens because the Figshare “ndownloader” URL now returns a blocked/redirect/challenge response to non-browser clients, so your script creates the output file but receives no real payload (or gets a 403/HTML instead of the .h5ad). Servers commonly enforce this via User-Agent or similar bot checks, which shows up as HTTP 403 “Forbidden” for bare urllib-style requests
file_url = "https://api.figshare.com/v2/file/download/25717328"
file_path = f'{cache_full_path}/data/'
os.makedirs(os.path.dirname(file_path), exist_ok=True)
file_path = f'{cache_full_path}/data/file.h5ad'
print(f"Dataset dir: {file_path}")

wget.download(file_url, str(file_path))



# COMMAND ----------

#: setup credential
import os

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
os.environ["SQL_WAREHOUSE"]=sql_warehouse_id
os.environ["IS_TOKEN_AUTH"]="Y"
os.environ["DATABRICKS_TOKEN"]=databricks_token

# COMMAND ----------

# MAGIC %md
# MAGIC #### Package ScGPT into UC using Mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC #### Import Library

# COMMAND ----------

import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.insert(0, "../")
import scgpt
import scanpy
from scanpy import AnnData
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed

import mlflow.pyfunc
import torch
import scipy

os.environ["KMP_WARNINGS"] = "off"
# warnings.filterwarnings("ignore")

from typing import TypedDict, Dict, List, Tuple, Any, Optional


# COMMAND ----------

# MAGIC %md
# MAGIC # Define the class TransformerModelWrapper inheriting mlflow.pyfunc.PythonModel

# COMMAND ----------

class TransformerModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, special_tokens=["<pad>", "<cls>", "<eoc>"]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.special_tokens = special_tokens

    def load_context(self, context):
        self.model_file = context.artifacts["model_file"]
        self.model_config_file = context.artifacts["model_config_file"]
        self.vocab_file = context.artifacts["vocab_file"]

        self.vocab = GeneVocab.from_file(self.vocab_file)
        for s in self.special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)
        self.gene2idx = self.vocab.get_stoi()
        self.ntokens = len(self.vocab)

        with open(self.model_config_file, "r") as f:
            self.model_configs = json.load(f)
        print(
            f"Resume model from {self.model_file}, the model args will override the "
            f"config {self.model_config_file}."
        )
        self.embsize = self.model_configs["embsize"]
        self.nhead = self.model_configs["nheads"]
        self.d_hid = self.model_configs["d_hid"]
        self.nlayers = self.model_configs["nlayers"]
        self.n_layers_cls = self.model_configs["n_layers_cls"]
        self.pad_value = self.model_configs["pad_value"]
        self.mask_value = self.model_configs["mask_value"]
        self.n_bins = self.model_configs["n_bins"]
        self.n_hvg = self.model_configs["n_hvg"]

    def preprocess(
        self,
        context,
        input_data_path: str = None,
        input_dataframe: pd.DataFrame = None,
        data_is_raw=False,
        params={
            "data_is_raw": False,
            "use_key": "X",
            "filter_gene_by_counts": 3,
            "filter_cell_by_counts": False,
            "normalize_total": 1e4,
            "result_normed_key": "X_normed",
            "log1p": False,
            "result_log1p_key": "X_log1p",
            "subset_hvg": 1200,
            "hvg_flavor": "cell_ranger",
            "binning": 51,
            "result_binned_key": "X_binned",
        },
    ):
        if input_data_path and input_dataframe is None:
            loaded_data = scanpy.read(str(input_data_path), cache=True)
            ori_batch_col = "batch"
            loaded_data.obs["celltype"] = loaded_data.obs["final_annotation"].astype(
                str
            )
        elif input_dataframe is not None and isinstance(input_dataframe, pd.DataFrame):
            print(input_dataframe)
            print(input_dataframe.shape)
            #
            adata_sparsematrix = scipy.sparse.csr_matrix(input_dataframe['adata_sparsematrix'][0])
            adata_obs = pd.read_json(input_dataframe['adata_obs'][0], orient='split')
            adata_var = pd.read_json(input_dataframe['adata_var'][0], orient='split')
            loaded_data = scanpy.AnnData(adata_sparsematrix, obs=adata_obs, var=adata_var)
            ori_batch_col = "batch"
            loaded_data.obs["celltype"] = loaded_data.obs["final_annotation"].astype(
                str
            )
        else:
            raise ValueError("No input data provided.")

        self.data_is_raw = params.get("data_is_raw", data_is_raw)

        preprocessor = Preprocessor(
            use_key=params.get("use_key", "X"),
            filter_gene_by_counts=params.get("filter_gene_by_counts", 3),
            filter_cell_by_counts=params.get("filter_cell_by_counts", False),
            normalize_total=params.get("normalize_total", 1e4),
            result_normed_key=params.get("result_normed_key", "X_normed"),
            log1p=params.get("log1p", self.data_is_raw),
            result_log1p_key=params.get("result_log1p_key", "X_log1p"),
            subset_hvg=params.get("subset_hvg", self.n_hvg),
            hvg_flavor=params.get("hvg_flavor", "seurat_v3" if self.data_is_raw else "cell_ranger"),
            binning=params.get("binning", self.n_bins),
            result_binned_key=params.get("result_binned_key", "X_binned"),
        )

        self.n_input_bins = params.get("binning", self.n_bins)
        preprocessor(loaded_data, batch_key=ori_batch_col)

        return loaded_data

    def filter(self, gene_embeddings: np.ndarray, preprocessed_data: AnnData) -> Dict[str, np.ndarray]:
        gene_embeddings = {
            gene: gene_embeddings[i]
            for i, gene in enumerate(self.gene2idx.keys())
            if gene in preprocessed_data.var.index.tolist()
        }
        print("Retrieved gene embeddings for {} genes.".format(len(gene_embeddings)))

        return gene_embeddings

    def predict(
        self,
        context,
        model_input: pd.DataFrame = None,
        params: Dict[str, mlflow.types.DataType] = {
            "need_preprocess": True,
            "input_data_path": None,
            "data_is_raw": False,
            "use_key": "X",
            "filter_gene_by_counts": 3,
            "filter_cell_by_counts": False,
            "normalize_total": 1e4,
            "result_normed_key": "X_normed",
            "log1p": False,
            "result_log1p_key": "X_log1p",
            "subset_hvg": 1200,
            "hvg_flavor": "cell_ranger",
            "binning": 51,
            "result_binned_key": "X_binned",
            "embsize": 512,
            "nhead": 8,
            "d_hid": 512,
            "nlayers": 12,
            "n_layers_cls": 3
        },
    ) -> Dict[str, np.ndarray]:
        print(
            "`model_input` is only needed when users have their own .h5ad gene file to be preprocessed and used to filter the 30k gene embeding result from the model."
        )
        if params["need_preprocess"]:
            assert (
                model_input is not None
            ), "'model_input' must be provided if 'need_preprocess' is True"
            preprocessed_data = self.preprocess(
                context,
                input_data_path=params["input_data_path"],
                input_dataframe=model_input,
                data_is_raw=params["data_is_raw"],
                params=params
            )
            print("preprocessing finished!")

        print("Now defining the TransformerModel!")
        self.model = TransformerModel(
            ntoken=params.get("ntokens", self.ntokens),
            d_model=params.get("embsize", self.embsize),
            nhead=params.get("nhead", self.nhead),
            d_hid=params.get("d_hid", self.d_hid),
            nlayers=params.get("nlayers", self.nlayers),
            vocab=self.vocab,
            pad_value=params.get("pad_value", self.pad_value),
            n_input_bins=params.get(
                "n_input_bins",
                self.n_input_bins if hasattr(self, "n_input_bins") else self.n_bins,
            ),
        )

        try:
            self.model.load_state_dict(torch.load(self.model_file))
            print(f"Loading all model params from {self.model_file}")
        except:
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(self.model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)

        self.model.to(self.device)
        self.model.eval()

        gene_ids = np.array([id for id in self.gene2idx.values()])
        gene_embeddings = self.model.encoder(
            torch.tensor(gene_ids, dtype=torch.long).to(self.device)
        )
        gene_embeddings = gene_embeddings.detach().cpu().numpy()

        if params["need_preprocess"]:
            gene_embeddings_dict = self.filter(gene_embeddings, preprocessed_data)
        else:
            gene_embeddings_dict = {
                gene: gene_embeddings[i] for i, gene in enumerate(self.gene2idx.keys())
            }
        return gene_embeddings_dict


# COMMAND ----------

# MAGIC %md
# MAGIC # Dry Load Test
# MAGIC

# COMMAND ----------

# DBTITLE 1,Setting Up Model Directory and File Paths
#
model_config_file = f"{model_dir}/args.json"
model_file = f"{model_dir}/best_model.pt"
vocab_file = f"{model_dir}/vocab.json"


print(model_dir)
print(model_config_file)
print(model_file)
print(vocab_file)
print(f"Dataset dir: {file_path}")

# COMMAND ----------

# DBTITLE 1,Instantiate the model class
from mlflow.pyfunc import PythonModelContext
context = PythonModelContext(artifacts = {
  "model_file": model_file,
  "model_config_file": model_config_file,
  "vocab_file": vocab_file
},model_config={})

tf_model = TransformerModelWrapper(special_tokens = ["<pad>", "<cls>", "<eoc>"])
tf_model.load_context(context)


# COMMAND ----------

# MAGIC %md
# MAGIC __the next step is the most tricky part as you need to make sure each column in the `input_data` dataframe is the right format serializable and the exact type which will be converted back after sending the str json object into the model serving endpoint and later restored back within the class.__
# MAGIC
# MAGIC E.g.,
# MAGIC __correct vs. wrong__.
# MAGIC adata_subset.X.toarray().tolist() vs. adata_subset.X.toarray()
# MAGIC adata_subset.obs.to_json(orient='split') vs. adata_subset.obs
# MAGIC

# COMMAND ----------

# DBTITLE 1,Loading and Preparing Immune Human Dataset
# Specify data path; here we load the Immune Human dataset
adata = scanpy.read(
    str(file_path), cache=True
)  # 33506 × 12303
ori_batch_col = "batch"
adata.obs["celltype"] = adata.obs["final_annotation"].astype(str)


# COMMAND ----------

dir(adata)

# COMMAND ----------

# DBTITLE 1,Generate input example from Subset of AnnData Object
adata_subset = adata[:1000,:1000]
#: version 1 (original format): input_data = pd.DataFrame({'adata_sparsematrix': [adata_subset.X], 'adata_obs': [adata_subset.obs], 'adata_var':[adata_subset.var]})
#: version 2 (serving_input format):
input_data = pd.DataFrame({'adata_sparsematrix': [adata_subset.X.toarray().tolist()], 'adata_obs': [adata_subset.obs.to_json(orient='split')], 'adata_var':[adata_subset.var.to_json(orient='split')]})



# COMMAND ----------

input_data

# COMMAND ----------

# DBTITLE 1,Generate output example
#: use previously instantiated model (e.g., in Dry tests 3) to predict gene embeddings for adata_df and adata.obs and var
output_example = tf_model.predict(context, model_input=input_data, params={
   "need_preprocess": True,
    "input_data_path":  None, # leave None so it will use model_input=().
    "data_is_raw": False,
    "use_key": "X",  # the key in adata.layers to use as raw data
    "filter_gene_by_counts": 3,  # step 1
    "filter_cell_by_counts": False,  # step 2
    "normalize_total": 1e4,  # 3. whether to normalize the raw data and to what sum
    "result_normed_key": "X_normed",  # the key in adata.layers to store the normalized data
    "log1p": False,  # 4. whether to log1p the normalized data, it references the data_is_raw value in the same dict.
    "result_log1p_key": "X_log1p",
    "subset_hvg": 1200,  # 5. whether to subset the raw data to highly variable genes
    "hvg_flavor": "cell_ranger",
    "binning": 51,  # 6. whether to bin the raw data and to what number of bins
    "result_binned_key": "X_binned",  # the key in adata.layers to store the binned data
    "embsize": 512,
    "nhead": 8,
    "d_hid": 512,
    "nlayers": 12,
    "n_layers_cls": 3})

# COMMAND ----------

output_example.__class__, len(output_example)


# COMMAND ----------

# DBTITLE 1,Inferring Model Signature with MLflow
import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output

# Infer the signature including parameters
signature = infer_signature(
    # model_input=(adata_df, adata.obs), # in this way, signature will become [Any (required)].
    model_input=input_data,
    model_output=output_example,
    params={
      "need_preprocess": True,
      "input_data_path":  "/path/folder/data.h5ad", # leave None so it will use model_input=().
      "data_is_raw": False,
      "use_key": "X",  # the key in adata.layers to use as raw data
      "filter_gene_by_counts": 3,  # step 1
      "filter_cell_by_counts": False,  # step 2
      "normalize_total": 1e4,  # 3. whether to normalize the raw data and to what sum
      "result_normed_key": "X_normed",  # the key in adata.layers to store the normalized data
      "log1p": False,  # 4. whether to log1p the normalized data, it references the data_is_raw value in the same dict.
      "result_log1p_key": "X_log1p",
      "subset_hvg": 1200,  # 5. whether to subset the raw data to highly variable genes
      "hvg_flavor": "cell_ranger",
      "binning": 51,  # 6. whether to bin the raw data and to what number of bins
      "result_binned_key": "X_binned",  # the key in adata.layers to store the binned data
      "embsize": 512,
      "nhead": 8,
      "d_hid": 512,
      "nlayers": 12,
      "n_layers_cls": 3
      }
  )


# COMMAND ----------

signature

# COMMAND ----------

# DBTITLE 1,Logging Gene Embeddings Model with MLflow
from databricks.sdk import WorkspaceClient
import mlflow

def set_mlflow_experiment(experiment_tag, user_email):
    w = WorkspaceClient()
    mlflow_experiment_base_path = "Shared/dbx_genesis_workbench_models"
    w.workspace.mkdirs(f"/Workspace/{mlflow_experiment_base_path}")
    experiment_path = f"/{mlflow_experiment_base_path}/{experiment_tag}"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    return mlflow.set_experiment(experiment_path)

#: for input_example, it is a tuple of (input_data, params).
# This code converts the csr_matrix to a dense array using the toarray() method, making it JSON serializable.
# each nested dataframe also needs to be to_json(orient='split')
# type hint: pd.DataFrame({'adata_sparsematrix': [scipy.sparse.spmatrix], 'adata_obs': [pd.DataFrame], 'adata_var':[pd.DataFrame]})

#: Alternatively, you can avoid passing input example and pass model signature instead when logging the model. To ensure the input example is valid prior to serving, please try calling `mlflow.models.validate_serving_input` on the model uri and serving input example. A serving input example can be generated from model input example using `mlflow.models.convert_input_example_to_serving_input` function.

#: example params to be put in the (input_data, params).
params = {
    "need_preprocess": True,
    "input_data_path":  "/path/folder/data.h5ad",
    "data_is_raw": False,
    "use_key": "X",  # the key in adata.layers to use as raw data
    "filter_gene_by_counts": 3,  # step 1
    "filter_cell_by_counts": False,  # step 2
    "normalize_total": 1e4,  # 3. whether to normalize the raw data and to what sum
    "result_normed_key": "X_normed",  # the key in adata.layers to store the normalized data
    "log1p": False,  # 4. whether to log1p the normalized data, it references the data_is_raw value in the same dict.
    "result_log1p_key": "X_log1p",
    "subset_hvg": 1200,  # 5. whether to subset the raw data to highly variable genes
    "hvg_flavor": "cell_ranger",
    "binning": 51,  # 6. whether to bin the raw data and to what number of bins
    "result_binned_key": "X_binned",  # the key in adata.layers to store the binned data
    "embsize": 512,
    "nhead": 8,
    "d_hid": 512,
    "nlayers": 12,
    "n_layers_cls": 3
    }


experiment = set_mlflow_experiment(experiment_tag=experiment_name, user_email=user_email)

with mlflow.start_run(run_name=f"{model_name}", experiment_id=experiment.experiment_id) as run:
    registered_model_name = f"{catalog}.{schema}.{model_name}"

    mlflow.pyfunc.log_model(
        "scgpt",
        python_model=TransformerModelWrapper(
            special_tokens=["<pad>", "<cls>", "<eoc>"]
        ),
        artifacts={
            "model_file": str(model_file),
            "model_config_file": str(model_config_file),
            "vocab_file": str(vocab_file),
        },
        # conda_env = {
        #     'name': 'mlflow-env',
        #     'channels': ['conda-forge'],
        #     'dependencies': [
        #         'python=3.11.11',
        #         'pip',
        #         'conda=25.5.0',
        #         {
        #             'pip': [
        #                 "--extra-index-url https://download.pytorch.org/whl/cu118",
        #                 "torch==2.0.1+cu118",
        #                 "torchvision==0.15.2+cu118",
        #                 "scgpt==0.2.4",
        #                 "wandb==0.19.11",
        #                 "gdown==5.2.0",
        #                 "wget==3.2",
        #                 "ipython==8.15.0",
        #                 "cloudpickle==2.2.1",
        #                 "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
        #             ],
        #         },
        #     ],
        # },
        pip_requirements="../requirements.txt",  # Specify the path to the requirements file
        # extra_pip_requirements=[f"{package_path}geneformer-0.1.0-py3-none-any.whl"], # only one can be specified, pip or extra_pip
        signature=signature,
        #: provide input_example using tuple (input_data, params).
        input_example=(input_data, params), #: try tuple first, then change to dict. # Including an input example while logging a model offers dual benefits. Firstly, it aids in inferring the model's signature. Secondly, and just as importantly, it validates the model's requirements. This input example is utilized to execute a prediction using the model that is about to be logged, thereby enhancing the accuracy in identifying model requirement dependencies. It is highly recommended to always include an input example along with your models when you log them. # .to_dict(orient='split')
        # We offer support for input_example with params by using tuple to combine model inputs and params. See examples below:(input_data, params)
        registered_model_name=registered_model_name,
    )



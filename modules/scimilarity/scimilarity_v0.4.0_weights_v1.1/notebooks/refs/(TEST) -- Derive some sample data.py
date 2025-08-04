# Databricks notebook source
# MAGIC %md
# MAGIC #### Setup: `%run ./utils` 

# COMMAND ----------

# DBTITLE 1,install/load dependencies | # ~5mins (including initial data processing)
# MAGIC %run ./utils_20250801

# COMMAND ----------

CATALOG, DB_SCHEMA, MODEL_FAMILY, MODEL_NAME, EXPERIMENT_NAME

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,## pyfunc that we are using for get embeddings to test
import csv
from typing import Any, Dict
import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc.model import PythonModelContext
from scimilarity import CellEmbedding, CellQuery
import torch

class SCimilarity_GetEmbedding(mlflow.pyfunc.PythonModel):
    r"""Create MLFlow Pyfunc class for SCimilarity embedding model."""

    def load_context(self, context: PythonModelContext):
        r"""Intialize pre-trained SCimilarity embedding model."""
        self.ce = CellEmbedding(context.artifacts["model_path"]) 
        
    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,         
    ) -> pd.DataFrame:
        r"""Output prediction on model."""

        final_results = []

        for index, row in model_input.iterrows():
            celltype_sample_json = row['celltype_sample']
            celltype_sample_obs_json = row.get('celltype_sample_obs', None)

            # Load DataFrames and preserve indices
            celltype_sample_df = pd.read_json(celltype_sample_json, orient='split')
            if celltype_sample_obs_json:
                celltype_sample_obs_df = pd.read_json(celltype_sample_obs_json, orient='split')

            embeddings_list = []

            for sample_index, sample_row in celltype_sample_df.iterrows():
                model_input_array = np.array(sample_row['celltype_subsample'], dtype=np.float64).reshape(1, -1)
                embedding = self.ce.get_embeddings(model_input_array)
                embeddings_list.append({
                    'input_index': index,
                    'celltype_sample_index': sample_index,
                    'embedding': embedding.tolist()
                })

            embedding_df = pd.DataFrame(embeddings_list)
            embedding_df.index = embedding_df['celltype_sample_index']
            embedding_df.index.name = None

            # Merge DataFrames
            # combined_df = pd.merge(celltype_sample_df, embedding_df, left_index=True, right_index=True)
            combined_df = embedding_df
            if celltype_sample_obs_json:
                combined_df = pd.merge(combined_df, celltype_sample_obs_df, left_index=True, right_index=True)

            final_results.append(combined_df)

        output_df = pd.concat(final_results).reset_index(drop=True)

        # Reorder columns
        if 'celltype_sample_obs_df' in locals():
            # columns_order = ['input_index', 'celltype_sample_index'] + list(celltype_sample_obs_df.columns) + ['celltype_subsample', 'embedding']
            columns_order = ['input_index', 'celltype_sample_index'] + list(celltype_sample_obs_df.columns) + ['embedding']
        else:
            # columns_order = ['input_index', 'celltype_sample_index', 'celltype_subsample', 'embedding']
            columns_order = ['input_index', 'celltype_sample_index', 'embedding']
        
        output_df = output_df[columns_order]

        # Convert input_index to string
        output_df['input_index'] = output_df['input_index'].astype(str)

        final_output = output_df.copy()
        for col in final_output.select_dtypes(include=['int']).columns:
            final_output[col] = final_output[col].astype(float)
        
        return final_output

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test Local Context for Defined Model

# COMMAND ----------

# DBTITLE 1,TEST Local Context
# Create a temporary context to initialize the model
class TempContext:
    artifacts = {
                  "model_path": model_path,        
                }

temp_context = TempContext()

# Initialize the model and test with temporary context
model = SCimilarity_GetEmbedding()
model.load_context(temp_context)

# COMMAND ----------

# DBTITLE 1,XXX | sample data ref
# sampledata_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/GSE136831_subsample.h5ad"

# ## READ sample czi dataset H5AD file + align + lognorm 
# adams = sc.read(sampledata_path)

# # gene_order = derive_gene_order(databricks_instance, endpoint_name = "mmt_scimilarity_gene_order")
# # gene_order = derive_gene_order(databricks_instance, endpoint_name = gene_order_endpoint)
# aligned = align_dataset(adams, gene_order) #

# lognorm = lognorm_counts(aligned)

# ### Filter sample data to "Disease: IFP" | celltype: "myofibroblast cell" | sample_ref: "DS000011735-GSM4058950"  
# adams_ipf = lognorm[lognorm.obs["Disease"] == "IPF"].copy()

# adams_myofib = adams_ipf[
#                           adams_ipf.obs["celltype_name"] == "myofibroblast cell"
#                         ].copy()

# ## Extract list for sample_ref
# subsample = adams_myofib[
#                           adams_myofib.obs["sample"] == "DS000011735-GSM4058950" # sample ref 
#                         ].copy()

# ## extract specific index in subsample | test batch inference? 
# # query_cell = subsample[subsample.obs.index == "123942"]
# # query_cell = subsample[subsample.obs.index == "124332"]
# # query_cell = subsample[subsample.obs.index == "126138"]

# query_cell = subsample ## test multiple rows

# ## extract subsample query (1d array or list)
# X_vals: sparse.csr_matrix = query_cell.X
# X_vals

# COMMAND ----------

adams

# COMMAND ----------

adams.obs

# COMMAND ----------

adams.obs.Disease.unique()

# IPF stands for Idiopathic Pulmonary Fibrosis. It is a chronic, progressive lung disease characterized by the scarring of lung tissue, making it difficult to breathe. The term "idiopathic" means the cause of the scarring is unknown. 

# Chronic obstructive pulmonary disease (COPD) is a type of progressive lung disease characterized by chronic respiratory symptoms and airflow limitation.

# COMMAND ----------

sorted(adams.obs.celltype_name.unique().dropna().tolist())

# COMMAND ----------

# n_obs × n_vars = 23803 × 44942
adams[adams.obs.Disease == 'IPF'].obs['celltype_name'].unique() #34

# COMMAND ----------

# n_obs × n_vars = 11048 × 44942
adams[adams.obs.Disease == 'COPD']

# COMMAND ----------

# n_obs × n_vars = 15149 × 44942
adams[adams.obs.Disease == 'healthy']

# COMMAND ----------

[x / 33 for x in [23803, 11048, 15149]]

# COMMAND ----------

celltype_myofib.obs.celltype_id.unique()

# COMMAND ----------

celltype_myofib

# COMMAND ----------

celltype_myofib.obs

# COMMAND ----------

# DBTITLE 1,"subsample" indices are unique
celltype_myofib.obs.index.unique()

# COMMAND ----------

# DBTITLE 1,celltype_sample_list
celltype_sample_list = sorted(celltype_myofib.obs['sample'].unique())
len(celltype_sample_list)

# COMMAND ----------

celltype_myofib = diseasetype_ipf[
                                  diseasetype_ipf.obs["celltype_name"] == "myofibroblast cell"
                                 ].copy()

# ## Extract list for sample_ref
# celltype_sample = celltype_myofib[
#                                   celltype_myofib.obs["sample"] == "DS000011735-GSM4058950" # sample ref 
#                                  ].copy()

# ## extract specific index in celltype_sample 
# celltype_subsample = celltype_sample[celltype_sample.obs.index == "123942"]

# COMMAND ----------

# extract unique_indices_per_sample

celltype_myofib_sample = {
    sample: {
        'subsample_indices': celltype_myofib.obs[celltype_myofib.obs['sample'] == sample].index.unique().tolist(),
        'n_subsamples': len(celltype_myofib.obs[celltype_myofib.obs['sample'] == sample].index.unique().tolist())
    }
    for sample in celltype_sample_list
}

celltype_myofib_sample_pd = pd.DataFrame(celltype_myofib_sample).T.reset_index().rename(
    columns={'index': 'celltype_sample'}
)#.set_index('celltype_sample') #.rename_axis('sample_index')

celltype_myofib_sample_pd

# COMMAND ----------

# DBTITLE 1,save to Vols
celltype_myofib_sample_pd.to_csv(f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/{disease_name}/{celltype_name}/celltype_myofib_sample_refid.csv")

# COMMAND ----------

# list(celltype_myofib_sample.keys())

# COMMAND ----------

celltype_myofib.obs.head(20)

# COMMAND ----------

celltype_myofib.obs['sample'] == 'DS000011735-GSM4058950' #['subsample_indices']

# COMMAND ----------

celltype_myofib_sample['DS000011735-GSM4058950']

# COMMAND ----------

celltype_myofib_sample['DS000011735-GSM4058950']['subsample_indices']

# COMMAND ----------

celltype_sample_list

# COMMAND ----------

# celltype_myofib_sample['DS000011735-GSM4058950']['subsample_indices'] 

for enum, sample in enumerate(celltype_sample_list):
    print(f"enum: {enum} -- nsample: {sample}")
    celltype_sample = celltype_myofib[
                                    # celltype_myofib.obs["sample"] == "DS000011735-GSM4058950" # sample ref
                                    celltype_myofib.obs["sample"] == sample # sample ref 
                                    ].copy()

    print(celltype_sample)                                 

    ## we want to save sample to tmp/{path} as h5ad files first then copy to Volumes 

    

# COMMAND ----------

# celltype_myofib_sample['DS000011735-GSM4058950']['subsample_indices'] 

# COMMAND ----------

# DBTITLE 1,write celltype_samples to tmp--Vols
## to move to utils 
import os
import shutil


# Filter sample data to "Disease: IFP" | celltype: "myofibroblast cell" | sample_ref: "DS000011735-GSM4058950" 

disease_name = "IPF"
celltype_name = "myofibroblast cell".replace(' ','-')

# sample_refid = "DS000011735-GSM4058950" #"sample"
# sample_refid = sample ## input

for enum, sample_refid in enumerate(celltype_sample_list):
    print(f"enum: {enum} -- nsample: {sample_refid}")
    celltype_sample = celltype_myofib[
                                    # celltype_myofib.obs["sample"] == "DS000011735-GSM4058950" # sample ref
                                    celltype_myofib.obs["sample"] == sample_refid # sample ref 
                                    ].copy()

    print(celltype_sample)      

    tmp_path = f"/tmp/adams_etal_2020/{disease_name}_{celltype_name}_tmp/{disease_name}_{celltype_name}_{sample_refid}.h5ad"
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

    #celltype_sample -- derived from celltype_myofib 
    celltype_sample.write_h5ad(tmp_path, compression=None)

    # Define the source and destination paths
    source_path = tmp_path 
    destination_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/{disease_name}/{celltype_name}/{disease_name}_{celltype_name}_{sample_refid}.h5ad"

    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    # Copy the file
    shutil.copy(source_path, destination_path)

    print(f"File copied from {source_path} to {destination_path}")

# COMMAND ----------

# celltype_myofib_sample_pd.write_csv(f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/{disease_name}/{celltype_name}/celltype_myofib_sample_pd.csv")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### previous testing  saving h5ad first to tmp then Vols 

# COMMAND ----------

# subsample

# COMMAND ----------

# utils X_vals
# X_vals

# COMMAND ----------

celltype_sample.obs

# COMMAND ----------

# DBTITLE 1,update query_cell to use multiple subsamples for this version
query_cell = celltype_sample #subsample ## test multiple rows

## extract subsample query (1d array or list)
X_vals: sparse.csr_matrix = query_cell.X
X_vals

# COMMAND ----------

query_cell.obs

# COMMAND ----------

# DBTITLE 1,check update
query_cell, X_vals

# COMMAND ----------

# DBTITLE 1,NOTES
### Filter sample data to "Disease: IFP" | celltype: "myofibroblast cell" | sample_ref: "DS000011735-GSM4058950" 

# h5ad: filename =f"{disease_name}_{celltype_name}_{sample_ref}.h5ad"

# X_vals: sparse.csr_matrix = query_cell.X
# subsample index refs 

# COMMAND ----------

# query_cell.write_h5ad(filename: 'PathLike[str] | str | None'=None, compression: "Literal['gzip', 'lzf'] | None"=None, compression_opts: 'int | Any'=None, as_dense: 'Sequence[str]'=(), *, convert_strings_to_categoricals: 'bool'=True)

# COMMAND ----------

len(query_cell.X.toarray().tolist())

# COMMAND ----------

## Check data size first
# import scipy

# print(f"Data shape: {query_cell.shape}")
# print(f"Memory usage estimate: {query_cell.X.nnz} B") # / 1e9:.2f} GB")
# print(f"Data type: {query_cell.X.dtype}")
# print(f"Is sparse: {scipy.sparse.issparse(query_cell.X)}")

# COMMAND ----------

# DBTITLE 1,write a subsample as zarr to UC vols
## to move to utils 
import os

# Filter sample data to "Disease: IFP" | celltype: "myofibroblast cell" | sample_ref: "DS000011735-GSM4058950" 

disease_name = "IPF"
celltype_name = "myofibroblast cell".replace(' ','-')
celltype_id = "CL:0000186"
sample_refid = "DS000011735-GSM4058950"

tmp_path = f"/tmp/adams_etal_2020/test_saving_samples/{disease_name}_{celltype_name}/{disease_name}_{celltype_name}_{sample_refid}.h5ad"

os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

query_cell.write_h5ad(tmp_path, compression=None)


# path2save_zarr = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/GSE136831_subsample/{disease_name}_{celltype_name}_{sample_refid}.zarr"

# Ensure the directory exists
# os.makedirs(os.path.dirname(path2save_h5ad), exist_ok=True)
# os.makedirs(os.path.dirname(path2save_zarr), exist_ok=True)

# print("path2save_zarr :", path2save_zarr)

# # Method 1: Use zarr backend (often more stable)
# try:
#     print("Writing cell_query subsample to UC Vols as Zarr .... ")
#     # query_cell.write_zarr("/Volumes/genesis_workbench/dev_mmt_core_test/scimilarity/data/adams_etal_2020/GSE136831_subsample/IFP_myofibroblast-cell_DS000011735-GSM4058950.zarr")
#     query_cell.write_zarr(path2save_zarr)
#     print("cell_query subsample write to UC Vols as Zarr: successful")
# except Exception as e:
#     print(f"cell_query subsample write to UC Vols as Zarr: failed: {e}")


### NOT working on DBX --> kernel crashes due to 
# The SIGSEGV (Segmentation fault) indicates a low-level memory access violation. This is happening in Databricks, and given your tiny dataset (3 × 28231), it's likely a library compatibility issue rather than memory problems.

# path2save_h5ad = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/GSE136831_subsample/{disease_name}_{celltype_name}_{sample_refid}.h5ad"

# query_cell.write_h5ad(path2save_h5ad, compression=None)

# COMMAND ----------

## test writing h5ad to local dbfs then Vols first 

# COMMAND ----------

!ls -lah /tmp/adams_etal_2020/test_saving_samples/GSE136831_subsample/

# COMMAND ----------

import shutil

# Define the source and destination paths
source_path = tmp_path #"/tmp/adams_etal_2020/GSE136831_subsample/{disease_name}_{celltype_name}_{sample_refid}.h5ad"
destination_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/test_saving_samples/{disease_name}_{celltype_name}/{disease_name}_{celltype_name}_{sample_refid}.h5ad"

# Ensure the destination directory exists
os.makedirs(os.path.dirname(destination_path), exist_ok=True)

# Copy the file
shutil.copy(source_path, destination_path)

print(f"File copied from {source_path} to {destination_path}")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,updated  | (from utils)
query_cell.obs

# COMMAND ----------

# DBTITLE 1,Read back file
## move to utils / UI inference 
import scanpy as sc
# import anndata as ad

# Read zarr file with full path
# cell_sample = ad.read_zarr(path2save_zarr)

cell_sample = sc.read_h5ad(destination_path)
print(cell_sample)

# COMMAND ----------

# DBTITLE 1,check
query_cell.X, cell_sample.X

# COMMAND ----------

query_cell.X.toarray(), cell_sample.X.toarray()


# COMMAND ----------

# DBTITLE 1,X_vals | sparse matrix
# X_vals: sparse.csr_matrix = query_cell.X
X_vals: sparse.csr_matrix = cell_sample.X
X_vals

# Assuming X_vals is a sparse matrix
X_vals_dense = X_vals.toarray()


# COMMAND ----------

cell_sample.obs

# COMMAND ----------

cell_sample.obs.index

# COMMAND ----------

cell_sample_obs_json = cell_sample.obs.to_json(orient='split')
cell_sample_obs_json

# COMMAND ----------

# DBTITLE 1,cell_sample_obs
pd.read_json(cell_sample_obs_json, orient='split')

# COMMAND ----------

# DBTITLE 1,cell_subsample
## preserve the index from original data
cell_subsample_pdf = pd.DataFrame([{'cell_subsample': row} for row in X_vals_dense ], index=cell_sample.obs.index)
cell_subsample_pdf
# cell_subsample_json = cell_subsample_pdf.to_json(orient='split')
# cell_subsample_json

# COMMAND ----------

cell_subsample_pdf.to_json(orient='split')

# COMMAND ----------

# DBTITLE 1,Specify model_input
# existing
## model_input = pd.DataFrame([{'subsample_query_array': X_vals.toarray().astype(np.float64)[0]}])
# model_input = pd.DataFrame([{'subsample_query_array': X_vals.toarray()[0].tolist() }]) ## simpler


model_input = pd.DataFrame([{"celltype_sample": cell_subsample_pdf.to_json(orient='split'), 
               "celltype_sample_obs": cell_sample.obs.to_json(orient='split')
              }
              ])


# model_input_tmp = pd.DataFrame([{"celltype_sample": cell_subsample_pdf.to_json(orient='split'), 
#                                 "celltype_sample_obs": cell_sample.obs.to_json(orient='split')
#                                 }
#                                 ])

# model_input = pd.concat([model_input_tmp, model_input_tmp], axis=0)

model_input

# COMMAND ----------

pd.concat([cell_subsample_pdf, cell_sample.obs], axis=1)

# COMMAND ----------

X_vals.toarray(), X_vals_dense

# COMMAND ----------

# DBTITLE 1,XX Specify model_input
# # Create a DataFrame containing the embeddings

# # X_vals.dtype # float64 
# # X_vals.shape # (1, 28231) -- 1d array

# # X_vals: sparse.csr_matrix = query_cell.X
# X_vals: sparse.csr_matrix = cell_sample.X
# X_vals

# # existing
# ## model_input = pd.DataFrame([{'subsample_query_array': X_vals.toarray().astype(np.float64)[0]}])
# # model_input = pd.DataFrame([{'subsample_query_array': X_vals.toarray()[0].tolist() }]) ## simpler


# # Assuming X_vals is a sparse matrix
# X_vals_dense = X_vals.toarray()

# # Create a DataFrame where each row in X_vals_dense is a separate entry in the DataFrame
# # model_input = pd.DataFrame([{'subsample_query_array': row for row in X_vals_dense}], index=cell_sample.obs.index)


# model_input = pd.DataFrame([{"celltype_sample": subsample_query_pdf.to_json(orient='split'), 
#                "celltype_sample_obs": cell_sample.obs.to_json(orient='split')
#               }
#               ])

# # Convert cell_sample.obs to a DataFrame
# # cell_sample_obs_df = pd.DataFrame(cell_sample.obs.to_dict(orient='list'), index=cell_sample.obs.index)

# # Concatenate the two DataFrames along the columns
# # model_input = pd.concat([model_input, cell_sample_obs_df], axis=1)

# # Add the index as a column to preserve it
# # model_input['index'] = model_input.index

# # Display the DataFrame
# model_input

# COMMAND ----------

pd.read_json(model_input["celltype_sample"].iloc[0], orient='split')

# COMMAND ----------

pd.read_json(model_input[["celltype_sample_obs"]].iloc[0][0], orient='split')

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Test model_input
# Call the predict method

embedding_output = model.predict(temp_context, model_input)

embedding_output

# COMMAND ----------

pd.read_json(embedding_output[['embedding']].iloc[0][0])

# COMMAND ----------

pd.read_json(embedding_output[['celltype_sample_obs']].iloc[0][0])

# COMMAND ----------

# embedding_output.to_dict()

# COMMAND ----------

workflow: h5ad --> embeddings --> delta 

# COMMAND ----------

# DBTITLE 1,THIS output
# pd.DataFrame(embedding_output)

pd.DataFrame(embedding_output.to_dict()) #--> output Delta | Catalog/schema/tablename

# COMMAND ----------

# embedding_output.to_dict()

# COMMAND ----------

## Assuming embedding_output.embedding is a 2D array where each row is an embedding
# results = []
# for embedding in embedding_output.embedding:
#     result = cq.search_nearest(embedding, k=10)
#     results.append(result)

# # Display the results
# results

# COMMAND ----------

# DBTITLE 1,test embedding_output with downstream KNN
# 1 row of input embeddings to search_nearest
# cq.search_nearest(embedding_output.embedding[0], k=10) 

#test 1row or multiple 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define MLflow Signature with local Model + Context

# COMMAND ----------

# DBTITLE 1,include subsample index/extra info. | disease/type / cell type / water
model_input

# COMMAND ----------

# DBTITLE 1,Define MLflow Signature
from mlflow.models import infer_signature

# Define a concrete example input as a Pandas DataFrame
example_input = model_input

# Ensure the example output is in a serializable format
example_output = embedding_output # from model.predict(temp_context, model_input)
# example_output = embedding_output.to_dict()
# example_output = embedding_output.to_json(orient='split')
# example_output = pd.read_json(embedding_output, orient='split')

# Infer the model signature
signature = infer_signature(
    model_input=example_input,
    model_output=example_output,    
)

# COMMAND ----------

# example_input, example_output

# COMMAND ----------

signature

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



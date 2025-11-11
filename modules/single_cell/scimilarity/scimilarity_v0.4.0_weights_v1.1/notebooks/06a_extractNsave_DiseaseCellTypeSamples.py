# Databricks notebook source
# MAGIC %md
# MAGIC ### Extract `Disease-CellType-Samples` 
# MAGIC #### save as `.h5ad` to UC Volumes (via `/tmp/{path}`)
# MAGIC
# MAGIC Extract CellType_samples' subsamples as `AnnData` and save as `h5ad` files to use with testing served model endpoints and/or bulk embedding processing for:     
# MAGIC - `DiseaseType:IPF` 
# MAGIC - `CellType:myofibroblast cell` 
# MAGIC - `sample_refid` 
# MAGIC
# MAGIC Run `./utils` and replace `catalog`, `schema` etc. with your own 

# COMMAND ----------

# DBTITLE 1,run utils and override catalog/schema variables
# MAGIC %run ./utils $catalog="genesis_workbench" $schema="dev_mmt_core_demo" 

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# DBTITLE 1,saved adams data sample path
# path of adams file 
# adams_data = sc.read_h5ad('/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/GSE136831_subsample.h5ad')

# COMMAND ----------

# DBTITLE 1,string_vars from utils
# disease_name = "IPF"
# celltype_name = "myofibroblast cell" 
# sample_refid = "DS000011735-GSM4058950"
# subsample_refid = "123942"

# COMMAND ----------

# DBTITLE 1,observations in sample celltype data
# adams
# AnnData object with n_obs × n_vars = 50000 × 44942
#     obs: 'celltype_raw', 'celltype_id', 'sample', 'study', 'n_counts', 'n_genes', 'celltype_name', 'doublet_score', 'pred_dbl', 'Disease'
#     var: 'row_names', 'ID', 'mt'
#     layers: 'counts'

# adams.obs

# adams.obs.Disease.unique()
# adams.obs['Disease'].value_counts()

# IPF        23803
# healthy    15149
# COPD       11048

# adams.obs.celltype_name.value_counts() #33

# [x / 32 for x in [23803, 15149, 11048]]
# [743.84375, 345.25, 473.40625]

# IPF: Idiopathic Pulmonary Fibrosis -- A chronic, progressive lung disease characterized by the scarring of lung tissue, making it difficult to breathe. The term "idiopathic" means the cause of the scarring is unknown. 

# Chronic obstructive pulmonary disease (COPD) -- A type of progressive lung disease characterized by chronic respiratory symptoms and airflow limitation.

# diseasetype_ipf.obs['sample'].value_counts().sum() #23803
# len(diseasetype_ipf.obs['sample'].value_counts()) #32

# adams.obs.celltype_name.value_counts() #33

# COMMAND ----------

# DBTITLE 1,We will use existing celltype_sample : myofibroblast for bulk embedding batch inference
# celltype_myofib

# celltype_myofib.obs.celltype_id.unique() | Categories (1, object): ['CL:0000186']

# COMMAND ----------

# DBTITLE 1,check | celltype_myofib.obs
celltype_myofib.obs

# COMMAND ----------

# DBTITLE 1,celltype_sample_list
# "subsample" indices are unique
# celltype_myofib.obs.index.unique()

celltype_sample_list = sorted(celltype_myofib.obs['sample'].unique())
# len(celltype_sample_list) # 29

celltype_myofib.obs['sample'].value_counts()

# COMMAND ----------

# DBTITLE 1,ref -- file -subsets- structure
## celltype_myofib = diseasetype_ipf[
#                                   diseasetype_ipf.obs["celltype_name"] == "myofibroblast cell"
#                                  ].copy()

# ## Extract list for sample_ref
# celltype_sample = celltype_myofib[
#                                   celltype_myofib.obs["sample"] == "{DS000011735-GSM4058950 -- update subsample refid}" # sample ref 
#                                  ].copy()

## extract specific index in celltype_subsample 
# celltype_subsample = celltype_sample[celltype_sample.obs.index == "123942"]

# COMMAND ----------

# DBTITLE 1,track refid of subsamples within celltype_sample
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

# DBTITLE 1,save to UC Vols
import os

output_dir = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/{disease_name}/{celltype_name.replace(' ','-')}"
os.makedirs(output_dir, exist_ok=True)

celltype_myofib_sample_pd.to_csv(
    f"{output_dir}/celltype_myofib_sample_refid.csv"
)

# COMMAND ----------

# DBTITLE 1,Extract celltype_samples to-tmp-then-Vols
import os
import shutil


## REF string_variables for Disease - celltype - sample 
# disease_name = "IPF"
celltype_name2use = celltype_name.replace(' ','-')
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

    ### Write each sample with corresponding subsample(s) within celltype_myofib to tmp_path
    #celltype_sample -- derived from celltype_myofib 
    celltype_sample.write_h5ad(tmp_path, compression=None)

    # Define the source and destination paths
    source_path = tmp_path 
    destination_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/{disease_name}/{celltype_name2use}/{disease_name}_{celltype_name2use}_{sample_refid}.h5ad"

    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    # Copy the file
    shutil.copy(source_path, destination_path)

    print(f"File copied from {source_path} to {destination_path}")

# COMMAND ----------

# DBTITLE 1,Quick test to read in written h5ad files
import scanpy as sc

# Read h5ad file with full path
print(destination_path)

cell_sample = sc.read_h5ad(destination_path)
print(cell_sample)

cell_sample.obs

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### For visualization of all embeddings derived from full sample data 
# MAGIC To save time, and for tutorial viz: we use 
# MAGIC - `cell_query.get_embeddings` to pre-process `adams0`, which is cell-order aligned, log-normed, and scimilarity embeddings applied to the adams et al dataset sample for displaying with umap.    
# MAGIC - `cell_query.cell_metadata` to pre-extract `scimilarity_cq_ref_metadata` and save as UC table for faster read and filtering to celltype for deriving `calculate_disease_proportions`
# MAGIC       
# MAGIC <br>   
# MAGIC
# MAGIC ----       
# MAGIC _NOTE_:    
# MAGIC The functions `scimilarity.cell_query.get_embeddings` and `scimilarity.cell_embedding.get_embeddings` both relate to generating embeddings (vector representations) of cells, but they are used in different contexts within the SCimilarity framework:
# MAGIC
# MAGIC <!-- ##### `scimilarity.cell_query.get_embeddings`
# MAGIC
# MAGIC - **Purpose:** This method is typically used as part of the cell query workflow. It allows users to transform a query cell or set of query cells (input gene expression matrix) into the *embedding space* defined by the SCimilarity model.
# MAGIC - **Use Case:** When you want to search for cells similar to a particular cell or profile across a large reference atlas, you would use this function to embed your query profile(s) first. The resulting embeddings represent the query in the same space as the reference and can be used for rapid nearest neighbor search or annotation tasks.
# MAGIC - **Works with:** Query cells—profiles you want to analyze or compare against the reference.
# MAGIC
# MAGIC ##### `scimilarity.cell_embedding.get_embeddings`
# MAGIC
# MAGIC - **Purpose:** This is a more general, lower-level method applied to any set of cells. It generates embeddings directly from an input gene expression matrix, returning the positions of these cells in the high-dimensional SCimilarity space.
# MAGIC - **Use Case:** You can use this directly for embedding *bulk* cell matrices, either for new data you wish to explore/visualize, for building new references, or for any analytic task where you want the direct SCimilarity embedding of a dataset (not necessarily for immediate search).
# MAGIC - **Works with:** Any set of cells—reference datasets, new scRNA-seq data, and general embedding needs. -->
# MAGIC
# MAGIC ##### How They Typically Differ
# MAGIC
# MAGIC | Function                                  | Intended For                                                    | Input Data                    | Typical Workflow                 |
# MAGIC |--------------------------------------------|-----------------------------------------------------------------|-------------------------------|-----------------------------------|
# MAGIC | `cell_query.get_embeddings`                | Querying/searching for similar cells in the reference atlas     | Query cells (profiles to search)| Input → embed → search/annotate   |
# MAGIC | `cell_embedding.get_embeddings`            | General embedding generation (any set of cells, including query or reference)| Any cell expression matrix     | Input → embed (for analysis, visualization, etc.) |
# MAGIC
# MAGIC ##### Practical Example
# MAGIC
# MAGIC - If you have a **new cell** profile and want to find its nearest neighbors in the Human Cell Atlas using SCimilarity’s prebuilt search features, use **`cell_query.get_embeddings`**.
# MAGIC - If you want to **embed a large external dataset** (e.g., your own scRNA-seq experiment) for downstream analyses, use **`cell_embedding.get_embeddings`**.
# MAGIC
# MAGIC In summary, both functions produce embeddings using the SCimilarity model, but `cell_query.get_embeddings` is tailored for embedding queries before performing search/annotation, while `cell_embedding.get_embeddings` is the direct, lower-level embedding engine for any bulk data processing or integration step.[1][2]
# MAGIC
# MAGIC [1] https://genentech.github.io/scimilarity/
# MAGIC [2] https://github.com/Genentech/scimilarity/issues/24
# MAGIC <!-- [3] https://github.com/Genentech/scimilarity
# MAGIC [4] https://www.nature.com/articles/s41586-024-08411-y
# MAGIC [5] https://pmc.ncbi.nlm.nih.gov/articles/PMC11864978/
# MAGIC [6] https://scgpt.readthedocs.io/en/latest/tutorial_reference_mapping.html
# MAGIC [7] https://www.youtube.com/watch?v=16s3Pi1InPU
# MAGIC [8] https://www.biorxiv.org/content/10.1101/2023.07.18.549537.full
# MAGIC [9] https://pmc.ncbi.nlm.nih.gov/articles/PMC8421403/
# MAGIC [10] https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents
# MAGIC [11] https://www.nature.com/articles/s41592-025-02627-0
# MAGIC [12] https://www.nature.com/articles/s41586-024-08411-y_reference.pdf
# MAGIC [13] https://python.langchain.com/docs/how_to/example_selectors_similarity/
# MAGIC [14] https://academic.oup.com/nargab/article/6/1/lqae004/7591099
# MAGIC [15] https://pmc.ncbi.nlm.nih.gov/articles/PMC9915700/
# MAGIC [16] https://www.youtube.com/watch?v=DIxxz_DvqLA
# MAGIC [17] https://pmc.ncbi.nlm.nih.gov/articles/PMC11310084/
# MAGIC [18] https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1.full-text
# MAGIC [19] https://dintor.eurac.edu/doc/python/classcls_1_1FunctionalSimilarity_1_1CFunctionalSimilarityBase.html
# MAGIC [20] https://zenodo.org/records/8286452
# MAGIC [21] https://pmc.ncbi.nlm.nih.gov/articles/PMC10591141/
# MAGIC [22] https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
# MAGIC [23] https://www.nature.com/articles/s41467-021-25534-2
# MAGIC [24] https://www.humancellatlas.org/publications/
# MAGIC [25] https://www.nature.com/articles/s41588-024-01993-3 -->

# COMMAND ----------

# DBTITLE 1,Load CellQuery from SCimilarity
## For full data embedding processing -- a local CellQuery.get_embeddings(adams.X) is more efficient
from scimilarity import CellQuery
model_path = f'/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/model/model_v1.1/' 
cq = CellQuery(model_path)

# COMMAND ----------

# DBTITLE 1,get cell ref metadata -- save as delta
ref_metadata = cq.cell_metadata #memory usage: 4.5+ GB | 23381150 entries | ~6mins

# Convert ref_metadata to a Spark DataFrame if it is not already
ref_metadata_spark_df = spark.createDataFrame(ref_metadata)

# Define the Unity Catalog path
uc_path = f"{CATALOG}.{DB_SCHEMA}.scimilarity_cq_ref_metadata"

# Write the DataFrame to the Unity Catalog path | partitioning?
ref_metadata_spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(uc_path)

## faster read via delta to pandasDF
# ref_metadata = spark.read.table(uc_path).toPandas()
# ref_metadata.info()

# COMMAND ----------

# DBTITLE 1,save adams subsample with compression as h5ad
# sampledata_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/GSE136831_subsample.h5ad"

# ## READ sample czi dataset H5AD file + align + lognorm 
# adams = sc.read(sampledata_path) ## should already be pre-read in the utils_20250801

import os, shutil

tmp_path = f"/tmp/adams_etal_2020/adams.h5ad"
os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

### Write the H5AD file first to tmp_path
# adams0.write_h5ad(tmp_path, compression=None) ## >2gigs
adams.write_h5ad(tmp_path, compression='gzip') ## ~600mb

# Define the source and destination paths
source_path = tmp_path 
destination_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/adams.h5ad"

# Ensure the destination directory exists
os.makedirs(os.path.dirname(destination_path), exist_ok=True)

# Copy the file
shutil.copy(source_path, destination_path)

print(f"File copied from {source_path} to {destination_path}")

# COMMAND ----------

# DBTITLE 1,Derive alignment + log-norm + embeddings 4 nearest neighbors
# sampledata_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/GSE136831_subsample.h5ad"

# adams0 = sc.read(sampledata_path)

## preprocessing performed per https://genentech.github.io/scimilarity/notebooks/cell_search_tutorial_1.html
adams0 = adams.copy() ## from utils_20250801
adams0 = align_dataset(adams0, cq.gene_order)
adams0 = lognorm_counts(adams0)

adams0.obsm["X_scimilarity"] = cq.get_embeddings(adams0.X) ## more efficient then endpoint (cq_version of get_embeddings was not wrapped as mlflow custom pyfunc)

sc.pp.neighbors(adams0, use_rep="X_scimilarity")
sc.tl.umap(adams0)

# COMMAND ----------

# DBTITLE 1,write adams0 to tmp then copy to UC vols
import os, shutil

tmp_path = f"/tmp/adams_etal_2020/adams0_alignedNlognormed_Xscim_umap.h5ad"
os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

### Write the H5AD file first to tmp_path
# adams0.write_h5ad(tmp_path, compression=None) ## >2gigs
adams0.write_h5ad(tmp_path, compression='gzip') ## ~600mb

# Define the source and destination paths
source_path = tmp_path 
destination_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/adams0_alignedNlognormed_Xscim_umap.h5ad"

# Ensure the destination directory exists
os.makedirs(os.path.dirname(destination_path), exist_ok=True)

# Copy the file
shutil.copy(source_path, destination_path)

print(f"File copied from {source_path} to {destination_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# DBTITLE 1,list files
display(dbutils.fs.ls(
        f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/")
       )

display(dbutils.fs.ls(f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/")
        )

display(dbutils.fs.ls(f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/IPF/myofibroblast-cell/")
        )

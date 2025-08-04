# Databricks notebook source
# MAGIC %md
# MAGIC ### Extract `Disease-CellType-Samples` save as `.h5ad` to UC Volumes (via `/tmp/{path}`)
# MAGIC
# MAGIC Extract CellType_samples' subsamples as `AnnData` and save as `h5ad` files for bulk embedding processing for `DiseaseType:IPF` - `CellType:myofibroblast cell` - `sample_refid` 

# COMMAND ----------

# DBTITLE 1,install/load dependencies | # ~5mins (including initial data processing)
# MAGIC %run ./utils_20250801

# COMMAND ----------

CATALOG, DB_SCHEMA, MODEL_FAMILY, MODEL_NAME, EXPERIMENT_NAME

# COMMAND ----------

# DBTITLE 1,saved adams data sample path
# path of adams file 
# /Volumes/genesis_workbench/dev_mmt_core_test/scimilarity/data/adams_etal_2020/GSE136831_subsample.h5ad
# adams_data = sc.read_h5ad('/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/GSE136831_subsample.h5ad')

# COMMAND ----------

# DBTITLE 1,string_varis from utils
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

# IPF stands for Idiopathic Pulmonary Fibrosis. It is a chronic, progressive lung disease characterized by the scarring of lung tissue, making it difficult to breathe. The term "idiopathic" means the cause of the scarring is unknown. 

# Chronic obstructive pulmonary disease (COPD) is a type of progressive lung disease characterized by chronic respiratory symptoms and airflow limitation.

# diseasetype_ipf.obs['sample'].value_counts().sum() #23803
# len(diseasetype_ipf.obs['sample'].value_counts()) #32

# adams.obs.celltype_name.value_counts() #33

# COMMAND ----------

# DBTITLE 1,We will use existing celltype_sample : myofibroblast for bulk embedding batch inference
# celltype_myofib

# celltype_myofib.obs.celltype_id.unique() | Categories (1, object): ['CL:0000186']

# COMMAND ----------

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
celltype_myofib_sample_pd.to_csv(f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/{disease_name}/{celltype_name.replace(' ','-')}/celltype_myofib_sample_refid.csv")

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

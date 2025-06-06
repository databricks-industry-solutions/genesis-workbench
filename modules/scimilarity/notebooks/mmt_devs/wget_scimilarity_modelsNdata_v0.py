# Databricks notebook source
# DBTITLE 1,get model & data files
## using serverless compute for this

## REF https://genentech.github.io/scimilarity/install.html

# COMMAND ----------

# Downloading the pretrained models
# You can download the following pretrained models for use with SCimilarity from Zenodo: https://zenodo.org/records/10685499

# COMMAND ----------

# Create the target directory if it doesn't exist
!mkdir -p /Volumes/mmt/genesiswb/scimilarity/downloads

# COMMAND ----------

!ls -lah /Volumes/mmt/genesiswb/scimilarity/downloads

# COMMAND ----------

# DBTITLE 1,download model
# MAGIC %sh
# MAGIC wget -O /Volumes/mmt/genesiswb/scimilarity/downloads/model_v1.1.tar.gz http://zenodo.org/records/10685499/files/model_v1.1.tar.gz?download=1 --verbose

# COMMAND ----------

!ls -lah /Volumes/mmt/genesiswb/scimilarity/downloads

# COMMAND ----------



# COMMAND ----------

# Create the target directory if it doesn't exist
!mkdir -p /Volumes/mmt/genesiswb/scimilarity/model

# COMMAND ----------

!ls -lah -p /Volumes/mmt/genesiswb/scimilarity/model

# COMMAND ----------

# MAGIC %sh
# MAGIC chmod u+rx /Volumes/mmt/genesiswb/scimilarity/downloads/model_v1.1.tar.gz

# COMMAND ----------

# DBTITLE 1,untargzip to model folder
# MAGIC %sh
# MAGIC tar --no-same-owner -xzvf /Volumes/mmt/genesiswb/scimilarity/downloads/model_v1.1.tar.gz -C /Volumes/mmt/genesiswb/scimilarity/model --verbose

# COMMAND ----------

# %sh
# tar --no-same-owner -xzvf /Volumes/mmt/genesiswb/scimilarity/downloads/model_v1.1.tar.gz -C /Volumes/mmt/genesiswb/scimilarity/model --wildcards '*/gene_order.tsv'

# COMMAND ----------

!ls -lah -p /Volumes/mmt/genesiswb/scimilarity/model/*

# COMMAND ----------



# COMMAND ----------

# Query data. We will use Adams et al., 2020 healthy and IPF lung scRNA-seq data. Download tutorial data.
# https://zenodo.org/records/13685881

# COMMAND ----------

# Create the target directory if it doesn't exist
!mkdir -p /Volumes/mmt/genesiswb/scimilarity/data/adams_etal_2020

# COMMAND ----------

!ls -lah /Volumes/mmt/genesiswb/scimilarity/data/adams_etal_2020

# COMMAND ----------

# DBTITLE 1,get sample data
# MAGIC %sh
# MAGIC wget -O /Volumes/mmt/genesiswb/scimilarity/data/adams_etal_2020/GSE136831_subsample.h5ad https://zenodo.org/records/13685881/files/GSE136831_subsample.h5ad?download=1 --verbose

# COMMAND ----------

!ls -lah /Volumes/mmt/genesiswb/scimilarity/data/adams_etal_2020

# COMMAND ----------



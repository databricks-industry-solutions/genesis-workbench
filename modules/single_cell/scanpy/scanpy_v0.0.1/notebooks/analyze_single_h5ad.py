# Databricks notebook source
# MAGIC %md
# MAGIC # Scanpy Analysis in a Databricks Notebook
# MAGIC  - load in the data from a **Unity Catalog Volume**
# MAGIC  - analyze the data using **scanpy**
# MAGIC  - track results using **mlflow**

# COMMAND ----------

# MAGIC %pip install 'numpy<2' scanpy==1.11.4 anndata igraph
# MAGIC %restart_python

# COMMAND ----------

import scanpy as sc
import anndata as ad
import os
import pandas as pd
import copy
import tempfile
import time

# COMMAND ----------

dbutils.widgets.text("data_path", "", "Data Path")
dbutils.widgets.text("mlflow_experiment", "", "MLflow Experiment")
dbutils.widgets.text("mlflow_run_name", "", "MLflow Run Name")
dbutils.widgets.text("gene_name_column", "gene_name", "Gene Name Column")
dbutils.widgets.text("min_genes", "200", "Min Genes per Cell")
dbutils.widgets.text("min_cells", "3", "Min Cells per Gene")
dbutils.widgets.text("pct_counts_mt", "5", "Max % Mitochondrial Counts")
dbutils.widgets.text("n_genes_by_counts", "2500", "Max Genes by Counts")
dbutils.widgets.text("target_sum", "10000", "Target Sum for Normalization")
dbutils.widgets.text("n_top_genes", "500", "Number of Highly Variable Genes")
dbutils.widgets.text("n_pcs", "50", "Number of Principal Components")
dbutils.widgets.text("leiden_resolution", "0.2", "Leiden Resolution")

# COMMAND ----------

parameters = {
  'data_path':dbutils.widgets.get("data_path"),
  'mlflow_experiment':dbutils.widgets.get("mlflow_experiment"),
  'mlflow_run_name':dbutils.widgets.get("mlflow_experiment"),
  'gene_name_column':dbutils.widgets.get("gene_name_column"),
  'min_genes': int(dbutils.widgets.get("min_genes")),
  'min_cells': int(dbutils.widgets.get("min_cells")),
  'pct_counts_mt': float(dbutils.widgets.get("pct_counts_mt")),
  'n_genes_by_counts': int(dbutils.widgets.get("n_genes_by_counts")),
  'target_sum': int(float(dbutils.widgets.get("target_sum"))),
  'n_top_genes': int(dbutils.widgets.get("n_top_genes")),
  'n_pcs': int(dbutils.widgets.get("n_pcs")),
  'leiden_resolution': float(dbutils.widgets.get("leiden_resolution")),
}

metrics = {}

# COMMAND ----------

parameters

# COMMAND ----------

# MAGIC %md
# MAGIC ### make a directory to save some results
# MAGIC  - we'll later do some logging of results with **mlflow** 

# COMMAND ----------

# we'll save some things to disk to move to our experiment run when we log our findings
tmpdir = tempfile.TemporaryDirectory()
sc.settings.figdir = tmpdir.name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load in an h5ad file from Unity Catalog Volume
# MAGIC  - The Volume is blob storage, but you can interact with it like a unix system

# COMMAND ----------

t0 = time.time()
adata = sc.read_h5ad(
    parameters['data_path'],
)
adata.obs_names_make_unique()

# here we use adata.raw if it exists as is assumeed to be unprocessed raw data
if adata.raw is not None:
    adata = adata.raw.to_adata()

GENE_NAME_COLUMN = parameters['gene_name_column']
adata.var = adata.var.reset_index()
adata.var[GENE_NAME_COLUMN] = adata.var[GENE_NAME_COLUMN].astype(str) 
adata.var['Gene name'] = adata.var[GENE_NAME_COLUMN]
adata.var = adata.var.set_index(GENE_NAME_COLUMN)

# COMMAND ----------

# mitochondrial genes, "MT-" for human, "Mt-" for mouse
adata.var["mt"] = adata.var['Gene name'].str.startswith("MT-",na=False)
# ribosomal genes
adata.var["ribo"] = adata.var['Gene name'].str.startswith(("RPS", "RPL"),na=False)
# hemoglobin genes
adata.var["hb"] = adata.var['Gene name'].str.contains("^HB[^(P)]",na=False)

sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)

current_cells = adata.shape[0]

# COMMAND ----------

# depending on input data and requirements may which to filter
sc.pp.filter_cells(adata, min_genes=parameters['min_genes'])
sc.pp.filter_genes(adata, min_cells=parameters['min_cells'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Let's keep track of how many cells we retain on filtering

# COMMAND ----------

metrics['filter_simple_retention'] = 100.0*adata.shape[0]/current_cells
current_cells = adata.shape[0]

# COMMAND ----------

# MAGIC %md
# MAGIC #### generate useful QC plots and save to disk

# COMMAND ----------

sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt", save="counts_plot_prefilter.png")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Perform additional filtration of cells

# COMMAND ----------

# filtering cells further
adata = adata[adata.obs.n_genes_by_counts < parameters['n_genes_by_counts'], :] # could use scrublet etc for doublet removal as desired
adata = adata[adata.obs.pct_counts_mt < parameters['pct_counts_mt'], :].copy() # or other threshold to remove dead/dying cells

# COMMAND ----------

metrics['filter_mtgenes_retention'] =  100.0*adata.shape[0]/current_cells
current_cells = adata.shape[0]

# COMMAND ----------

sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt", save="counts_plot_post_filter.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalize the data, identify variable genes, dimension reduction

# COMMAND ----------

sc.pp.normalize_total(adata,target_sum=parameters['target_sum'])
sc.pp.log1p(adata)

# COMMAND ----------

sc.pp.highly_variable_genes(adata, n_top_genes=parameters['n_top_genes'])

# COMMAND ----------

sc.pl.highly_variable_genes(adata, save="hvg.png")

# COMMAND ----------

# MAGIC %md
# MAGIC #### PCA

# COMMAND ----------

sc.tl.pca(adata)

# COMMAND ----------

sc.pl.pca(
    adata,
    color=["pct_counts_mt", "pct_counts_mt"],
    dimensions=[(0, 1), (1, 2)],
    ncols=2,
    size=2,
    save='pca.png'
)

# COMMAND ----------

# optionally add PCA coords into cell (var) table
for i in range(adata._obsm['X_pca'].shape[1]):
    adata.obs['PCA_'+str(i)] = adata._obsm['X_pca'][:,i]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Do UMAP

# COMMAND ----------

sc.pp.neighbors(adata)

# COMMAND ----------

sc.tl.leiden(
    adata,
    resolution=parameters['leiden_resolution'],
    random_state=0,
    flavor="igraph",
    n_iterations=2,
    directed=False,
)

# COMMAND ----------

sc.tl.umap(adata)

# COMMAND ----------

sc.pl.umap(
    adata,
    color="leiden",
    size=2,
    save='umap_cluster.png'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mlflow to track runs with parameters
# MAGIC
# MAGIC  - save processed runs as adata to mlflow, also with parameters, metrics. Can place these in adata.uns also.
# MAGIC  - But keeping track of experiments with varying parameters can be useful for later review
# MAGIC  - **mlflow** often used in both classic ML and agentic AI offers some features that can be useful `here` 

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# optionally set a directory for experiment (if do not gets tied to this notebook, sometimes convenient too)
mlflow.set_experiment(parameters['mlflow_experiment'])

# save adata and our figures to disk
adata.write_h5ad(tmpdir.name+"/adata_output.h5ad")

t1 = time.time()
total_time = t1-t0
metrics = {'total_time': total_time}

if parameters['mlflow_run_name'] != '':
  run_name = None
else:
  run_name = parameters['mlflow_run_name']


with mlflow.start_run(run_name=run_name) as run:
  mlflow.log_metrics(metrics)
  mlflow.log_params(parameters)
  mlflow.log_artifacts(tmpdir.name)

# COMMAND ----------


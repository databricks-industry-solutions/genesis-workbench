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
dbutils.widgets.text("user_email", "", "User Email")
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
  'user_email':dbutils.widgets.get("user_email"),
  'mlflow_experiment':dbutils.widgets.get("mlflow_experiment"),
  'mlflow_run_name':dbutils.widgets.get("mlflow_run_name"),
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
metrics['total_cells_starting'] = float(current_cells)

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

# optionally add PCA coords into cell (obs) table
for i in range(4): #adata._obsm['X_pca'].shape[1]):
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

for i in range(2):
    adata.obs['UMAP_'+str(i)] = adata._obsm['X_umap'][:,i]

# COMMAND ----------

sc.pl.umap(
    adata,
    color="leiden",
    size=2,
    save='umap_cluster.png'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find marker genes for each cluster

# COMMAND ----------

# Run differential expression analysis
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")

# COMMAND ----------

# Visualize top markers
sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False, save='_marker_genes.png')

# COMMAND ----------

# Extract top N marker genes per cluster
n_markers_per_cluster = 10  # Adjust as needed

# Get all unique marker genes across all clusters
marker_genes = set()
n_clusters = len(adata.obs['leiden'].unique())

for i in range(n_clusters):
    cluster_markers = adata.uns['rank_genes_groups']['names'][str(i)][:n_markers_per_cluster]
    marker_genes.update(cluster_markers)

marker_genes = list(marker_genes)
print(f"Total unique marker genes: {len(marker_genes)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create marker gene dataframe for logging

# COMMAND ----------

# Create a DataFrame with top markers per cluster (for visualization)
marker_df = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(n_markers_per_cluster)
marker_df.to_csv(tmpdir.name + "/top_markers_per_cluster.csv", index=False)

# Also save full marker gene list
pd.DataFrame({'marker_genes': marker_genes}).to_csv(
    tmpdir.name + "/top_marker_genes.csv", 
    index=False
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Subsample the data to a maximum number of cells

# COMMAND ----------
import scipy
import numpy as np

adata_markers = adata[:, marker_genes].copy()

max_cells = 10000
if adata_markers.shape[0] > max_cells:
    # Get cluster counts and proportions
    cluster_counts = adata_markers.obs['leiden'].value_counts().sort_index()
    total_cells = cluster_counts.sum()
    metrics['total_cells_before_subsample'] = float(total_cells)
    
    # Calculate target cells per cluster (proportional)
    target_per_cluster = {}
    for cluster_id, count in cluster_counts.items():
        proportion = count / total_cells
        target_per_cluster[cluster_id] = int(np.round(proportion * max_cells))
    
    # Adjust to ensure exactly max_cells (handle rounding errors)
    current_total = sum(target_per_cluster.values())
    if current_total != max_cells:
        diff = max_cells - current_total
        # Adjust the largest cluster
        largest_cluster = cluster_counts.idxmax()
        target_per_cluster[largest_cluster] += diff
    
    # Perform stratified sampling
    np.random.seed(42)
    sampled_indices = []
    
    for cluster_id, target_n in target_per_cluster.items():
        # Get all cells in this cluster
        cluster_mask = adata_markers.obs['leiden'] == cluster_id
        cluster_cells = adata_markers.obs.index[cluster_mask]
        
        # Sample the target number (or all if cluster is smaller than target)
        n_to_sample = min(target_n, len(cluster_cells))
        sampled_cells = np.random.choice(cluster_cells, size=n_to_sample, replace=False)
        sampled_indices.extend(sampled_cells)
    
    # Create the subsampled object
    adata_markers = adata_markers[sampled_indices, :].copy()
    print(f"Subsampled to {len(adata_markers)} cells")
    print("\nCells per cluster (before → after):")
    for cluster_id in sorted(cluster_counts.index):
        before = cluster_counts[cluster_id]
        after = (adata_markers.obs['leiden'] == cluster_id).sum()
        pct_before = 100 * before / total_cells
        pct_after = 100 * after / len(adata_markers)
        print(f"  Cluster {cluster_id}: {before:>6} ({pct_before:>5.1f}%) → {after:>5} ({pct_after:>5.1f}%)")
        # add to metrics
        metrics[f"cluster_{cluster_id}_pct_before_subample"] = pct_before
        metrics[f"cluster_{cluster_id}_pct_after_subample"] = pct_after
        metrics[f"cluster_{cluster_id}_cells_before_subample"] = before
        metrics[f"cluster_{cluster_id}_cells_after_subample"] = after
else:
    # No subsampling needed
    metrics['total_cells_before_subsample'] = float(adata_markers.shape[0])

# Convert to a flat DataFrame
df_flat = adata_markers.obs.copy()

# Add marker gene expression as columns
# Convert sparse matrix to dense if needed
if scipy.sparse.issparse(adata_markers.X):
    expression_matrix = adata_markers.X.toarray()
else:
    expression_matrix = adata_markers.X

# Add each marker gene as a column
for i, gene in enumerate(marker_genes):
    df_flat[f"expr_{gene}"] = expression_matrix[:, i]

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

# Set the MLflow experiment
experiment = mlflow.set_experiment(parameters['mlflow_experiment'])

# Tag the experiment for Genesis Workbench search
mlflow.set_experiment_tags({
    "used_by_genesis_workbench": "yes"
})

# COMMAND ----------

# save adata and our figures to disk
adata.write_h5ad(tmpdir.name+"/adata_output.h5ad")
# save the flat dataframe to disk
df_flat.to_parquet(tmpdir.name + "/markers_flat.parquet")


t1 = time.time()
total_time = t1-t0
metrics['total_time'] = total_time

run_name = parameters['mlflow_run_name'] if parameters['mlflow_run_name'] else None

# Log to MLflow with proper tags for Genesis Workbench
with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id) as run:
    # Log metrics and params
    mlflow.log_metrics(metrics)
    mlflow.log_params(parameters)
    mlflow.log_artifacts(tmpdir.name)
    
    # Set required tags for Genesis Workbench search
    mlflow.set_tag("origin", "genesis_workbench")
    mlflow.set_tag("created_by", parameters['user_email'])
    mlflow.set_tag("processing_mode", "scanpy")

# COMMAND ----------


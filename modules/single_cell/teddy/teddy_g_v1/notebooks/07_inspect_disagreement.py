# Databricks notebook source
# MAGIC %md
# MAGIC ### TEDDY vs SCimilarity disagreement inspector
# MAGIC
# MAGIC One-off notebook to debug a specific Scanpy run where the two annotators
# MAGIC disagree on a cluster. Confirms (or refutes) the reference-bias hypothesis:
# MAGIC if our 250k-cell TEDDY reference (`disease != normal` filter, 5/10 chunks
# MAGIC done) has too few NK cells and too many plasma cells, KNN majority-vote
# MAGIC pulls disputed clusters toward plasma.
# MAGIC
# MAGIC Default target: run `scanpy_20260507_1950`. The disputed cluster is
# MAGIC the one SCimilarity calls NK and TEDDY calls plasma.

# COMMAND ----------

# MAGIC %pip install -q "numpy<2" mlflow==2.22.0 databricks-sdk==0.50.0 databricks-sql-connector==4.0.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "srijit_nair", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("run_name", "scanpy_20260507_1950", "MLflow run name")
dbutils.widgets.text("teddy_endpoint", "gwb_demo_teddy_endpoint", "TEDDY embedding endpoint")
dbutils.widgets.text("teddy_index", "teddy_cell_index", "TEDDY VS index name")
dbutils.widgets.text("gene_mapping_volume_path", "/Volumes/srijit_nair/genesis_workbench/teddy/gene_mapping.json", "HGNC→ENSG mapping file")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
run_name = dbutils.widgets.get("run_name")
endpoint = dbutils.widgets.get("teddy_endpoint")
index_name = f"{catalog}.{schema}.{dbutils.widgets.get('teddy_index')}"
mapping_path = dbutils.widgets.get("gene_mapping_volume_path")

print(f"Catalog/Schema: {catalog}.{schema}")
print(f"Run name: {run_name}")
print(f"Endpoint: {endpoint}")
print(f"VS index: {index_name}")

# COMMAND ----------

# DBTITLE 1,Resolve run by name + load markers + both annotations
import json
import tempfile
import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

client = MlflowClient()

# Scope to GWB experiments (workspace has 205+ experiments → exceeds the
# 100-experiment cap on SearchRuns). Same tag the app uses in
# `single_cell_analysis.search_singlecell_runs`.
exp_list = mlflow.search_experiments(
    filter_string="tags.used_by_genesis_workbench='yes'"
)
if not exp_list:
    raise RuntimeError("No GWB experiments found (tag `used_by_genesis_workbench='yes'`).")
experiment_ids = [e.experiment_id for e in exp_list][:100]
print(f"Scoped search to {len(experiment_ids)} GWB experiments")

hits_df = mlflow.search_runs(
    experiment_ids=experiment_ids,
    filter_string=f"tags.mlflow.runName = '{run_name}'",
    max_results=10,
    output_format="pandas",
)
if hits_df is None or len(hits_df) == 0:
    raise RuntimeError(
        f"No MLflow run named '{run_name}' was found in {len(experiment_ids)} "
        f"GWB experiments. Check the name in the GWB app's Search Past Runs view."
    )
run_id = hits_df.iloc[0]["run_id"]
experiment_id = hits_df.iloc[0].get("experiment_id", "?")
print(f"Resolved '{run_name}' → run_id={run_id}  experiment_id={experiment_id}")

# COMMAND ----------

import pandas as pd
import numpy as np

def _download_artifact_to_pandas(artifact_name, kind="parquet"):
    """Download a run artifact and return as a DataFrame (parquet or json)."""
    with tempfile.TemporaryDirectory() as tmp:
        local = client.download_artifacts(run_id, artifact_name, dst_path=tmp)
        if kind == "parquet":
            return pd.read_parquet(local)
        else:
            with open(local, "r") as f:
                payload = json.load(f)
            return pd.DataFrame(payload.get("results", {})), payload.get("cluster_col")

markers_df = _download_artifact_to_pandas("markers_flat.parquet", kind="parquet")
print(f"markers_flat.parquet shape: {markers_df.shape}")
print(f"columns sample: {list(markers_df.columns)[:10]}")

scim_anno_df, scim_cluster_col = _download_artifact_to_pandas("scimilarity_annotation.json", kind="json")
teddy_anno_df, teddy_cluster_col = _download_artifact_to_pandas("teddy_annotation.json", kind="json")
print(f"\nSCimilarity annotation: {len(scim_anno_df)} clusters; cluster_col={scim_cluster_col}")
print(f"TEDDY annotation:       {len(teddy_anno_df)} clusters; cluster_col={teddy_cluster_col}")

# COMMAND ----------

# DBTITLE 1,NEW: Load hvg_matrix.parquet if present (what the app sends to TEDDY)
hvg_df = None
try:
    from mlflow.exceptions import MlflowException
    with tempfile.TemporaryDirectory() as tmp:
        local = client.download_artifacts(run_id, "hvg_matrix.parquet", dst_path=tmp)
        hvg_df = pd.read_parquet(local)
    print(f"hvg_matrix.parquet shape: {hvg_df.shape}")
    hvg_expr_cols = [c for c in hvg_df.columns if c.startswith("expr_")]
    hvg_gene_names = [c.replace("expr_", "") for c in hvg_expr_cols]
    print(f"  expr_ columns: {len(hvg_expr_cols):,}")
    print(f"  First 10 gene names: {hvg_gene_names[:10]}")
    # Format check — ENSG, HGNC symbol, or other?
    ensg_count = sum(1 for g in hvg_gene_names if g.startswith("ENSG"))
    ensg_with_version = sum(1 for g in hvg_gene_names if g.startswith("ENSG") and "." in g)
    print(f"  Genes starting with ENSG: {ensg_count:,}/{len(hvg_gene_names):,}  (with .version suffix: {ensg_with_version:,})")
except Exception as e:
    print(f"No hvg_matrix.parquet on this run: {e}")

# COMMAND ----------

# DBTITLE 1,Surface the disagreement
sci_short = scim_anno_df[["Cluster", "Predicted Cell Type"]].rename(columns={"Predicted Cell Type": "SCimilarity"})
ted_short = teddy_anno_df[["Cluster", "Predicted Cell Type"]].rename(columns={"Predicted Cell Type": "TEDDY"})
disagree_df = sci_short.merge(ted_short, on="Cluster")
disagree_df["agree"] = disagree_df["SCimilarity"] == disagree_df["TEDDY"]
print("\n=== ALL CLUSTERS — SCimilarity vs TEDDY ===")
print(disagree_df.to_string(index=False))

print("\n=== DISAGREEING CLUSTERS ===")
print(disagree_df[~disagree_df["agree"]].to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Per-cluster NK vs plasma marker means (biological tiebreak)
NK_MARKERS = ["NCAM1", "NKG7", "GNLY", "PRF1", "GZMB", "KLRC1"]
PLASMA_MARKERS = ["CD38", "MZB1", "XBP1", "JCHAIN", "IGHG1", "IGHA1"]

expr_cols_all = [c for c in markers_df.columns if c.startswith("expr_")]
def _expr_col(symbol):
    return f"expr_{symbol}"

nk_present = [_expr_col(m) for m in NK_MARKERS if _expr_col(m) in markers_df.columns]
pl_present = [_expr_col(m) for m in PLASMA_MARKERS if _expr_col(m) in markers_df.columns]
print(f"NK markers found in run: {[c.replace('expr_','') for c in nk_present]}")
print(f"Plasma markers found in run: {[c.replace('expr_','') for c in pl_present]}")

cluster_col = teddy_cluster_col or scim_cluster_col
if cluster_col not in markers_df.columns:
    # Fall back to common defaults
    for c in ("cluster", "leiden", "louvain"):
        if c in markers_df.columns:
            cluster_col = c
            break

# Compute per-cluster marker means via simple groupby aggregations, then
# merge into disagree_df keyed by string Cluster. Avoid the rename/index dance.
def _cluster_mean(panel_cols):
    if not panel_cols:
        return {}
    s = markers_df.groupby(cluster_col)[panel_cols].mean().mean(axis=1)
    return {str(k): float(v) for k, v in s.items()}

nk_mean_map = _cluster_mean(nk_present)
pl_mean_map = _cluster_mean(pl_present)

per_cluster = disagree_df.copy()
per_cluster["NK_mean"] = per_cluster["Cluster"].astype(str).map(nk_mean_map)
per_cluster["Plasma_mean"] = per_cluster["Cluster"].astype(str).map(pl_mean_map)
per_cluster["NK > Plasma?"] = per_cluster["NK_mean"].fillna(0) > per_cluster["Plasma_mean"].fillna(0)
print("\n=== Per-cluster marker means + decision ===")
print(per_cluster.to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Pick disputed cluster + a representative cell
disputed = disagree_df[~disagree_df["agree"]]
if disputed.empty:
    print("No disagreements on this run — nothing to inspect.")
    dbutils.notebook.exit("no-disagreement")

# Prefer the cluster where SCimilarity says NK and TEDDY says plasma
nk_vs_plasma = disputed[
    disputed["SCimilarity"].str.contains("natural killer|NK ", case=False, na=False)
    & disputed["TEDDY"].str.contains("plasma", case=False, na=False)
]
focus_row = (nk_vs_plasma.iloc[0] if not nk_vs_plasma.empty else disputed.iloc[0])
focus_cluster = focus_row["Cluster"]
print(f"Focusing on cluster {focus_cluster}: SCimilarity='{focus_row['SCimilarity']}'  TEDDY='{focus_row['TEDDY']}'")

# Pick a single representative cell from that cluster
cells_in_cluster = markers_df[markers_df[cluster_col].astype(str) == str(focus_cluster)]
if cells_in_cluster.empty:
    raise RuntimeError(f"No cells found in markers_flat for cluster {focus_cluster}")
rep_cell = cells_in_cluster.iloc[0]
rep_idx = rep_cell.name
print(f"Representative cell index: {rep_idx}")

# COMMAND ----------

# DBTITLE 1,NEW: Per-cluster embedding degeneracy check
# Send ONE cell from each cluster to TEDDY (using the same HVG input the app
# uses). If the resulting embeddings are all very similar (cosine > 0.99),
# the model is collapsing the input — that explains every cluster mapping to
# the same plasma-cell neighborhood in the index.
from databricks.sdk import WorkspaceClient
w_diag = WorkspaceClient()

# Build the SAME input the app sends — HVG if available, else markers_flat.
if hvg_df is not None and not hvg_df.empty:
    src_df = hvg_df
    src_cluster_col = "cluster"
    src_expr_cols = hvg_expr_cols
    print("Diagnostic input: HVG matrix")
else:
    src_df = markers_df
    src_cluster_col = cluster_col
    src_expr_cols = expr_cols_all
    print("Diagnostic input: markers_flat (no HVG artifact)")

src_gene_names = [c.replace("expr_", "") for c in src_expr_cols]

# Reuse HGNC→ENSG mapping
import os, json
if os.path.exists(mapping_path):
    with open(mapping_path) as f:
        _mapping = json.load(f)
else:
    _mapping = {}
translated_genes = [_mapping.get(g, g) for g in src_gene_names]

# Pick one representative cell per cluster
diag_clusters = sorted(src_df[src_cluster_col].astype(str).unique(), key=str)
rep_rows = []
for cl in diag_clusters[:10]:  # at most 10 clusters for compactness
    cl_rows = src_df[src_df[src_cluster_col].astype(str) == cl]
    if not cl_rows.empty:
        rep_rows.append((cl, cl_rows.iloc[0]))
print(f"Picking one rep cell from each of {len(rep_rows)} clusters")

# Embed all reps in one request (batch=N small enough to fit)
diag_expr = [[float(r[col]) for col in src_expr_cols] for cl, r in rep_rows]
var_df = pd.DataFrame({"index": translated_genes})
obs_df = pd.DataFrame(index=[f"c{cl}" for cl, r in rep_rows])
payload = [{
    "adata_sparsematrix": diag_expr,
    "adata_obs": obs_df.to_json(orient="split"),
    "adata_var": var_df.to_json(orient="split"),
}]
import time
t0 = time.time()
resp = w_diag.serving_endpoints.query(name=endpoint, inputs=payload)
print(f"Endpoint call took {time.time()-t0:.1f}s")
preds = resp.predictions
print(f"Got {len(preds)} embeddings, dim={len(preds[0]['embedding'])}")

# Cosine similarity matrix
emb_arr = np.array([p["embedding"] for p in preds], dtype=np.float32)
norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
emb_n = emb_arr / np.maximum(norms, 1e-9)
sim = emb_n @ emb_n.T

print("\n=== Cosine similarity matrix between cluster representatives ===")
print("(values ~1.0 mean the embedding does NOT distinguish that pair)")
import pandas as _pd
sim_df = _pd.DataFrame(
    sim,
    index=[f"cl_{cl}" for cl, _ in rep_rows],
    columns=[f"cl_{cl}" for cl, _ in rep_rows],
)
print(sim_df.round(4).to_string())

# Off-diagonal stats
n = sim.shape[0]
mask = np.ones_like(sim, dtype=bool); np.fill_diagonal(mask, False)
off = sim[mask]
print(f"\nOff-diagonal stats (similarity between DIFFERENT clusters):")
print(f"  min={off.min():.4f}  mean={off.mean():.4f}  max={off.max():.4f}")
if off.mean() > 0.95:
    print("  ⚠️  Embeddings are NEAR-DEGENERATE — distinguishing different clusters is failing.")
elif off.mean() > 0.85:
    print("  ⚠️  Embeddings are weakly distinguishing — likely poor annotation quality.")
else:
    print("  ✓  Embeddings vary meaningfully across clusters.")

# Also: for each rep, top-5 neighbors. If different clusters return the
# SAME neighbors, the index lookup is also collapsing.
print("\n=== Top-5 neighbor cell_ids per cluster (overlap = bad) ===")
neighbor_sets = {}
for i, (cl, _) in enumerate(rep_rows):
    try:
        res = w_diag.vector_search_indexes.query_index(
            index_name=index_name,
            columns=["cell_id"],
            query_vector=[float(x) for x in emb_arr[i].tolist()],
            num_results=5,
        )
        ids = tuple(row[0] for row in (res.result.data_array or []))
        neighbor_sets[cl] = ids
        print(f"  cluster {cl}: {ids}")
    except Exception as e:
        print(f"  cluster {cl}: error {e}")
# Pairwise overlap
overlaps = []
keys = list(neighbor_sets.keys())
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        inter = len(set(neighbor_sets[keys[i]]) & set(neighbor_sets[keys[j]]))
        overlaps.append((keys[i], keys[j], inter))
print("\nPairwise overlap of top-5 neighbors across clusters (5 = identical, 0 = disjoint):")
for a, b, n_o in overlaps:
    print(f"  cl_{a} ∩ cl_{b}: {n_o}/5")

# COMMAND ----------

# DBTITLE 1,Embed the representative cell via TEDDY endpoint
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Reuse the same HGNC→ENSG mapping the app uses.
gene_names_raw = [c.replace("expr_", "") for c in expr_cols_all]
mapping = {}
if os.path.exists(mapping_path):
    with open(mapping_path) as f:
        mapping = json.load(f)
    print(f"Loaded gene mapping: {len(mapping):,} entries")
else:
    print(f"WARNING: gene mapping not found at {mapping_path} — gene names sent as-is")

translated = [mapping.get(g, g) for g in gene_names_raw]
n_already_ensg = sum(1 for g in gene_names_raw if g.startswith("ENSG"))
n_mapped = sum(1 for g, t in zip(gene_names_raw, translated) if g != t)
n_unmapped = sum(1 for g, t in zip(gene_names_raw, translated) if g == t and not g.startswith("ENSG"))
print(f"  already ENSG: {n_already_ensg:,}  mapped HGNC→ENSG: {n_mapped:,}  unmapped: {n_unmapped:,}")

X_row = rep_cell[expr_cols_all].astype(float).tolist()
var_df = pd.DataFrame({"index": translated})
obs_df = pd.DataFrame(index=[str(rep_idx)])
payload = [{
    "adata_sparsematrix": [X_row],
    "adata_obs": obs_df.to_json(orient="split"),
    "adata_var": var_df.to_json(orient="split"),
}]
resp = w.serving_endpoints.query(name=endpoint, inputs=payload)
embedding = resp.predictions[0]["embedding"]
print(f"Got embedding of dim {len(embedding)}")

# COMMAND ----------

# DBTITLE 1,KNN search the TEDDY index for top-50 neighbors of this cell
result = w.vector_search_indexes.query_index(
    index_name=index_name,
    columns=["cell_id"],
    query_vector=[float(x) for x in embedding],
    num_results=50,
)
neighbor_ids = [row[0] for row in (result.result.data_array or [])]
print(f"Got {len(neighbor_ids)} neighbors")

# Fetch (cell_type, disease, tissue, dataset_id) for those neighbors
ids_sql = ", ".join(f"'{cid}'" for cid in neighbor_ids)
meta_df = spark.sql(f"""
    SELECT cell_id, cell_type, disease, tissue, tissue_general, dataset_id
    FROM {catalog}.{schema}.teddy_cells
    WHERE cell_id IN ({ids_sql})
""").toPandas()
# Preserve VS ranking
meta_df = meta_df.set_index("cell_id").reindex(neighbor_ids).reset_index()

print(f"\n=== TEDDY top-50 neighbors of representative cell in cluster {focus_cluster} ===")
print(meta_df.head(20).to_string())

print(f"\n=== cell_type distribution of top-50 neighbors ===")
print(meta_df["cell_type"].value_counts())

print(f"\n=== disease distribution of top-50 neighbors ===")
print(meta_df["disease"].value_counts())

# COMMAND ----------

# DBTITLE 1,Global teddy_cells reference distribution (the bias check)
ref_dist = spark.sql(f"""
    SELECT cell_type, COUNT(*) AS n
    FROM {catalog}.{schema}.teddy_cells
    GROUP BY cell_type
    ORDER BY 2 DESC
""").toPandas()

total = ref_dist["n"].sum()
ref_dist["pct"] = ref_dist["n"] / total * 100
print(f"\n=== TEDDY reference cell_type distribution (top 20 of {len(ref_dist)}) ===")
print(ref_dist.head(20).to_string(index=False))

print(f"\nTotal cells in reference: {int(total):,}")
nk_rows = ref_dist[ref_dist["cell_type"].str.contains("natural killer|NK ", case=False, na=False)]
plasma_rows = ref_dist[ref_dist["cell_type"].str.contains("plasma", case=False, na=False)]
print(f"\nNK-related rows in reference:")
print(nk_rows.to_string(index=False))
print(f"\nPlasma-related rows in reference:")
print(plasma_rows.to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Summary verdict
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Run:               {run_name}  (id={run_id})")
print(f"Disputed cluster:  {focus_cluster}")
print(f"  SCimilarity:     {focus_row['SCimilarity']}")
print(f"  TEDDY:           {focus_row['TEDDY']}")
focus_row_df = per_cluster[per_cluster["Cluster"].astype(str) == str(focus_cluster)]
if not focus_row_df.empty:
    r = focus_row_df.iloc[0]
    nk_m = float(r["NK_mean"]) if pd.notna(r["NK_mean"]) else 0.0
    pl_m = float(r["Plasma_mean"]) if pd.notna(r["Plasma_mean"]) else 0.0
    print(f"  NK marker mean:  {nk_m:.3f}")
    print(f"  Plasma marker mean: {pl_m:.3f}")
    print(f"  Biology favors:  {'NK' if nk_m > pl_m else 'Plasma'}")

nk_neigh = meta_df["cell_type"].str.contains("natural killer|NK ", case=False, na=False).sum()
plasma_neigh = meta_df["cell_type"].str.contains("plasma", case=False, na=False).sum()
print(f"\nOf top-50 TEDDY neighbors:")
print(f"  NK-like:     {nk_neigh}/50")
print(f"  Plasma-like: {plasma_neigh}/50")

nk_in_ref = int(nk_rows["n"].sum())
plasma_in_ref = int(plasma_rows["n"].sum())
print(f"\nReference imbalance:")
print(f"  NK-like cells in teddy_cells:     {nk_in_ref:,} ({nk_in_ref/total*100:.2f}%)")
print(f"  Plasma-like cells in teddy_cells: {plasma_in_ref:,} ({plasma_in_ref/total*100:.2f}%)")
print(f"  Plasma/NK ratio in reference:     {plasma_in_ref/max(1,nk_in_ref):.1f}×")

if plasma_in_ref > nk_in_ref * 3 and plasma_neigh > nk_neigh * 2:
    print("\n→ Reference-bias hypothesis CONFIRMED.")
    print("  Fix path: rebuild reference with healthy cells + IDF-weighted voting.")
else:
    print("\n→ Reference-bias hypothesis NOT cleanly confirmed — look at marker means + neighbor diseases.")

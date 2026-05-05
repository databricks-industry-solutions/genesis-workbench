# Databricks notebook source
# MAGIC %md
# MAGIC # Extract SCimilarity Reference Corpus to Delta
# MAGIC
# MAGIC Walks the `CellQuery` in-memory reference (the same `model_path` downloaded in `01_wget_scimilarity.py`)
# MAGIC and writes a single managed Delta table `{catalog}.{schema}.scimilarity_cells` with:
# MAGIC
# MAGIC - `cell_id STRING` (primary key)
# MAGIC - `embedding ARRAY<FLOAT>` (128-d)
# MAGIC - All metadata columns from `cq.cell_metadata` (`cell_type`, `disease`, `tissue`, `study`, ...)
# MAGIC
# MAGIC Change Data Feed is enabled so the Vector Search Delta Sync index (notebook 06c) can track it.
# MAGIC
# MAGIC The reference matrix is ~12 GB, so rows are written in batches. This notebook is idempotent —
# MAGIC it skips the build if the target table already has the full reference populated.

# COMMAND ----------

# DBTITLE 1,install/load dependencies
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Target table + batch size
CELLS_TABLE = f"{CATALOG}.{DB_SCHEMA}.scimilarity_cells"
BATCH_SIZE = 1_000_000  # ~1M cells per batch; keeps each Spark DF well under driver RAM

print(f"Target table : {CELLS_TABLE}")
print(f"Batch size   : {BATCH_SIZE:,}")

# COMMAND ----------

# DBTITLE 1,Load CellQuery (metadata eager; embeddings stay in TileDB)
import numpy as np
from scimilarity import CellQuery

# load_knn=False skips the HNSW index — we only need metadata + embedding access.
cq = CellQuery(model_path, load_knn=False)

# CellQuery only exposes cell_metadata as a DataFrame attribute.
# Embeddings live in a TileDB array at cq.embedding_tiledb_uri and are read
# on demand via cq.get_precomputed_embeddings(indices).
cell_metadata = cq.cell_metadata
n_cells = len(cell_metadata)

# Probe the embedding dimension by reading a single row.
_probe = cq.get_precomputed_embeddings(np.array([0], dtype=np.int64))
_probe = np.asarray(_probe).reshape(1, -1)
emb_dim = _probe.shape[1]

print(f"Reference cells    : {n_cells:,}")
print(f"Embedding dim      : {emb_dim}")
print(f"Metadata columns   : {list(cell_metadata.columns)}")
print(f"Metadata row count : {len(cell_metadata):,}")

assert emb_dim == 128, f"Expected embedding dim 128, got {emb_dim}"

# COMMAND ----------

# DBTITLE 1,Idempotency: skip if the table is already populated
skip_build = False
if spark.catalog.tableExists(CELLS_TABLE):
    existing = spark.table(CELLS_TABLE).count()
    print(f"Table {CELLS_TABLE} already exists with {existing:,} rows")
    if existing >= n_cells:
        print("Existing table already has the full reference — skipping rebuild.")
        skip_build = True
    else:
        print(f"Existing table is incomplete ({existing:,} < {n_cells:,}) — rebuilding.")

# COMMAND ----------

# DBTITLE 1,Build cell_id and metadata schema
import numpy as np
import pandas as pd
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType, FloatType,
)

# Derive cell_id from the metadata index. Fall back to a synthesized row number if the index is not unique.
cell_ids = cell_metadata.index.astype(str)
if not cell_ids.is_unique:
    print("cell_metadata index is not unique — synthesizing cell_id from row number.")
    cell_ids = pd.Index([str(i) for i in range(n_cells)])

# Build a Spark schema: cell_id, embedding, then every metadata column as STRING.
# We cast to string at write time because metadata column types vary (categoricals, mixed NAs).
metadata_cols = list(cell_metadata.columns)
spark_fields = [
    StructField("cell_id", StringType(), nullable=False),
    StructField("embedding", ArrayType(FloatType(), containsNull=False), nullable=False),
]
for col in metadata_cols:
    spark_fields.append(StructField(col, StringType(), nullable=True))
spark_schema = StructType(spark_fields)

print(f"Spark schema columns: {[f.name for f in spark_fields]}")

# COMMAND ----------

# DBTITLE 1,Write reference to Delta in batches
if not skip_build:
    # Clean slate for a full rebuild
    spark.sql(f"DROP TABLE IF EXISTS {CELLS_TABLE}")

    written = 0
    batch_num = 0

    for start in range(0, n_cells, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_cells)
        batch_num += 1

        emb_batch = np.asarray(
            cq.get_precomputed_embeddings(np.arange(start, end, dtype=np.int64))
        ).astype(np.float32).reshape(end - start, emb_dim)
        # reset_index(drop=True) is critical: cell_metadata's original index is preserved
        # by .iloc, so for batches past the first one its values (start..end-1) don't match
        # pdf's default 0..len-1 RangeIndex and pandas-aligned column assignment silently
        # produces all-NaN. Resetting the index makes alignment by position.
        meta_batch = cell_metadata.iloc[start:end].reset_index(drop=True)
        ids_batch = cell_ids[start:end]

        # Assemble a pandas DataFrame: cell_id, embedding as python list, metadata columns as strings
        pdf = pd.DataFrame({"cell_id": ids_batch.to_numpy()})
        pdf["embedding"] = [row.tolist() for row in emb_batch]
        for col in metadata_cols:
            # astype(str) turns NaN/None into the literal "nan"/"None"; use where() to preserve nulls.
            col_vals = meta_batch[col]
            pdf[col] = col_vals.astype(object).where(col_vals.notna(), None).astype(object).map(
                lambda v: None if v is None else str(v)
            )

        sdf = spark.createDataFrame(pdf, schema=spark_schema)
        sdf.write.format("delta").mode("append").saveAsTable(CELLS_TABLE)

        written += (end - start)
        print(f"Batch {batch_num}: wrote rows [{start:,}, {end:,}) — total {written:,}/{n_cells:,}")

    print(f"Finished writing {written:,} cells to {CELLS_TABLE}")

# COMMAND ----------

# DBTITLE 1,Enable Change Data Feed (required by Vector Search Delta Sync)
spark.sql(
    f"ALTER TABLE {CELLS_TABLE} "
    f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
print(f"CDF enabled on {CELLS_TABLE}")

# COMMAND ----------

# DBTITLE 1,Verify write
row_count = spark.table(CELLS_TABLE).count()
print(f"Final row count: {row_count:,}")
assert row_count == n_cells, f"Row count mismatch: {row_count:,} vs expected {n_cells:,}"

# Spot-check embedding length
from pyspark.sql import functions as F
size_stats = (
    spark.table(CELLS_TABLE)
    .select(F.min(F.size("embedding")).alias("min_dim"), F.max(F.size("embedding")).alias("max_dim"))
    .collect()[0]
)
print(f"Embedding size min/max: {size_stats['min_dim']}/{size_stats['max_dim']}")
assert size_stats["min_dim"] == 128 and size_stats["max_dim"] == 128

display(spark.table(CELLS_TABLE).limit(5))

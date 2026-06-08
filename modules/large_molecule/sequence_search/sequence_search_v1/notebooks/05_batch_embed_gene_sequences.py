# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Embed Human SwissProt Proteins with ESM-2
# MAGIC
# MAGIC Generates 1280-dimensional mean-pooled embeddings for the human reviewed
# MAGIC proteins in `gene_sequences` (built by core's `ingest_uniprot_genes.py`)
# MAGIC using the SAME model as the UniRef corpus — `facebook/esm2_t33_650M_UR50D` —
# MAGIC so a single query embedding can search both indexes in one vector space.
# MAGIC
# MAGIC Output: `gene_sequence_embeddings` (seq_id = UniProt accession, embedding).
# MAGIC This is the human-protein companion to `sequence_embeddings`; it exists so
# MAGIC protein search returns human targets (e.g. PARP1) that the microbe-dominated
# MAGIC UniRef90 slice does not contain.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q torch==2.3.1 transformers==4.41.2 databricks-sdk==0.50.0 databricks-sql-connector==4.0.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run utils (declares widgets, creates UC resources)
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Read widget values
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# DBTITLE 1,Config — same model + pooling as the UniRef corpus (03)
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "32")

NUM_WORKERS = 4
NUM_PARTITIONS = NUM_WORKERS * 4  # gene_sequences is ~20K rows — keep partitions modest
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
print(f"Arrow batch size: 32, partitions: {NUM_PARTITIONS}, workers: {NUM_WORKERS}, model: {ESM2_MODEL}")

# COMMAND ----------

# DBTITLE 1,Define pandas_udf for ESM-2 embedding (identical to 03)
import pandas as pd
from typing import Iterator
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import ArrayType, FloatType


@pandas_udf(ArrayType(FloatType()))
def embed_sequences(batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
    """Iterator pandas_udf — loads ESM-2 once per worker in FP16, mean-pools the
    last hidden state over non-padding tokens. Matches 03 + the serving model."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
    model = AutoModel.from_pretrained(ESM2_MODEL, torch_dtype=torch.float16).cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"ESM-2 (FP16) loaded on {torch.cuda.get_device_name(0)}")

    for sequences in batches:
        seq_list = sequences.tolist()
        tokens = tokenizer(
            seq_list, return_tensors="pt", truncation=True,
            max_length=1024, padding=True
        ).to("cuda")
        with torch.no_grad():
            output = model(**tokens)
        attention_mask = tokens["attention_mask"]
        hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        embeddings = (summed / counts).cpu().float().tolist()
        yield pd.Series(embeddings)

# COMMAND ----------

# DBTITLE 1,Run batch embedding over gene_sequences
source_table = f"{catalog}.{schema}.gene_sequences"
target_table = f"{catalog}.{schema}.gene_sequence_embeddings"

if not spark.catalog.tableExists(source_table):
    raise RuntimeError(
        f"{source_table} not found — run core's ingest_uniprot_genes.py first "
        "(it builds the human SwissProt gene_sequences table)."
    )

skip_embedding = False
if spark.catalog.tableExists(target_table):
    existing_count = spark.table(target_table).count()
    if existing_count > 100:
        print(f"Embeddings table {target_table} already has {existing_count} rows, skipping.")
        skip_embedding = True

if not skip_embedding:
    # accession is the unique UniProt key → use it as the index primary key (seq_id).
    df = spark.table(source_table).select(col("accession").alias("seq_id"), "sequence")
    total_rows = df.count()
    print(f"Source: {source_table}, embedding {total_rows:,} human proteins with {ESM2_MODEL}")

    embeddings_df = (
        df.repartition(NUM_PARTITIONS)
          .select("seq_id", embed_sequences("sequence").alias("embedding"))
    )
    embeddings_df.write.format("delta").mode("overwrite").saveAsTable(target_table)
    print(f"Embeddings written to {target_table}")

# COMMAND ----------

# DBTITLE 1,Verify embeddings
from pyspark.sql.functions import size

result_df = spark.table(target_table)
print(f"Total embeddings: {result_df.count()}")
dim_check = result_df.select(size("embedding").alias("dim")).limit(1).collect()[0]["dim"]
print(f"Embedding dimension: {dim_check}")
assert dim_check == 1280, f"Expected 1280d embeddings, got {dim_check}d"
display(result_df.limit(5))

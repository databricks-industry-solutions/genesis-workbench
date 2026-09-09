# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Embed Human SwissProt Proteins with ESM-2 — Serverless GPU (single A10)
# MAGIC
# MAGIC Serverless-GPU variant of `05_batch_embed_gene_sequences.py`. Generates
# MAGIC 1280-dimensional mean-pooled embeddings for the human reviewed proteins in
# MAGIC `gene_sequences` (built by core's `ingest_uniprot_genes.py`) using the SAME
# MAGIC model as the UniRef corpus — `facebook/esm2_t33_650M_UR50D` — and writes them
# MAGIC to `gene_sequence_embeddings`.
# MAGIC
# MAGIC `gene_sequences` is small (~20K human proteins), so instead of a distributed
# MAGIC fan-out (see `03_..._sgc.py`) this runs directly on the single A10 the task is
# MAGIC scheduled on: read the table, embed in FP16 batches, write the Delta table.
# MAGIC The embedding math (tokenize → forward → attention-mask-weighted mean pool)
# MAGIC is identical to `03` and the UC-registered serving model.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# torch/CUDA are preinstalled on the serverless GPU AI runtime; add transformers.
# hf_transfer is required because the runtime sets HF_HUB_ENABLE_HF_TRANSFER=1.
# MAGIC %pip install -q transformers==4.41.2 hf_transfer==0.1.9
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
BATCH_SIZE = 32
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"

SOURCE_TABLE = f"{catalog}.{schema}.gene_sequences"
TARGET_TABLE = f"{catalog}.{schema}.gene_sequence_embeddings"
print(f"batch size: {BATCH_SIZE}, model: {ESM2_MODEL}")
print(f"source: {SOURCE_TABLE}")
print(f"target: {TARGET_TABLE}")

# COMMAND ----------

# DBTITLE 1,Guards — source must exist; skip if already populated
if not spark.catalog.tableExists(SOURCE_TABLE):
    raise RuntimeError(
        f"{SOURCE_TABLE} not found — run core's ingest_uniprot_genes.py first "
        "(it builds the human SwissProt gene_sequences table)."
    )

skip_embedding = False
if spark.catalog.tableExists(TARGET_TABLE):
    existing_count = spark.table(TARGET_TABLE).count()
    if existing_count > 100:
        print(f"Embeddings table {TARGET_TABLE} already has {existing_count} rows, skipping.")
        skip_embedding = True

# COMMAND ----------

# DBTITLE 1,Read gene_sequences to the driver (small table, ~20K rows)
if not skip_embedding:
    from pyspark.sql.functions import col

    # accession is the unique UniProt key → use it as the index primary key (seq_id).
    pdf = (
        spark.table(SOURCE_TABLE)
        .select(col("accession").alias("seq_id"), "sequence")
        .toPandas()
    )
    print(f"Loaded {len(pdf):,} human proteins from {SOURCE_TABLE}")

# COMMAND ----------

# DBTITLE 1,Embed on the local A10 (FP16), matching 03's pooling exactly
if not skip_embedding:
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
    model = AutoModel.from_pretrained(ESM2_MODEL, torch_dtype=torch.float16).cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"ESM-2 (FP16) loaded on {torch.cuda.get_device_name(0)}")

    seq_ids: list = []
    embeddings: list = []
    n = len(pdf)
    for start in range(0, n, BATCH_SIZE):
        batch = pdf.iloc[start:start + BATCH_SIZE]
        tokens = tokenizer(
            batch["sequence"].tolist(),
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        ).to("cuda")
        with torch.no_grad():
            output = model(**tokens)
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        summed = (output.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        embs = (summed / counts).cpu().float().tolist()
        seq_ids.extend(batch["seq_id"].tolist())
        embeddings.extend(embs)
        if (start // BATCH_SIZE) % 50 == 0:
            print(f"processed {start + len(batch):,}/{n:,}")
    print(f"Embedded {len(seq_ids):,} sequences")

# COMMAND ----------

# DBTITLE 1,Write embeddings to the Delta table (array<float>, matching 03)
if not skip_embedding:
    from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

    out_schema = StructType([
        StructField("seq_id", StringType(), False),
        StructField("embedding", ArrayType(FloatType()), False),
    ])
    out_df = spark.createDataFrame(
        list(zip(seq_ids, embeddings)), schema=out_schema
    )
    out_df.write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)
    print(f"Embeddings written to {TARGET_TABLE}")

# COMMAND ----------

# DBTITLE 1,Verify embeddings
from pyspark.sql.functions import size

result_df = spark.table(TARGET_TABLE)
print(f"Total embeddings: {result_df.count()}")
dim_check = result_df.select(size("embedding").alias("dim")).limit(1).collect()[0]["dim"]
print(f"Embedding dimension: {dim_check}")
assert dim_check == 1280, f"Expected 1280d embeddings, got {dim_check}d"
display(result_df.limit(5))

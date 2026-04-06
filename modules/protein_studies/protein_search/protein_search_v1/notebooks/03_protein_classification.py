# Databricks notebook source
# MAGIC %md
# MAGIC # Protein Classification: Water Soluble vs Membrane Transport
# MAGIC
# MAGIC Uses the `Rostlab/prot_bert_bfd_membrane` protein language model to classify
# MAGIC protein sequences as either water-soluble or membrane transport proteins.
# MAGIC
# MAGIC Requires GPU compute (multi-GPU A10 driver node) for efficient inference.

# COMMAND ----------

# DBTITLE 1,Install torch & transformers
# MAGIC %pip install -q torch==2.3.1 transformers==4.41.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run utils (declares widgets, creates UC resources)
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Read widget values
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume_name = dbutils.widgets.get("volume_name")

# COMMAND ----------

# DBTITLE 1,Downsample enriched protein data
spark.sql(f"USE {catalog}.{schema}")

spark.sql(f"""
    CREATE OR REPLACE TABLE {catalog}.{schema}.tiny_sample_data AS
    SELECT *
    FROM {catalog}.{schema}.enriched_protein
    TABLESAMPLE (0.5 PERCENT) REPEATABLE (42)
""")

print(f"Sample size: {spark.table(f'{catalog}.{schema}.tiny_sample_data').count()} records")

# COMMAND ----------

# DBTITLE 1,Configure Spark for GPU-aware batching
import torch

num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
print(f"GPUs available: {num_gpus}")

# Control how many rows Spark sends to each pandas_udf call
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "32")

# COMMAND ----------

# DBTITLE 1,Define Pandas UDF with singleton model per GPU
from pyspark.sql import functions as F, types as T
import pandas as pd

schema_struct = T.StructType([
    T.StructField("label", T.StringType(), True),
    T.StructField("score", T.FloatType(), True),
])

MAX_SEQ_LEN = 1024

_pipe_singleton = None

def _get_pipeline():
    global _pipe_singleton
    if _pipe_singleton is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
        gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
        _pipe_singleton = TextClassificationPipeline(
            model=AutoModelForSequenceClassification.from_pretrained("Rostlab/prot_bert_bfd_membrane"),
            tokenizer=AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd_membrane"),
            device=gpu_id if gpu_id >= 0 else "cpu",
        )
        print(f"Model loaded on GPU {gpu_id}")
    return _pipe_singleton


@F.pandas_udf(schema_struct)
def classify_protein(sequences: pd.Series) -> pd.DataFrame:
    import torch
    pipe = _get_pipeline()
    labels = []
    scores = []
    batch_size = 4
    seq_list = sequences.tolist()
    for i in range(0, len(seq_list), batch_size):
        batch = [s[:MAX_SEQ_LEN] for s in seq_list[i:i + batch_size]]
        with torch.no_grad():
            results = pipe(batch, batch_size=batch_size, truncation=True, max_length=MAX_SEQ_LEN)
        for result in results:
            item = result[0] if isinstance(result, list) else result
            labels.append(item["label"])
            scores.append(item["score"])
        del results, batch
        torch.cuda.empty_cache()
    return pd.DataFrame({"label": labels, "score": scores})

# COMMAND ----------

# DBTITLE 1,Run batch classification
df = spark.read.table(f"{catalog}.{schema}.tiny_sample_data")

df = df.withColumn("spaced_sequence", F.expr("concat_ws(' ', split(Sequence, ''))"))
df = df.withColumn("spaced_sequence", F.expr("trim(spaced_sequence)"))

# Repartition to number of GPUs so each partition runs on one GPU
df = df.repartition(num_gpus)

df = df.withColumn("classification", classify_protein("spaced_sequence"))
df = df.select("*", "classification.*")

# COMMAND ----------

# DBTITLE 1,Write classified proteins to UC
df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
    f"{catalog}.{schema}.proteinclassification_tiny"
)

print(f"Classified proteins written to {catalog}.{schema}.proteinclassification_tiny")

# Databricks notebook source
# MAGIC %md
# MAGIC # Protein Classification: Water Soluble vs Membrane Transport
# MAGIC
# MAGIC Uses the `Rostlab/prot_bert_bfd_membrane` protein language model to classify
# MAGIC protein sequences as either water-soluble or membrane transport proteins.
# MAGIC
# MAGIC Uses `predict_batch_udf` for GPU-efficient batch inference — the model is
# MAGIC loaded once per worker and reused across all batches, avoiding OOM issues.

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

# DBTITLE 1,Prepare sequences for classification
from pyspark.sql import functions as F

df = spark.read.table(f"{catalog}.{schema}.tiny_sample_data")
df = df.withColumn("spaced_sequence", F.expr("concat_ws(' ', split(Sequence, ''))"))
df = df.withColumn("spaced_sequence", F.expr("trim(spaced_sequence)"))

print(f"Ready for classification: {df.count()} rows")

# COMMAND ----------

# DBTITLE 1,Define predict_batch_udf for GPU inference
import numpy as np
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.ml.functions import predict_batch_udf

MAX_SEQ_LEN = 1024

def make_predict_fn():
    """Factory that loads model once per worker and returns a predict function."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = TextClassificationPipeline(
        model=AutoModelForSequenceClassification.from_pretrained("Rostlab/prot_bert_bfd_membrane"),
        tokenizer=AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd_membrane"),
        device=device,
    )
    print(f"ProtBERT membrane classifier loaded on {device}")

    def predict(inputs: np.ndarray) -> list:
        """Classify a batch of spaced protein sequences."""
        sequences = [s[:MAX_SEQ_LEN] if len(s) > MAX_SEQ_LEN else s for s in inputs.tolist()]
        with torch.no_grad():
            results = pipe(sequences, truncation=True, max_length=MAX_SEQ_LEN, batch_size=4)
        torch.cuda.empty_cache()

        output = []
        for result in results:
            item = result[0] if isinstance(result, list) else result
            output.append({"label": item["label"], "score": float(item["score"])})
        return output

    return predict

classify_udf = predict_batch_udf(
    make_predict_fn,
    return_type=StructType([
        StructField("label", StringType()),
        StructField("score", FloatType()),
    ]),
    batch_size=8,
)

# COMMAND ----------

# DBTITLE 1,Run batch classification
df_classified = df.withColumn("classification", classify_udf("spaced_sequence"))
df_classified = df_classified.select("*", "classification.label", "classification.score")
df_classified = df_classified.drop("classification")

# COMMAND ----------

# DBTITLE 1,Write classified proteins to UC
df_classified.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
    f"{catalog}.{schema}.proteinclassification_tiny"
)

print(f"Classified proteins written to {catalog}.{schema}.proteinclassification_tiny")

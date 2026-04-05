# Databricks notebook source
# MAGIC %md
# MAGIC # Protein Classification: Water Soluble vs Membrane Transport
# MAGIC
# MAGIC Uses the `Rostlab/prot_bert_bfd_membrane` protein language model to classify
# MAGIC protein sequences as either water-soluble or membrane transport proteins.
# MAGIC
# MAGIC Requires GPU compute (A10) for efficient inference.

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

# DBTITLE 1,Import libraries
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import re

# COMMAND ----------

# DBTITLE 1,Initialize ProtBERT membrane classifier
pipeline = TextClassificationPipeline(
    model=AutoModelForSequenceClassification.from_pretrained("Rostlab/prot_bert_bfd_membrane"),
    tokenizer=AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd_membrane"),
    device="cuda" if torch.cuda.is_available() else "cpu",
)

print(f"Model loaded on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

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

# DBTITLE 1,Define Pandas UDF for protein classification
from pyspark.sql import functions as F, types as T
import pandas as pd

schema_struct = T.StructType([
    T.StructField("label", T.StringType(), True),
    T.StructField("score", T.FloatType(), True),
])


@F.pandas_udf(schema_struct)
def classify_protein(sequences: pd.Series) -> pd.DataFrame:
    results = [pipeline(sequence) for sequence in sequences]
    labels = [result[0]["label"] for result in results]
    scores = [result[0]["score"] for result in results]
    return pd.DataFrame({"label": labels, "score": scores})

# COMMAND ----------

# DBTITLE 1,Run batch classification
df = spark.read.table(f"{catalog}.{schema}.tiny_sample_data")

df = df.withColumn("spaced_sequence", F.expr("concat_ws(' ', split(Sequence, ''))"))
df = df.withColumn("spaced_sequence", F.expr("trim(spaced_sequence)"))

df = df.withColumn("classification", classify_protein("spaced_sequence"))
df = df.select("*", "classification.*")

display(df.limit(100))

# COMMAND ----------

# DBTITLE 1,Write classified proteins to UC
df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
    f"{catalog}.{schema}.proteinclassification_tiny"
)

print(f"Classified {df.count()} proteins written to {catalog}.{schema}.proteinclassification_tiny")

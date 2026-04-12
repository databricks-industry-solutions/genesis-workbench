# Databricks notebook source
# MAGIC %md
# MAGIC ## Create Sequence Search Delta Tables
# MAGIC
# MAGIC Reads the downloaded UniRef90 FASTA file and creates a Delta table
# MAGIC in Unity Catalog. Processes in batches to handle the ~150M sequences
# MAGIC without running out of memory.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q biopython==1.86
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

# DBTITLE 1,Parse FASTA and create sequence_db table in batches
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from Bio import SeqIO

fasta_path = f"/Volumes/{catalog}/{schema}/{volume_name}/uniref90.fasta"
table_name = f"{catalog}.{schema}.sequence_db"

# Check if table already exists with data
skip_creation = False
if spark.catalog.tableExists(table_name):
    table_row_count = spark.table(table_name).count()
    print(f"Table {table_name} already exists with {table_row_count} rows")
    if table_row_count > 100:
        print(f"Table has sufficient data, skipping table creation.")
        skip_creation = True
    else:
        print(f"Table has too few rows ({table_row_count}), recreating table.")

if not skip_creation:
    schema_def = StructType([
        StructField("seq_id", StringType(), False),
        StructField("sequence", StringType(), False),
        StructField("description", StringType(), True),
        StructField("seq_length", IntegerType(), False),
    ])

    BATCH_SIZE = 500_000
    batch = []
    total_count = 0
    batch_num = 0

    # Drop existing table for clean initial load
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    with open(fasta_path, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            seq_str = str(record.seq)
            batch.append(Row(
                seq_id=record.id,
                sequence=seq_str,
                description=record.description,
                seq_length=len(seq_str),
            ))
            total_count += 1

            if len(batch) >= BATCH_SIZE:
                batch_num += 1
                df = spark.createDataFrame(batch, schema=schema_def)
                df.write.format("delta").mode("append").saveAsTable(table_name)
                print(f"Batch {batch_num}: wrote {len(batch)} records (total: {total_count})")
                batch = []

    # Write any remaining records
    if batch:
        batch_num += 1
        df = spark.createDataFrame(batch, schema=schema_def)
        df.write.format("delta").mode("append").saveAsTable(table_name)
        print(f"Batch {batch_num}: wrote {len(batch)} records (total: {total_count})")

    print(f"sequence_db table created with {total_count} total records")

# COMMAND ----------

# DBTITLE 1,Verify table
display(spark.table(table_name).limit(10))
print(f"Total rows: {spark.table(table_name).count()}")

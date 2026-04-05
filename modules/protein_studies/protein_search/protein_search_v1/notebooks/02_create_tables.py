# Databricks notebook source
# MAGIC %md
# MAGIC ## Create Protein Delta Tables (without DLT pipeline)
# MAGIC
# MAGIC Reads the downloaded FASTA file and creates bronze, silver, and enriched
# MAGIC Delta tables in Unity Catalog.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q biopython
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run utils
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Configure UC paths
uc_config = setup_uc_paths(silent=False)

catalog_name = uc_config["catalog_name"]
schema_name = uc_config["schema_name"]
volume_name = uc_config["volume_name"]

# COMMAND ----------

# DBTITLE 1,Parse FASTA and create Bronze table
from pyspark.sql import Row
from Bio import SeqIO

records = []
file_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/uniprot_sprot.fasta"

with open(file_path, "r") as f:
    for record in SeqIO.parse(f, "fasta"):
        records.append(
            Row(
                ID=record.id,
                Sequence=str(record.seq),
                Description=record.description,
            )
        )

df = spark.createDataFrame(records)
df.write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog_name}.{schema_name}.bronze_protein"
)

print(f"Bronze table created with {df.count()} records")

# COMMAND ----------

# DBTITLE 1,Extract protein info into Silver table
from pyspark.sql.functions import regexp_extract

fasta_df = spark.table(f"{catalog_name}.{schema_name}.bronze_protein")

os_regex = r"OS=([^ ]+ [^ ]+|\([^)]+\))"
ox_regex = r"OX=(\d+)"
gn_regex = r"GN=([^ ]+)"
pe_regex = r"PE=(\d)"
sv_regex = r"SV=(\d)"

fasta_df = fasta_df.withColumn(
    "ProteinName", regexp_extract("Description", r" (.+?) OS=", 1)
)
fasta_df = fasta_df.withColumn("OrganismName", regexp_extract("Description", os_regex, 1))
fasta_df = fasta_df.withColumn(
    "OrganismIdentifier", regexp_extract("Description", ox_regex, 1)
)
fasta_df = fasta_df.withColumn("GeneName", regexp_extract("Description", gn_regex, 1))
fasta_df = fasta_df.withColumn(
    "ProteinExistence", regexp_extract("Description", pe_regex, 1)
)
fasta_df = fasta_df.withColumn(
    "SequenceVersion", regexp_extract("Description", sv_regex, 1)
)

fasta_df.write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog_name}.{schema_name}.silver_protein"
)

print("Silver table created")

# COMMAND ----------

# DBTITLE 1,Add molecular weight to create Enriched table
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
from Bio.SeqUtils import molecular_weight
from Bio.Seq import Seq
import pandas as pd


@pandas_udf(DoubleType())
def get_molecular_weight_pandas_udf(sequence: pd.Series) -> pd.Series:
    def calculate_mw(seq):
        try:
            return molecular_weight(Seq(seq), seq_type="protein")
        except ValueError:
            return 1.0

    return sequence.apply(calculate_mw)


df = spark.table(f"{catalog_name}.{schema_name}.silver_protein")
df = df.withColumn("Molecular_Weight", get_molecular_weight_pandas_udf(df["Sequence"]))
df = df.drop("Description")

df.write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog_name}.{schema_name}.enriched_protein"
)

print("Enriched table created")

# COMMAND ----------

# DBTITLE 1,Preview enriched protein table
display(spark.table(f"{catalog_name}.{schema_name}.enriched_protein"))

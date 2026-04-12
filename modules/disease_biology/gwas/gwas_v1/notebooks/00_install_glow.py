# Databricks notebook source
# MAGIC %md
# MAGIC # Install Glow Libraries
# MAGIC
# MAGIC Builds Glow from source and copies the jar + whl to a UC Volume so that
# MAGIC downstream GWAS clusters can install them without rebuilding.
# MAGIC
# MAGIC Adapted from [mini-glow-demo](https://github.com/projectglow/glow).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.gwas_libraries")

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /local_disk0/
# MAGIC if [ ! -d "glow" ]; then
# MAGIC   git clone --depth=1 https://github.com/projectglow/glow.git
# MAGIC fi

# COMMAND ----------

# MAGIC %sh
# MAGIC sudo apt-get update -qq
# MAGIC sudo apt-get install -yqq apt-transport-https curl gnupg
# MAGIC echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
# MAGIC echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | sudo tee /etc/apt/sources.list.d/sbt_old.list
# MAGIC curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo -H gpg --no-default-keyring --keyring gnupg-ring:/etc/apt/trusted.gpg.d/scalasbt-release.gpg --import
# MAGIC sudo chmod 644 /etc/apt/trusted.gpg.d/scalasbt-release.gpg
# MAGIC sudo apt-get update -qq
# MAGIC sudo apt-get install -yqq sbt

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p ~/miniconda3
# MAGIC if [ ! -f ~/miniconda3/bin/conda ]; then
# MAGIC   wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# MAGIC   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# MAGIC   rm ~/miniconda3/miniconda.sh
# MAGIC fi

# COMMAND ----------

# MAGIC %sh
# MAGIC source ~/miniconda3/bin/activate
# MAGIC conda env create -f /local_disk0/glow/python/environment.yml || conda env update -f /local_disk0/glow/python/environment.yml
# MAGIC conda activate glow
# MAGIC pip install -U databricks-sdk

# COMMAND ----------

# MAGIC %sh
# MAGIC source ~/miniconda3/bin/activate
# MAGIC conda activate glow
# MAGIC export SPARK_VERSION="3.5.1"
# MAGIC export SCALA_VERSION="2.12.19"
# MAGIC cd /local_disk0/glow
# MAGIC ./bin/build --scala --python

# COMMAND ----------

import os, glob

lib_volume = f"/Volumes/{catalog}/{schema}/gwas_libraries"
os.makedirs(lib_volume, exist_ok=True)

jar_files = glob.glob("/local_disk0/glow/core/target/scala-2.12/glow-spark3-assembly-*.jar")
whl_files = glob.glob("/local_disk0/glow/python/dist/glow.py-*.whl")

import shutil
for f in jar_files + whl_files:
    dest = os.path.join(lib_volume, os.path.basename(f))
    shutil.copy2(f, dest)
    print(f"Copied {os.path.basename(f)} to {lib_volume}")

# COMMAND ----------

print("Glow libraries installed to UC Volume:")
for f in os.listdir(lib_volume):
    print(f"  {f}")

# Databricks notebook source
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("run_alphafold_job_id", "150680385258622", "AlphaFold Job ID")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
run_alphafold_job_id = dbutils.widgets.get("run_alphafold_job_id")


# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")

spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

spark.sql(f"""
INSERT INTO settings VALUES
('run_alphafold_job_id', '{run_alphafold_job_id}')
""")

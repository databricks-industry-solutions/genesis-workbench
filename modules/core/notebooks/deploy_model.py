# Databricks notebook source
#parameters to the notebook
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
fq_model_uc_name = dbutils.widgets.get("fq_model_uc_name")
model_version = dbutils.widgets.get("model_version")
workload_type = dbutils.widgets.get("workload_type")
workload_size = dbutils.widgets.get("workload_size")


# COMMAND ----------

print(f"catalog: {catalog}")
print(f"schema: {schema}")
print(f"fq_model_uc_name: {fq_model_uc_name}")
print(f"model_version: {model_version}")
print(f"workload_type: {workload_type}")
print(f"workload_size: {workload_size}")

# COMMAND ----------

assert catalog and schema, "Catalog and schema must be provided"
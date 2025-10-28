# Databricks notebook source
dbutils.widgets.text("sample_path", "", "Sample Path")
path = dbutils.widgets.get("sample_path")

# COMMAND ----------

path

# COMMAND ----------

file_info = dbutils.fs.ls(path)[0]
file_size = file_info.size
file_size_gb = file_size / (1024 ** 3)
dbutils.jobs.taskValues.set(key="file_size_gb", value=file_size_gb)

# COMMAND ----------

file_size_gb

# COMMAND ----------


# Databricks notebook source
# MAGIC %md
# MAGIC # Run this notebook first and follow the directions here.

# COMMAND ----------

# MAGIC %md
# MAGIC # Steps to deploy
# MAGIC 1. Run the next cell.
# MAGIC 1. Open web terminal.
# MAGIC 1. Run the output of the next cell in the web terminal.

# COMMAND ----------

import os
import subprocess
current_directory = os.getcwd()
print(f"{current_directory}/deploy.sh {current_directory}")

# COMMAND ----------



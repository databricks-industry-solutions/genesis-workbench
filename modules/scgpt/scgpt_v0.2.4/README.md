# MAGIC %md
# MAGIC Copyright (2025) Databricks, Inc.
# MAGIC
# MAGIC DB license
# MAGIC
# MAGIC Definitions.
# MAGIC
# MAGIC Agreement: The agreement between Databricks, Inc., and you governing the use of the Databricks Services, as that term is defined in the Master Cloud Services Agreement (MCSA) located at www.databricks.com/legal/mcsa.
# MAGIC
# MAGIC Licensed Materials: The source code, object code, data, and/or other works to which this license applies.
# MAGIC
# MAGIC Scope of Use. You may not use the Licensed Materials except in connection with your use of the Databricks Services pursuant to the Agreement. Your use of the Licensed Materials must comply at all times with any restrictions applicable to the Databricks Services, generally, and must be used in accordance with any applicable documentation. You may view, use, copy, modify, publish, and/or distribute the Licensed Materials solely for the purposes of using the Licensed Materials within or connecting to the Databricks Services. If you do not agree to these terms, you may not view, use, copy, modify, publish, and/or distribute the Licensed Materials.
# MAGIC
# MAGIC Redistribution. You may redistribute and sublicense the Licensed Materials so long as all use is in compliance with these terms. In addition:
# MAGIC
# MAGIC You must give any other recipients a copy of this License;
# MAGIC You must cause any modified files to carry prominent notices stating that you changed the files;
# MAGIC You must retain, in any derivative works that you distribute, all copyright, patent, trademark, and attribution notices, excluding those notices that do not pertain to any part of the derivative works; and
# MAGIC If a "NOTICE" text file is provided as part of its distribution, then any derivative works that you distribute must include a readable copy of the attribution notices contained within such NOTICE file, excluding those notices that do not pertain to any part of the derivative works.
# MAGIC You may add your own copyright statement to your modifications and may provide additional license terms and conditions for use, reproduction, or distribution of your modifications, or for any such derivative works as a whole, provided your use, reproduction, and distribution of the Licensed Materials otherwise complies with the conditions stated in this License.
# MAGIC
# MAGIC Termination. This license terminates automatically upon your breach of these terms or upon the termination of your Agreement. Additionally, Databricks may terminate this license at any time on notice. Upon termination, you must permanently delete the Licensed Materials and all copies thereof.
# MAGIC
# MAGIC DISCLAIMER; LIMITATION OF LIABILITY.
# MAGIC
# MAGIC THE LICENSED MATERIALS ARE PROVIDED “AS-IS” AND WITH ALL FAULTS. DATABRICKS, ON BEHALF OF ITSELF AND ITS LICENSORS, SPECIFICALLY DISCLAIMS ALL WARRANTIES RELATING TO THE LICENSED MATERIALS, EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, IMPLIED WARRANTIES, CONDITIONS AND OTHER TERMS OF MERCHANTABILITY, SATISFACTORY QUALITY OR FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. DATABRICKS AND ITS LICENSORS TOTAL AGGREGATE LIABILITY RELATING TO OR ARISING OUT OF YOUR USE OF OR DATABRICKS’ PROVISIONING OF THE LICENSED MATERIALS SHALL BE LIMITED TO ONE THOUSAND ($1,000) DOLLARS.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE LICENSED MATERIALS OR THE USE OR OTHER DEALINGS IN THE LICENSED MATERIALS.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download scgpt and Store Runtime Logic in MLflow
# MAGIC - Tested on DBR 15.4LTS ML (CPU or GPU)
# MAGIC - Note: Loading and using the model later requires a GPU (a small T4 GPU is sufficient)
# MAGIC
# MAGIC ### Steps:
# MAGIC - Define an **MLflow** PyFunc model that wraps the scgpt model
# MAGIC   - Add pre-processing of the model outputs to get standard PDB formatted string output
# MAGIC   - The model is downloaded from "https://github.com/bowang-lab/scGPT?tab=readme-ov-file", the whole-human 
# (recommended), Pretrained on 33 million normal human cells (https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y)
# MAGIC - Define the model signature and an input example
# MAGIC   - This allows others to easily see how to use the model later
# MAGIC - Log the model and register it in **Unity Catalog**
# MAGIC   - This allows us to easily control **governance of the model**
# MAGIC   - Set permissions for individual users on this model
# MAGIC - Serve the model on a small GPU serving endpoint
# MAGIC   - Enable **Inference Tables**: this allows auto-tracking of the model's inputs and outputs in a table
# MAGIC   - This is key for auditability

# Deployment Steps:
1. deploy the core steps using this command. Make sure you change to your `prefix` and `schema` before press enter otherwise it might overwrite others' asset.  
``` ./deploy.sh core dev --var="dev_user_prefix=yyang,core_catalog_name=genesis_workbench,core_schema_name=dev_yyang_genesis_workbench"```
   1. please first install `poetry` locally and make sure it is callable with the right `$PATH` setup locally.
   2. feed the right bionemo docker token env variable, e.g., `bionemo_docker_token=xxxxxxxxxxxxxxxx` to the `--var=""` part. 
   3. 
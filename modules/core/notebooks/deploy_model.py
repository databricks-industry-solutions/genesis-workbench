# Databricks notebook source
#some example data for reference
gwb_model_id = 1
input_adapter_str = """
from genesis_workbench.adapters import BaseAdapter

class MyInAdapter(BaseAdapter):
    def process(self, data):
        #data is gene_ids (array) 
        #model expects json with gene_ids and length
        return {
            "gene_id" : data,
            "length" : len(data)
        }
"""
output_adapter_str = """
from genesis_workbench.adapters import BaseAdapter

class MyOutAdapter(BaseAdapter):
    def process(self, data):
        #data is gene_embeddings (array) 
        #need to return a dict
        return {
            "embeddings" : data
        }
"""
sample_input_data_dict_as_json = """
{"data": [1.0, 2.0, 3.0, 4.0, 5.0],
 "type":"list"
}
"""

sample_params_as_json = """
{"index": "a", "num_embeddings": 10}
"""

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("gwb_model_id", str(gwb_model_id), "Model Id")
dbutils.widgets.text("deployment_name", "my_finetuned_deployment", "Deployment Name")
dbutils.widgets.text("deployment_description", "description", "Deployment Description")
dbutils.widgets.text("input_adapter_str", input_adapter_str, "Input Adapter Class Content")
dbutils.widgets.text("output_adapter_str", output_adapter_str, "Output Adapter Class Content")
dbutils.widgets.text("sample_input_data_dict_as_json", sample_input_data_dict_as_json, "Sample Input with Adapters")
dbutils.widgets.text("sample_params_as_json", sample_params_as_json, "Sample Params with Adapters")

dbutils.widgets.text("workload_type", "CPU", "Endpoint Workload Type")
dbutils.widgets.text("workload_size", "Medium", "Endpoint Workload Size")
dbutils.widgets.text("deploy_user", "a@b.com", "User Id")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")


# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if(lib.name.startswith("genesis_workbench")):
        gwb_library_path = lib.path.replace("dbfs:","")

print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

#parameters to the notebook

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
gwb_model_id = dbutils.widgets.get("gwb_model_id")
deployment_name = dbutils.widgets.get("deployment_name")
deployment_description = dbutils.widgets.get("deployment_description")
input_adapter_str = dbutils.widgets.get("input_adapter_str")
output_adapter_str = dbutils.widgets.get("output_adapter_str")
sample_input_data_dict_as_json = dbutils.widgets.get("sample_input_data_dict_as_json")
sample_params_as_json = dbutils.widgets.get("sample_params_as_json")

workload_type = dbutils.widgets.get("workload_type")
workload_size = dbutils.widgets.get("workload_size")
deploy_user = dbutils.widgets.get("deploy_user")


# COMMAND ----------

print(f"catalog: {catalog}")
print(f"schema: {schema}")
print(f"gwb_model_id: {gwb_model_id}")
print(f"deployment_name: {deployment_name}")
print(f"deployment_description: {deployment_description}")
print(f"workload_type: {workload_type}")
print(f"workload_size: {workload_size}")

# COMMAND ----------

from genesis_workbench.adapters import BaseAdapter, GWBModel
from genesis_workbench.models import get_latest_model_version, deploy_model_endpoint

def get_adapter_instance(class_str):
    adapter_instance = None

    if class_str and class_str != "none":
        namespace = {}
        exec(class_str, namespace)

        for key in namespace.keys():
            if key != "BaseAdapter" and key != "__builtins__":
                input_adapter_class = namespace[key]

                if issubclass(input_adapter_class, BaseAdapter):
                    adapter_instance = input_adapter_class()
    
    return adapter_instance


# COMMAND ----------

input_adapter_instance = get_adapter_instance(input_adapter_str)
output_adapter_instance = get_adapter_instance(output_adapter_str)


# COMMAND ----------

import json
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel
import mlflow.models.utils

def process_model_with_adapters(core_catalog_name:str,
                                core_schema_name:str,
                                deployment_id:int,
                                model_uc_name:str,
                                model_uc_version : int,
                                input_adapter_instance:BaseAdapter,
                                output_adapter_instance:BaseAdapter,
                                sample_input_data_json : str,
                                sample_params_as_json : str) -> (str,str):
    
    params_input = None
    if sample_params_as_json and sample_params_as_json != "none":
        params_input = json.loads(sample_params_as_json)
    
    pip_requirements = ["mlflow==2.22.0", "cloudpickle==3.0.0", "numpy==1.23.5", "pandas==1.5.3"]
    mlflow.set_registry_uri("databricks-uc")
    model_uri = f"models:/{model_uc_name}/{model_uc_version}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    #create a wrapper around the mlflow model
    wrapped_model = GWBModel(model=loaded_model,
                             input_adapter=input_adapter_instance,
                             output_adapter=output_adapter_instance)
    gwb_libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")

    for lib in gwb_libraries:
        pip_requirements.append(f"/Volumes/{catalog}/{schema}/libraries/{lib.name}")

    model_name = model_uc_name.split(".")[2]
    json_input = json.loads(sample_input_data_json)

    input_data = None
    if json_input["type"] == "dataframe":
       input_data = pd.DataFrame(**json_input["data"])
    elif json_input["type"] == "list":
       input_data = json_input["data"]
    else:
        raise Exception(f"Invalid input type: {json_input['type']}")

    #get the model signature
    output_data = wrapped_model.predict(context=None, model_input=input_data, params=params_input)

    print(f"output data: {output_data}")

    signature = infer_signature(input_data, output_data, params_input)
    print(f"Signature: {signature}")
    print(f"New pip requirements: {pip_requirements}")

    #register the model
    wrapped_model_name = f"{core_catalog_name}.{core_schema_name}.{model_name}_{deployment_id}_wrapped"    
    with mlflow.start_run():
        # Create model instance
        model = wrapped_model            
        # Log model with metadata
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model,
            signature=signature,
            input_example=input_data,
            registered_model_name=wrapped_model_name,
            pip_requirements = pip_requirements
        )
        model_version = get_latest_model_version(wrapped_model_name)
        model_uri = f"models:/{wrapped_model_name}/{model_version}"
        mlflow.models.utils.add_libraries_to_model(model_uri)
        new_model_version = get_latest_model_version(wrapped_model_name)
        return (wrapped_model_name, new_model_version)



# COMMAND ----------

#get model details
result_df = spark.sql(f"SELECT model_uc_name, model_uc_version, deployment_ids FROM {catalog}.{schema}.models WHERE model_id = {gwb_model_id}")

deploy_result = None
current_deployment_ids = []
model_deployed = False
deploy_id = -1
is_adapter = False

if result_df.count() > 0:
   #create a unique deploy id
   deploy_id = np.datetime64('now', 'ms').view('int64')
   print(f"Creating new deployment id: {deploy_id}")   
   #gather model details
   model_uc_name, model_uc_version, deployment_ids = result_df.limit(1).toPandas().values[0]
   print(f"Model found: {model_uc_name}, {model_uc_version}, {deployment_ids}")

   if input_adapter_instance or output_adapter_instance:
      #if any adapters are present, create a new wrapped instance
      model_uc_name, model_uc_version = process_model_with_adapters(
               core_catalog_name=catalog,
               core_schema_name=schema,
               deployment_id=deploy_id,
               model_uc_name=model_uc_name,
               model_uc_version=model_uc_version,
               input_adapter_instance = input_adapter_instance,
               output_adapter_instance=output_adapter_instance,
               sample_input_data_json=sample_input_data_dict_as_json,
               sample_params_as_json=sample_params_as_json)
      is_adapter = True

   print(f"Deploying model: {model_uc_name}, {model_uc_version}")

   current_deployment_ids = deployment_ids.split(",") if len(deployment_ids.strip()) > 0 else []
   current_deployment_ids.append(str(deploy_id))
   
   model_uri = f"models:/{model_uc_name}/{model_uc_version}"
   
   #deploy the model to model serving endpoint
   deploy_result = deploy_model_endpoint(catalog, schema, model_uc_name, model_uc_version, workload_type, workload_size)
   model_deployed = True
else:
    print("No model found to deploy")

# COMMAND ----------

if model_deployed:
    hostname = spark.conf.get("spark.databricks.workspaceUrl")
    #update the model deployment table
    spark.sql(f"""
        INSERT INTO {catalog}.{schema}.model_deployments(
            deployment_id,
            deployment_name,
            deployment_description,
            model_id,
            input_adapter,
            output_adapter,
            is_adapter,
            deploy_model_uc_name,
            deploy_model_uc_version,
            model_deployed_date,
            model_deployed_by,
            model_deploy_platform, 
            model_endpoint_name,
            model_invoke_url,
            is_active,
            deactivated_timestamp
        ) VALUES (
            {deploy_id},
            '{deployment_name}',
            '{deployment_description}',
            {gwb_model_id},
            '{input_adapter_str}',
            '{output_adapter_str}',
            {is_adapter},
            '{model_uc_name}',
            {model_uc_version},
            CURRENT_TIMESTAMP(),
            '{deploy_user}',
            'model_serving',
            '{deploy_result.config.served_entities[0].entity_name}',
            'https://{hostname}/serving-endpoints/{deploy_result.config.served_entities[0].entity_name}/invocations',
            true,
            NULL
        )
    """) 
else:
    print("No deployments made")


# COMMAND ----------

if model_deployed:
    #update model table
    spark.sql(f"""
            UPDATE {catalog}.{schema}.models SET 
                is_model_deployed = true,
                deployment_ids = '{','.join(current_deployment_ids)}'
            WHERE model_id = {gwb_model_id}
    """)
else:
    print("No deployments made")


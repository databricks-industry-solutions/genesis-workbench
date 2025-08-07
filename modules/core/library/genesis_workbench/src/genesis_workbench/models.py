import os
import mlflow
import time
import numpy as np
from mlflow import MlflowClient
from mlflow.exceptions import RestException
from mlflow.pyfunc import PythonModel
from mlflow.models.model import ModelInfo
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import StrEnum, auto
from datetime import datetime, timedelta
from typing import Union, List
import pandas as pd
import numpy as np
from databricks.sdk import WorkspaceClient
from databricks.sdk import errors
from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
        AutoCaptureConfigInput,
        ServingEndpointDetailed,
        ServingModelWorkloadType,
        EndpointTag
    )

from .workbench import (UserInfo, 
                        AppContext,
                        get_user_settings,
                         execute_select_query, 
                         execute_non_select_query,
                         execute_parameterized_inserts,
                         execute_workflow)
from .adapters import BaseAdapter, GWBModel


class ModelSource(StrEnum):
    UNITY_CATALOG = auto()
    PYPI = auto()
    HUGGINGFACE = auto()

class ModelCategory(StrEnum):
    SINGLE_CELL = auto()
    PROTEIN_STUDIES = auto()
    SMALL_MOLECULES = auto()

class ModelDeployPlatform(StrEnum):
    MODEL_SERVING = auto()
    EXTERNAL = auto()

class MLflowExperimentAccessException(Exception):
    """
    A custom exception for access issues to MLflow.
    """
    def __init__(self, message="Error accessing MLflow folder"):
        self.message = message
        super().__init__(self.message)

@dataclass
class GWBModelInfo:
    """Class that contains info about available models"""    
    model_id:int
    model_name : str
    model_display_name : str
    model_source_version : str
    model_origin : ModelSource 
    model_description_url : str #website to find more details about model
    model_uc_name : str
    model_uc_version : int
    model_owner : str
    model_category : ModelCategory
    model_input_schema : str
    model_output_schema : str
    model_params_schema : str
    model_added_by : str #id of user
    model_added_date : datetime
    is_model_deployed : bool
    deployment_ids: str
    is_active: bool 
    deactivated_timestamp : datetime

@dataclass
class ModelDeploymentInfo:
    """Class that contains information on model deployments"""
    deployment_id:int
    deployment_name: str
    deployment_description: str
    model_id:int
    input_adapter:str
    output_adapter:str 
    is_adapter:bool
    deploy_model_uc_name: str
    deploy_model_uc_version: str
    model_deployed_date : datetime
    model_deployed_by : str
    model_deploy_platform : ModelDeployPlatform
    model_endpoint_name: str    
    model_invoke_url : str
    is_active: bool 
    deactivated_timestamp : datetime

def set_mlflow_experiment(experiment_tag, user_email, host=None, token=None, shared=False): 

    user_settings = get_user_settings(user_email=user_email)

    if host and not host.startswith("https://"):
        host = f"https://{host}"    

    w = WorkspaceClient() if not token else WorkspaceClient(host=host, token=token, auth_type="pat")

    mlflow_experiment_base_path = ""

    if shared:
        mlflow_experiment_base_path = "Shared/dbx_genesis_workbench_models"
    else:
        mlflow_experiment_base_path = f"Users/{user_email}/{user_settings['mlflow_experiment_folder']}"
    
    try:
        w.workspace.mkdirs(f"/Workspace/{mlflow_experiment_base_path}")
        experiment_path = f"/{mlflow_experiment_base_path}/{experiment_tag}"
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_tracking_uri("databricks")
        experiment = mlflow.set_experiment(experiment_path)

        mlflow.set_experiment_tags({
            "used_by_genesis_workbench":"yes"
        })
        
        return experiment
    except RestException as e:
        if e.error_code=="RESOURCE_DOES_NOT_EXIST":
            raise MLflowExperimentAccessException("Error accessing the experiment location")
        else:
            raise e

def get_latest_model_version(model_name):
    client = MlflowClient()
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([int(model_version_info.version) for model_version_info in model_version_infos])

def get_available_models(model_category : ModelCategory) -> pd.DataFrame:
    
    """Gets all models that are available for deployment"""
    query = f"SELECT \
                model_id, model_name, model_display_name, model_source_version, model_uc_name, model_uc_version \
            FROM \
                {os.environ['CORE_CATALOG_NAME']}.{os.environ['CORE_SCHEMA_NAME']}.models \
            WHERE \
                model_category = '{str(model_category)}' AND is_active=true \
            ORDER BY model_id DESC "
    
    result_df = execute_select_query(query)
    return result_df

def get_deployed_models(model_category : ModelCategory)-> pd.DataFrame:
    """Gets all models that are available for deployment"""
    
    query = f"SELECT models.model_id as model_id, deployment_id, deployment_name, deployment_description, model_display_name, model_source_version, \
                concat(model_uc_name,'/',model_uc_version) as uc_name , model_endpoint_name \
            FROM \
                {os.environ['CORE_CATALOG_NAME']}.{os.environ['CORE_SCHEMA_NAME']}.model_deployments \
            INNER JOIN {os.environ['CORE_CATALOG_NAME']}.{os.environ['CORE_SCHEMA_NAME']}.models ON \
                models.model_id = model_deployments.model_id \
            WHERE \
                model_category = '{str(model_category)}' and model_deployments.is_active=true \
            ORDER BY deployment_id DESC "
    
    result_df = execute_select_query(query)
    return result_df

def insert_model_info(model_info : GWBModelInfo):
    """Register the model in GWB"""
    columns = []
    values = []
    params = []

    for key, value in asdict(model_info).items():
        columns.append(key)
        if isinstance(value, StrEnum):
            values.append(str(value))
        else:
            values.append(value)
        params.append("?")

    insert_sql = f"""
        INSERT INTO {os.environ['CORE_CATALOG_NAME']}.{os.environ['CORE_SCHEMA_NAME']}.models ({",".join(columns)}) 
        values ({",".join(params)})
        """
    execute_parameterized_inserts(insert_sql, [ values ])

def get_uc_model_info(model_uc_name_fq : str, model_uc_version:int) -> ModelInfo:
    mlflow.set_registry_uri("databricks-uc")
    model_uri = f"models:/{model_uc_name_fq}/{model_uc_version}"
    model_info = mlflow.models.get_model_info(model_uri)
    return model_info


def get_gwb_model_info(model_id:int)-> GWBModelInfo:
    """Method to get model details using id"""

    query = f"SELECT * FROM \
                {os.environ['CORE_CATALOG_NAME']}.{os.environ['CORE_SCHEMA_NAME']}.models \
            WHERE model_id = {model_id} "
        
    result_df = execute_select_query(query)

    model_info = result_df.apply(lambda row: GWBModelInfo(**row), axis=1).tolist()[0]
    return model_info

def get_gwb_model_deployment_info(deployment_id:int)-> ModelDeploymentInfo:
    """Method to get model deploymenmt details using id"""

    query = f"SELECT * FROM \
                {os.environ['CORE_CATALOG_NAME']}.{os.environ['CORE_SCHEMA_NAME']}.model_deployments \
            WHERE deployment_id = {deployment_id} "
        
    result_df = execute_select_query(query)

    model_deployment_info = result_df.apply(lambda row: ModelDeploymentInfo(**row), axis=1).tolist()[0]
    return model_deployment_info

def import_model_from_uc(user_email : str,
                        model_category : ModelCategory,
                        model_uc_name : str,
                        model_uc_version: int,                       
                        model_name: None,
                        model_source_version : None,
                        model_display_name:str = None, 
                        model_description_url:str = None                      
                      ) -> int:
    """Imports a UC model intp GWB"""    
    model_info = get_uc_model_info(model_uc_name, model_uc_version)
    #throws exception if not found 
    model_signature = model_info.signature.to_dict()

    if not model_name:
        model_name = model_uc_name.split('.')[2]
    
    if not model_display_name:
        model_display_name = model_name

    gwb_model_id = time.time_ns()

    gwb_model = GWBModelInfo(
        model_id = gwb_model_id, #will be ignored
        model_name = model_name,
        model_display_name = model_display_name,
        model_source_version = model_source_version,
        model_origin = ModelSource.UNITY_CATALOG,
        model_description_url = model_description_url, #website to find more details about model
        model_category = str(model_category),
        model_uc_name = model_uc_name,
        model_uc_version = model_uc_version,
        model_input_schema = model_signature.get("inputs"),
        model_output_schema = model_signature.get("outputs"),
        model_params_schema = model_signature.get("params",None),
        model_owner = "TBD",
        model_added_by = user_email, #id of user
        model_added_date = datetime.now(),
        is_model_deployed = False,
        deployment_ids = "",
        is_active = True,
        deactivated_timestamp=None
    )
    insert_model_info(gwb_model)
    return gwb_model_id


def deploy_model_endpoint(catalog_name: str,
                 schema_name : str,
                 dev_user_prefix:str,
                 fq_model_uc_name : str,
                 model_version: int,
                 workload_type: str,
                 workload_size:str,
                 creating_user_email:str) -> ServingEndpointDetailed:

    w = WorkspaceClient()

    model_name = fq_model_uc_name.split(".")[2]
    endpoint_name = f"gwb_{dev_user_prefix}_{model_name}_endpoint" if dev_user_prefix and dev_user_prefix.strip() != "" else f"gwb_{model_name}_endpoint"
    scale_to_zero = True

    served_entities = [
        ServedEntityInput(
            entity_name=fq_model_uc_name,
            entity_version=model_version,
            name=model_name,
            workload_type=ServingModelWorkloadType(workload_type),
            workload_size=workload_size,
            scale_to_zero_enabled=scale_to_zero,
        )
    ]
    auto_capture_config = AutoCaptureConfigInput(
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_name_prefix=f"{endpoint_name}_serving",
        enabled=True,
    )

    print(f"Checking if endpoint: {endpoint_name} exists")
    
    try:
        # try to update the endpoint if already have one
        existing_endpoint = w.serving_endpoints.get(endpoint_name)
        print(f"Updating existing endpoint {endpoint_name}")
        # may take some time to actually do the update
        endpoint_details = w.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=served_entities,
            auto_capture_config=auto_capture_config,
            timeout = timedelta(minutes=180)
        )
    except errors.platform.ResourceDoesNotExist as e:
        # if no endpoint yet, make it, wait for it to spin up, and put model on endpoint
        print(f"Creating new endpoint {endpoint_name}")
        endpoint_details = w.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                name=endpoint_name,
                served_entities=served_entities,
                auto_capture_config=auto_capture_config
            ),
            tags=[
                EndpointTag(key="application", value="genesis_workbench"),
                EndpointTag(key="created_by", value=creating_user_email)
            ],
            timeout = timedelta(minutes=180) #wait upto three hours. some models take very long
        )

    return endpoint_details

def deploy_model(user_email: str,
                 gwb_model_id:int,
                 deployment_name:str,
                 deployment_description:str,
                 input_adapter_str:str,
                 output_adapter_str,
                 sample_input_data_dict_as_json:str,
                 sample_params_as_json: str,
                 workload_type:str,
                 workload_size: str):
    print(f"Deploying model id: {gwb_model_id}")
    model_deploy_job_id = os.environ["DEPLOY_MODEL_JOB_ID"]
    params = {
        "gwb_model_id": gwb_model_id,
        "deployment_name" : deployment_name,
        "deployment_description" : deployment_description,
        'input_adapter_str': input_adapter_str,
        'output_adapter_str': output_adapter_str,
        'sample_input_data_dict_as_json': sample_input_data_dict_as_json,
        'sample_params_as_json': sample_params_as_json,
        "workload_type" : workload_type,
        "workload_size" : workload_size,
        "deploy_user": user_email
    }
    
    run_id = execute_workflow(model_deploy_job_id,params)
    return run_id

def delete_endpoint(core_catalog:str, core_schema:str, deployment_id:str):
    """Deletes the endpoint and archives the inference table"""
    
    print("==============================================")
    print(f"⏩️ Getting model details for deployment {deployment_id}")
    deployment_info : ModelDeploymentInfo = get_gwb_model_deployment_info(deployment_id)
    model_info : GWBModelInfo = get_gwb_model_info(deployment_info.model_id)

    model_id = deployment_info.model_id
    endpoint_name = deployment_info.model_endpoint_name

    w = WorkspaceClient()

    print(f"⏩️ Deleting endpoint {endpoint_name}")
    try:
        w.serving_endpoints.delete(name = endpoint_name)
    except Exception as e:
        print(f"Error deleting endpoint {endpoint_name}: {e}. Delete it manually")
    
    print(" ")

    inf_table_name = f"{endpoint_name}_serving_payload"
    print(f"⏩️ Archiving the inference table {inf_table_name}")
    execute_non_select_query(f"""
        ALTER TABLE {core_catalog}.{core_schema}.{inf_table_name} 
        RENAME TO {core_catalog}.{core_schema}.{inf_table_name}_bkup_{datetime.now().strftime('%Y%m%d%H%M%S')}
    """)
 
    print(" ")

    print(f"⏩️ Deactivating the deployment {deployment_id}")
    execute_non_select_query(f"""
        UPDATE {core_catalog}.{core_schema}.model_deployments SET 
          is_active = 'false',
          deactivated_timestamp = current_timestamp()

        WHERE deployment_id = {deployment_id}
    """)

    print(" ")

    print(f"⏩️ Removing deployment from model {model_id}")
    deployed_endpoint_ids = model_info.deployment_ids
    if deployed_endpoint_ids :
      deployed_endpoint_ids_arr = deployed_endpoint_ids.split(",")
      new_deployed_endpoint_ids = ",".join([x for x in deployed_endpoint_ids_arr if x != deployment_id])

      execute_non_select_query(f"""
          UPDATE {core_catalog}.{core_schema}.models SET 
            deployment_ids = '{new_deployed_endpoint_ids}'

          WHERE model_id = {model_id}
    """)
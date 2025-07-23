import os
import mlflow
from mlflow import MlflowClient
from mlflow.pyfunc import PythonModel
from mlflow.models.model import ModelInfo
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import StrEnum, auto
from datetime import datetime
from typing import Union, List
import pandas as pd
import numpy as np
from databricks.sdk import WorkspaceClient
from databricks.sdk import errors
from .workbench import (UserInfo, 
                        AppContext,
                         execute_select_query, 
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
    model_invoke_url : str
    is_active: bool 
    deactivated_timestamp : datetime

def set_mlflow_experiment(experiment_tag, user_email, host=None, token=None):     
    if host and not host.startswith("https://"):
        host = f"https://{host}"    

    w = WorkspaceClient() if not token else WorkspaceClient(host=host, token=token, auth_type="pat")

    mlflow_experiment_base_path = f"Users/{user_email}/mlflow_experiments"
    w.workspace.mkdirs(f"/Workspace/{mlflow_experiment_base_path}")

    experiment_path = f"/{mlflow_experiment_base_path}/{experiment_tag}"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    return mlflow.set_experiment(experiment_path)

def get_latest_model_version(model_name):
    client = MlflowClient()
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([int(model_version_info.version) for model_version_info in model_version_infos])

def get_available_models(model_category : ModelCategory,app_context:AppContext) -> pd.DataFrame:
    
    """Gets all models that are available for deployment"""
    query = f"SELECT \
                model_id, model_name, model_display_name, model_source_version, model_uc_name, model_uc_version \
            FROM \
                {app_context.core_catalog_name}.{app_context.core_schema_name}.models \
            WHERE \
                model_category = '{str(model_category)}' AND is_active=true \
            ORDER BY model_id DESC "
    
    print(query)
    result_df = execute_select_query(query)
    return result_df

def get_deployed_models(model_category : ModelCategory, app_context:AppContext)-> pd.DataFrame:
    """Gets all models that are available for deployment"""
    
    query = f"SELECT deployment_id, deployment_name, deployment_description, model_display_name, model_source_version, \
                concat(model_uc_name,'/',model_uc_version) as uc_name  \
            FROM \
                {app_context.core_catalog_name}.{app_context.core_schema_name}.model_deployments \
            INNER JOIN {app_context.core_catalog_name}.{app_context.core_schema_name}.models ON \
                models.model_id = model_deployments.model_id \
            WHERE \
                model_category = '{str(model_category)}' and model_deployments.is_active=true \
            ORDER BY deployment_id DESC "
    
    print(query)
    
    result_df = execute_select_query(query)
    return result_df

def insert_model_info(model_info : GWBModelInfo, app_context:AppContext):
    """Register the model in GWB"""
    columns = []
    values = []
    params = []

    for key, value in asdict(model_info).items():
        if key != "model_id":
            columns.append(key)
            if isinstance(value, StrEnum):
                values.append(str(value))
            else:
                values.append(value)
            params.append("?")

    # #delete any existing records    
    # if model_info.model_id != -1:
    #     delete_query = f"DELETE FROM {app_context.core_catalog_name}.{app_context.core_schema_name}.models \
    #                     WHERE model_id = {model_info.model_id}"
    #     execute_upsert_delete_query(delete_query)

    #insert the record
    insert_sql = f"""
        INSERT INTO {app_context.core_catalog_name}.{app_context.core_schema_name}.models ({",".join(columns)}) 
        values ({",".join(params)})
        """
    execute_parameterized_inserts(insert_sql, [ values ])

def get_uc_model_info(model_uc_name_fq : str, model_uc_version:int) -> ModelInfo:
    mlflow.set_registry_uri("databricks-uc")
    model_uri = f"models:/{model_uc_name_fq}/{model_uc_version}"
    model_info = mlflow.models.get_model_info(model_uri)
    return model_info


def get_gwb_model_info(model_id:int, app_context: AppContext)-> GWBModelInfo:
    """Method to get model details using id"""

    query = f"SELECT * FROM \
                {app_context.core_catalog_name}.{app_context.core_schema_name}.models \
            WHERE model_id = {model_id} "
        
    result_df = execute_select_query(query)

    model_info = result_df.apply(lambda row: GWBModelInfo(**row), axis=1).tolist()[0]
    return model_info

def import_model_from_uc(app_context: AppContext ,
                        user_email : str,
                        model_category : ModelCategory,
                        model_uc_name : str,
                        model_uc_version: int,                       
                        model_name: None,
                        model_source_version : None,
                        model_display_name:str = None, 
                        model_description_url:str = None                      
                      ):
    """Imports a UC model intp GWB"""    
    model_info = get_uc_model_info(model_uc_name, model_uc_version)
    #throws exception if not found 
    model_signature = model_info.signature.to_dict()

    if not model_name:
        model_name = model_uc_name.split('.')[2]
    
    if not model_display_name:
        model_display_name = model_name

    gwb_model = GWBModelInfo(
        model_id = -1, #will be ignored
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
    insert_model_info(gwb_model, app_context)


def deploy_model(user_info: UserInfo,
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
        "deploy_user": "a@b.com" if not user_info.user_email else user_info.user_email
    }
    
    run_id = execute_workflow(model_deploy_job_id,params)
    return run_id
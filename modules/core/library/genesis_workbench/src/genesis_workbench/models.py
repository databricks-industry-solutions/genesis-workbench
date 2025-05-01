import os
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.models.model import ModelInfo
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import StrEnum, auto
from datetime import datetime
from typing import Union, List
from databricks.sdk import WorkspaceClient

from databricks.sdk import errors
from .workbench import (UserInfo, 
                        get_user_info, 
                        get_app_context ,
                         execute_select_query, 
                         execute_upsert_delete_query, 
                         execute_parameterized_inserts,
                         execute_workflow)

class BaseAdapter(ABC):
    """Asbtract class for an input/output adapter"""
    @abstractmethod
    def process(self,data):
        return

class GWBModel(PythonModel, ABC):
    """This class that will wrap any pyfunc model with adapters"""
    def __init__(self, 
                 input_adapter:BaseAdapter,
                 output_adapter:BaseAdapter,
                 model:PythonModel):
        
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.model = model

    def predict(self, model_input, params):
        return self.output_adapter.process(
            self.model.predict(
                self.input_adapter.process(model_input)
            )
        )

class ModelSource(StrEnum):
    UNITY_CATALOG = auto()
    PYPI = auto()
    HUGGINGFACE = auto()

class ModelCategory(StrEnum):
    SINGLE_CELL = auto()
    PROTEIN_FOLDING = auto()
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


def get_available_models(model_category : ModelCategory):
    """Gets all models that are available for deployment"""
    app_context = get_app_context()
    query = f"SELECT \
                model_id, model_name, model_display_name, model_source_version, model_uc_name, model_uc_version \
            FROM \
                {app_context.core_catalog_name}.{app_context.core_schema_name}.models \
            WHERE \
                model_category = '{str(model_category)}' AND is_active=true"
    
    
    result_df = execute_select_query(query)
    return result_df

def get_deployed_models(model_category : ModelCategory):
    """Gets all models that are available for deployment"""
    app_context = get_app_context()
    
    query = f"SELECT deployment_id, deployment_name, deployment_description, model_display_name, model_source_version, \
                concat(model_uc_name,'/',model_uc_version) as uc_name  \
            FROM \
                {app_context.core_catalog_name}.{app_context.core_schema_name}.model_deployments \
            INNER JOIN {app_context.core_catalog_name}.{app_context.core_schema_name}.models ON \
                models.model_id = model_deployments.model_id \
            WHERE \
                model_category = '{str(model_category)}' and model_deployments.is_active=true"
    
    result_df = execute_select_query(query)
    return result_df

def insert_model_info(model_info : GWBModelInfo):
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

    app_context = get_app_context()

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


def get_gwb_model_info(model_id:int)-> GWBModelInfo:
    """Method to get model details using id"""
    app_context = get_app_context()
    query = f"SELECT * FROM \
                {app_context.core_catalog_name}.{app_context.core_schema_name}.models \
            WHERE model_id = {model_id} "
        
    result_df = execute_select_query(query)

    model_info = result_df.apply(lambda row: GWBModelInfo(**row), axis=1).tolist()[0]
    return model_info

def import_model_from_uc(model_category : ModelCategory,
                      model_uc_name : str,
                      model_uc_version: int,                       
                      model_name: None,
                      model_source_version : None,
                      model_display_name:str = None, 
                      model_description_url:str = None
                      ):
    """Imports a UC model intp GWB"""    
    user_info = get_user_info()
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
        model_added_by = user_info.user_email, #id of user
        model_added_date = datetime.now(),
        is_model_deployed = False,
        deployment_ids = "",
        is_active = True,
        deactivated_timestamp=None
    )
    insert_model_info(gwb_model)


def deploy_model(gwb_model_id:int, deployment_name:str, deployment_description: str, workload_type: str, workload_size: str):
    print(f"Deploying model id: {gwb_model_id}")
    user_info = get_user_info()
    model_deploy_job_id = os.environ["DEPLOY_MODEL_JOB_ID"]
    params = {
        "gwb_model_id": gwb_model_id,
        "deployment_name" : deployment_name,
        "deployment_description" : deployment_description,
        "workload_type" : workload_type,
        "workload_size" : workload_size,
        "deploy_user": "a@b.com" if not user_info.user_email else user_info.user_email
    }
    
    run_id = execute_workflow(model_deploy_job_id,params)
    return run_id
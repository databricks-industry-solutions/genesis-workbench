import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.models.model import ModelInfo
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import StrEnum, auto
from datetime import datetime
from typing import Union, List
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
        ServedModelInputWorkloadSize,
        ServedModelInputWorkloadType,
        AutoCaptureConfigInput,
    )
from databricks.sdk import errors
from .workbench import (UserInfo, 
                        get_user_info, 
                        get_app_context ,
                         execute_select_query, 
                         execute_upsert_delete_query, 
                         execute_parameterized_inserts)

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

@dataclass
class ModelDeploymentInfo:
    """Class that contains information on model deployments"""
    deployment_id:int
    model_id:int
    model_deployed_date : datetime
    model_deployed_by : str
    model_deploy_platform : ModelDeployPlatform
    model_invoke_url : str

def get_available_models(model_category : ModelCategory):
    """Gets all models that are available for deployment"""
    app_context = get_app_context()
    query = f"SELECT \
                model_id, model_name, model_display_name, model_source_version, model_uc_name, model_uc_version \
            FROM \
                {app_context.core_catalog_name}.{app_context.core_schema_name}.models \
            WHERE \
                is_model_deployed = false AND model_category = '{str(model_category)}' "
    
    
    result_df = execute_select_query(query)
    return result_df

def get_deployed_models(model_category : ModelCategory):
    """Gets all models that are available for deployment"""
    app_context = get_app_context()
    
    query = f"SELECT \
                model_id, model_display_name, model_source_version, model_uc_name,\
                      model_uc_version, deployment_ids\
            FROM \
                {app_context.core_catalog_name}.{app_context.core_schema_name}.models \
            WHERE \
                is_model_deployed = true AND model_category = '{str(model_category)}' "
    
    result_df = execute_select_query(query)
    return result_df

def upsert_model_info(model_info : GWBModelInfo):
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

    #delete any existing records    
    if model_info.model_id != -1:
        delete_query = f"DELETE FROM {app_context.core_catalog_name}.{app_context.core_schema_name}.models \
                        WHERE model_id = {model_info.model_id}"
        execute_upsert_delete_query(delete_query)

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
        deployment_ids = ""
    )
    upsert_model_info(gwb_model)


def deploy_model(fq_model_uc_name : str, model_version: int, workload_type: str, workload_size:str):

    w = WorkspaceClient()

    app_context = get_app_context()
    model_name = fq_model_uc_name.split(".")[2]
    endpoint_name = f"{fq_model_uc_name.replace('.','_')}"

    served_entities = [
        ServedEntityInput(
            entity_name=fq_model_uc_name,
            entity_version=model_version,
            name=model_name,
            workload_type=workload_type,
            workload_size=workload_size,
            scale_to_zero_enabled=True,
        )
    ]
    auto_capture_config = AutoCaptureConfigInput(
        catalog_name=app_context.core_catalog_name,
        schema_name=app_context.core_schema_name,
        table_name_prefix=f"{model_name}_serving",
        enabled=True,
    )

    try:
        # try to update the endpoint if already have one
        existing_endpoint = w.serving_endpoints.get(endpoint_name)
        # may take some time to actually do the update
        status = w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=served_entities,
            auto_capture_config=auto_capture_config,
        )
    except errors.platform.ResourceDoesNotExist as e:
        # if no endpoint yet, make it, wait for it to spin up, and put model on endpoint
        status = w.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                name=endpoint_name,
                served_entities=served_entities,
                auto_capture_config=auto_capture_config,
            ),
        )
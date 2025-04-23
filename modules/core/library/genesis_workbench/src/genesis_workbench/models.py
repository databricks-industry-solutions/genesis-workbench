import mlflow
from mlflow.pyfunc import PythonModel
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from typing import Union, List
from .workbench import UserInfo, get_app_context , execute_select_query, execute_upsert_delete_query

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

class ModelSource(Enum):
    UNITY_CATALOG = 1
    PYPI = 2
    HUGGING_FACE = 3

class ModelDeployPlatform(Enum):
    MODEL_SERVING = 1
    EXTERNAL = 2

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
    model_input_schema : str
    model_output_schema : str
    model_params_schema : str
    model_added_by : str #id of user
    model_added_date : datetime
    is_model_deployed : bool
    model_deployed_date : datetime
    model_deployed_by : str
    model_deploy_platform : ModelDeployPlatform
    model_invoke_url : str

def get_available_models():
    """Gets all models that are available for deployment"""
    app_context = get_app_context()
    query = f"SELECT \
                model_id, model_name, model_display_name, model_source_version, model_uc_name, model_uc_version \
            FROM \
                {app_context.core_catalog_name}.{app_context.core_schema_name}.models \
            WHERE \
                is_model_deployed = false"
    result_df = execute_select_query(query)
    return result_df

def get_deployed_models():
    """Gets all models that are available for deployment"""
    app_context = get_app_context()
    query = f"SELECT \
                model_id, model_display_name, model_source_version, model_uc_name, model_uc_version, date_format(model_deployed_date,'MMM dd,yyyy') as deployed_date \
            FROM \
                {app_context.core_catalog_name}.{app_context.core_schema_name}.models \
            WHERE \
                is_model_deployed = true"
    result_df = execute_select_query(query)
    return result_df

def upsert_model_info(model_info : GWBModelInfo):
    """Register the model in GWB"""
    columns = []
    values = []
    
    for key, value in asdict(model_info).items():
        if key != "model_id":
            columns.append(key)
            values.append(value)

    app_context = get_app_context()

    #delete any existing records
    if model_info.model_id == -1:
        delete_query = f"DELETE FROM {app_context.core_catalog_name}.{app_context.core_schema_name}.models \
                        WHERE model_id = {model_info.model_id}"
        execute_upsert_delete_query(delete_query)

    #insert the record
    insert_query = f"INSERT INTO {app_context.core_catalog_name}.{app_context.core_schema_name}.models \
                ( { ','.join(columns) })  \
            VALUES  \
                ({ ','.join(values) }) "
    
    execute_upsert_delete_query(insert_query)


def import_model_from_uc(model_uc_name : str,
                      model_uc_version: int, 
                      user_info: UserInfo,
                      model_name: None,
                      model_source_version : None,
                      model_display_name:str = None, 
                      model_description_url:str = None
                      ):
    """Imports a UC model intp GWB"""    
    app_context = get_app_context()
    model_info = mlflow.models.get_model_info(f"models://{model_uc_name}/{model_uc_version}")
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
        model_uc_name = model_uc_name,
        model_uc_version = model_uc_version,
        model_input_schema = model_signature.get("inputs"),
        model_output_schema = model_signature.get("outputs"),
        model_params_schema = model_signature.get("params",None),
        model_owner = model_info.created_by,
        model_uc_added_by = user_info.user_email, #id of user
        model_uc_added_date = datetime.now(),
        is_model_deployed = False,
        model_deployed_date = None,
        model_deployed_by = None,
        model_deploy_platform = None,
        model_invoke_url = None
    )

    upsert_model_info(gwb_model)
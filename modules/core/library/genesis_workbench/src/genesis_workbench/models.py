import mlflow
from mlflow.pyfunc import PythonModel
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class BaseInputAdapter(ABC):
    """Asbtract class for an input adapter"""
    @abstractmethod
    def process(self,data):
        return
        
class BaseOutputAdapter(ABC):
    """Asbtract class for an output adapter"""
    @abstractmethod
    def process(self,data):
        return

class GWBModel(PythonModel, ABC):
    """This class that will wrap any pyfunc model with adapters"""
    def __init__(self, 
                 input_adapter:BaseInputAdapter,
                 output_adapter:BaseOutputAdapter,
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
    model_uc_added_by : str #id of user
    model_uc_added_date : datetime
    is_model_deployed : bool
    model_deployed_date : datetime
    model_deployed_by : str
    model_deploy_platform : ModelDeployPlatform
    model_invoke_url : str

    
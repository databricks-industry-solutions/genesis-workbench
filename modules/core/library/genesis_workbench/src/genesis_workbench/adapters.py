import os
import mlflow
from mlflow.pyfunc import PythonModel
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

#This module is required to keep the MLFlow serving dependencies thin

class BaseAdapter(ABC):
    """Asbtract class for an input/output adapter"""
    @abstractmethod
    def process(self,data):
        return

class GWBModel(PythonModel):
    """This class that will wrap any pyfunc model with adapters"""
    def __init__(self, 
                 input_adapter:BaseAdapter,
                 output_adapter:BaseAdapter,
                 model:PythonModel):
        
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.model = model

    def predict(self, context, model_input, params):        
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.to_dict(orient="records")
        elif isinstance(model_input, np.ndarray):
            model_input = model_input[:, 0].tolist()
        elif isinstance(model_input, str):
            model_input = [model_input]
        
        proc_input = self.input_adapter.process(model_input)
        model_out = self.model.predict(data=proc_input, params=params)
        proc_out = self.output_adapter.process(model_out)

        return proc_out
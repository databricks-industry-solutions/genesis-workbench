
import os
import pandas as pd
from dataclasses import dataclass, asdict
from enum import StrEnum, auto
from datetime import datetime
from typing import Union, List
from .workbench import (UserInfo, 
                        AppContext,
                         execute_select_query, 
                         execute_parameterized_inserts,
                         execute_workflow)

class BionemoModelType(StrEnum):
    ESM2 = auto()

@dataclass
class GWBBionemoFTInfo:
    """Class that contains info about available models"""    
    ft_id: int
    ft_label:str
    model_type: BionemoModelType
    variant:str
    experiment_name:str
    run_id:str
    weights_volume_location:str
    created_by:str
    created_datetime:datetime
    is_active:bool
    deactivated_timestamp:datetime

def get_variants(model_type: BionemoModelType) -> List[str]:
    if model_type == BionemoModelType.ESM2:
        return ["650M","3B"]

def start_finetuning(user_info: UserInfo,
            esm_variant:str,
            train_data_volume_location:str,
            validation_data_volume_location:str,
            should_use_lora:bool,
            finetune_label:str,
            experiment_name:str,
            task_type:str,
            num_steps:int,
            micro_batch_size:int,
            precision:str,
            mlp_ft_dropout:float,
            mlp_hidden_size:int,
            mlp_target_size:int,
            mlp_lr:float,
            mlp_lr_multiplier:float):
    
    print(f"Starting a finetune run with label: {finetune_label}")
    bionemo_finetune_job_id = os.environ["BIONEMO_ESM_FINETUNE_JOB_ID"]
    params = {
        "user_email": "a@b.com" if not (user_info and user_info.user_email) else user_info.user_email,
        "esm_variant" :esm_variant,
        "train_data_volume_location" : train_data_volume_location,
        "validation_data_volume_location" : validation_data_volume_location,
        "should_use_lora" : "true" if should_use_lora else "false",
        "finetune_label" : finetune_label,
        "task_type": task_type,
        "mlp_ft_dropout" : mlp_ft_dropout,
        "mlp_hidden_size" : mlp_hidden_size,
        "mlp_target_size" : mlp_target_size,
        "experiment_name" : experiment_name,
        "num_steps" : num_steps,
        "lr" : mlp_lr,
        "lr_multiplier" : mlp_lr_multiplier,
        "micro_batch_size" : micro_batch_size,
        "precision" : precision
    }
    print(params)
    run_id = execute_workflow(bionemo_finetune_job_id,params)
    return run_id


def list_finetuned_weights(model_type: BionemoModelType, app_context: AppContext) -> pd.DataFrame:
    """Gets all finetuned model weight details for a model type"""
    
    query = f"SELECT ft_id, ft_label, model_type, variant, experiment_name, run_id, created_by, created_datetime \
            FROM \
                {app_context.core_catalog_name}.{app_context.core_schema_name}.bionemo_weights \
            WHERE \
                model_type = '{str(model_type)}' and is_active=true"
    
    result_df = execute_select_query(query)
    return result_df

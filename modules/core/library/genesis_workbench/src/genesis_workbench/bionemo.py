
import os
from databricks.sdk import WorkspaceClient
from .workbench import (UserInfo, 
                         execute_select_query, 
                         execute_parameterized_inserts,
                         execute_workflow)

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
import os
import streamlit as st
from genesis_workbench.bionemo import start_finetuning
from utils.streamlit_helper import get_user_info, get_app_context, open_run_window


st.title(":material/genetics: BioNeMo - ESMFold2")

settings_tab, finetune_tab, inference_tab = st.tabs(["Settings","Fine Tune", "Inference"])

with finetune_tab:
    with st.form("finetune_esm_form", enter_to_submit=False ):
        c1,c2,c3 = st.columns([1,1,1], vertical_alignment="top")
        with c1:
            esm_variant = st.selectbox("ESM Variant",["650M","3B"])
            train_data = st.text_input("Train Data Folder (UC Volume):")
            evaluation_data = st.text_input("Evaluation Data Folder (UC Volume):")
            should_use_lora =  st.toggle("Use LoRA?") 
            finetune_label = st.text_input("Finetuning Label:")
            experiment_name = st.text_input("Experiment Name:")
            start_finetune_button_clicked = st.form_submit_button("Start Finetuning Run")

        with c2:
            task_type = st.selectbox("Task Type:",["regression","classification"])
            mlp_num_steps = st.number_input("Number of steps:", value=50)
            micro_batch_size = st.number_input("Micro Batch Size:", value=2)
            mlp_precision = st.selectbox("Precision:",["bf16-mixed","fp16","bf16","fp32","fp32-mixed","16-mixed","fp16-mixed"])
        
        with c3:
            with st.expander("Advanced Parameters:", expanded=False):
                mlp_ft_dropout = st.number_input("Dropout:", value=0.25)
                mlp_hidden_size = st.number_input("Hidden Size:", value=256)
                mlp_target_size = st.number_input("Target Size:", value=1)
                mlp_lr = st.number_input("Learning Rate:", value=5e-3)
                mlp_lr_multiplier = st.number_input("Learning Rate Multiplier:", value=1e2)

    finetuning_started = False
    if start_finetune_button_clicked:

        with st.spinner("Launching finetuning job"):
            try:
                run_id = start_finetuning(user_info=get_user_info(),
                        esm_variant= esm_variant,
                        train_data_volume_location=train_data,
                        validation_data_volume_location=evaluation_data,
                        should_use_lora=should_use_lora,
                        finetune_label=finetune_label,
                        experiment_name=experiment_name,
                        task_type=task_type,
                        num_steps=mlp_num_steps,
                        micro_batch_size=int(micro_batch_size),
                        precision=mlp_precision,
                        mlp_ft_dropout=mlp_ft_dropout,
                        mlp_hidden_size=int(mlp_hidden_size),
                        mlp_target_size=int(mlp_target_size),
                        mlp_lr=mlp_lr,
                        mlp_lr_multiplier=mlp_lr_multiplier)

                finetuning_started = True
            except Exception as e:
                print(e)
                st.error("Error launching finetuning job.")                
                finetuning_started = False

        if finetuning_started:
            st.success(f"Finetuning run has started with a run id {run_id}.")       
            job_id = os.getenv("BIONEMO_ESM_FINETUNE_JOB_ID")
            view_deploy_run_btn = st.button("View Run", on_click=lambda: open_run_window(job_id,run_id))         

                    

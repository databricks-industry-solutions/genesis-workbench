import os
import streamlit as st
from genesis_workbench.bionemo import BionemoModelType, get_variants, start_finetuning, list_finetuned_weights
from utils.streamlit_helper import get_user_info, get_app_context, open_run_window



def display_finetune_tab():
    with st.form("finetune_esm_form", enter_to_submit=False ):
        c1,c2,c3 = st.columns([1,1,1], vertical_alignment="top")
        with c1:
            esm_variant = st.selectbox("ESM Variant",get_variants(BionemoModelType.ESM2))
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


def display_inference_tab():
    st.markdown("###### Run Inference")       
    use_ft_weight =  st.toggle("Use Finetuned Weight") 

    if not use_ft_weight:
        c1,c2 = st.columns([1,1])
        with c1:
            esm_variant_for_inference = st.selectbox("ESM Variant",get_variants(BionemoModelType.ESM2))

    else:
        st.write("Select a finetuned weight:")
        if len(finetuned_esm_weights_df) > 0:
            st.dataframe(finetuned_esm_weights_df, 
                            use_container_width=True,
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="single-row")
        else:
            st.write("There are no finetuned weights available")

    col1,col2 = st.columns([1,1])
    with col1:
        with st.form("run_bionemo_ft_inference_form", enter_to_submit=False ):
            st.text_input("Data Location:","")
            st.text_input("Result Schema Name:","")
            st.text_input("Result Table Prefix:","")
            
            st.form_submit_button("Run Inference")


#load data for page
with st.spinner("Loading data"):
    if "finetuned_esm_weights_df" not in st.session_state:
            finetuned_esm_weights_df = list_finetuned_weights(model_type=BionemoModelType.ESM2, app_context=get_app_context())

            finetuned_esm_weights_df.columns = ["Label", "Model type", "Variant", "MLflow Experiment","Run Id", "Weights Location", "Created By" , "Create On"]
            # available_models_df["model_labels"] = (available_models_df["model_id"].astype(str) + " - " 
            #                                     + available_models_df["model_display_name"].astype(str) + " [ " 
            #                                     + available_models_df["model_uc_name"].astype(str) + " v"
            #                                     + available_models_df["model_uc_version"].astype(str) + " ]"
            #                                     )
            st.session_state["finetuned_esm_weights_df"] = finetuned_esm_weights_df

    finetuned_esm_weights_df = st.session_state["finetuned_esm_weights_df"]


st.title(":material/genetics: BioNeMo - ESMFold2")

settings_tab, finetune_tab, inference_tab = st.tabs(["Settings","Fine Tune", "Inference"])

with finetune_tab:
    display_finetune_tab()

with inference_tab:
    display_inference_tab()

                    

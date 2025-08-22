import os
import streamlit as st
from genesis_workbench.bionemo import (BionemoModelType, 
                                       get_variants, 
                                       start_esm2_finetuning,
                                       start_esm2_inference,
                                       list_finetuned_weights)
from utils.streamlit_helper import get_user_info, open_run_window



def display_finetune_tab():
    with st.form("finetune_esm_form", enter_to_submit=False ):
        c1,c2,c3 = st.columns([1,1,1], vertical_alignment="top")
        with c1:
            esm_variant = st.selectbox("ESM Variant",get_variants(BionemoModelType.ESM2))
            train_data = st.text_input("Train Data (UC Volume Path *.csv):", " ", help="A CSV file with `sequence` column and a `target` column")
            evaluation_data = st.text_input("Evaluation Data (UC Volume Path *.csv):", help="A CSV file with `sequence` column and a `target` column")
            should_use_lora =  st.toggle("Use LoRA?") 
            finetune_label = st.text_input("Finetuning Label:", " ", help="A unique name for the finetuning run, so that you can refer it later")
            experiment_name = st.text_input("Experiment Name:", " ", help="An MLflow experiment name to track the run")
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
                train_data = train_data.strip()
                evaluation_data = evaluation_data.strip()
                finetune_label = finetune_label.strip()
                experiment_name = experiment_name.strip()

                if (train_data.endswith(".csv") and 
                    evaluation_data.endswith(".csv") and 
                    train_data.startswith("/Volumes") and
                    evaluation_data.startswith("/Volumes") 
                ):
                    if finetune_label != "" and experiment_name != "":
                        run_id = start_esm2_finetuning(user_info=get_user_info(),
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
                    else:
                        st.error("Provide a finetune label and an MLflow experiment name to track the run.")                
                        finetuning_started = False        
                else:
                    st.error("Train data and validation data must be CSV files in a UC Volume.")                
                    finetuning_started = False    
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

    c1,c2 = st.columns([1,2])
    with c1:
        esm_variant_for_inference = st.selectbox("ESM Variant",get_variants(BionemoModelType.ESM2))

    model_weight_source =  st.pills("Model:",["Base Model","Finetuned Weight"], 
                                    selection_mode="single",
                                    default="Base Model") 

    with st.form("run_bionemo_ft_inference_form", enter_to_submit=False ):    
        if model_weight_source != "Base Model":
            st.write("Select a finetuned weight:")
            if len(finetuned_esm_weights_df) > 0:
                infer_ft_weights= st.dataframe(finetuned_esm_weights_df, 
                                use_container_width=True,
                                hide_index=True,
                                on_select="rerun",
                                selection_mode="single-row")
            else:
                st.write("There are no finetuned weights available")

        else:
            infer_ft_weights = None

        col1,col2 = st.columns([1,1])
        with col1:
            inf_task_type = st.selectbox("Task Type:",["regression","classification"])
            inf_data_location = st.text_input("Data Location:(UC Volume Path *.csv):","", help="A CSV file with `sequence` column")
            inf_sequence_column_name = st.text_input("Sequence Column Name:","", help="The column containing the sequence in the csv file")
            inf_result_location = st.text_input("Result Location: (UC Volume Folder)","", help="Results will be saved as results.csv in the given folder. Please make sure the folder exists.")
            start_inference_button_clicked = st.form_submit_button("Run Inference")
        
    inference_started = False
    if start_inference_button_clicked:
        with st.spinner("Launching inference job"):
            try:
                is_base_model = True if model_weight_source=="Base Model" else False,
                selected_rows = []
                
                if infer_ft_weights:
                    selected_rows = infer_ft_weights.selection.rows

                ft_run_id = 0
                if len(selected_rows) > 0:
                    ft_run_id = finetuned_esm_weights_df.iloc[selected_rows]["Id"].iloc[0].item()

                if is_base_model or (not is_base_model and len(selected_rows) > 0):
                    if inf_data_location.endswith(".csv") and inf_data_location.strip().startswith("/Volumes") :
                        if inf_result_location.strip() != "" and inf_result_location.strip().startswith("/Volumes") :
                            if inf_sequence_column_name.strip() != "" :
                                
                                run_id = start_esm2_inference(user_info=get_user_info(),
                                        esm_variant= esm_variant_for_inference, 
                                        is_base_model = is_base_model,                                       
                                        finetune_run_id = ft_run_id,
                                        task_type=inf_task_type,
                                        data_volume_location=inf_data_location,
                                        sequence_column_name=inf_sequence_column_name,
                                        result_location= inf_result_location)

                                inference_started = True
                            else:
                                st.error("Provide the name of the column that has the sequence to infer on.")                
                                inference_started = False        
                        
                        else:
                            st.error("Provide a valid UC Volume location to store the result.")                
                            inference_started = False 
                    else:
                        st.error("Data must be a CSV file in a UC Volume.")                
                        inference_started = False 
                else:
                    st.error("Select a fine tune weight.") 
                    inference_started = False    
            except Exception as e:
                print(e)
                st.error("Error launching finetuning job.")                
                inference_started = False

        if inference_started:
            st.success(f"Inference run has started with a run id {run_id}.")       
            job_id = os.getenv("BIONEMO_ESM_INFERENCE_JOB_ID")
            view_inference_run_btn = st.button("View Run", on_click=lambda: open_run_window(job_id,run_id))     


#load data for page
with st.spinner("Loading data"):
    if "finetuned_esm_weights_df" not in st.session_state:
            finetuned_esm_weights_df = list_finetuned_weights(model_type=BionemoModelType.ESM2)[["ft_id", "ft_label", "model_type", "variant", "created_by"]]

            finetuned_esm_weights_df.columns = ["Id", "Label", "Model type", "Variant", "Created By"]

            st.session_state["finetuned_esm_weights_df"] = finetuned_esm_weights_df

    finetuned_esm_weights_df = st.session_state["finetuned_esm_weights_df"]


st.title(":material/genetics: BioNeMo")

esm2_tab, geneformer_tab = st.tabs(["ESM2", "Geneformer"])

#with settings_tab:
#settings_esm2_tab = st.tabs(["ESM2"])


with esm2_tab:

    finetune_tab, inference_tab = st.tabs(["Fine Tune", "Inference"])

    with finetune_tab:
        display_finetune_tab()

    with inference_tab:
        display_inference_tab()

                    
